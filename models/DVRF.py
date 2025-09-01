from typing import Optional, Tuple, Union
import torch
from tqdm import tqdm
import numpy as np

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps


def lr_hump_beta(k: int, N: int, alpha_max: float,
                 a: float = 3.0, b: float = 6.0) -> float:
    """
    Compute learning rate using beta distribution hump shape.
    
    Args:
        k: Current step (1-indexed)
        N: Total number of steps
        alpha_max: Maximum learning rate value
        a: Beta distribution shape parameter
        b: Beta distribution shape parameter
        
    Returns:
        Learning rate value for step k
    """
    if not (1 <= k <= N):
        raise ValueError("k must be in [1, N]")
    x = (k - 1) / (N - 1)
    pdf = x**(a - 1) * (1 - x)**(b - 1)
    peak = ((a - 1) / (a + b - 2))**(a - 1) * ((b - 1)/(a + b - 2))**(b - 1)
    return alpha_max * pdf / peak


def lr_hump_tail_beta(k: int, N: int, alpha_max: float, beta: float,
                      a: float = 3.0, b: float = 6.0) -> float:
    """
    Compute learning rate using beta distribution hump with linear tail.
    
    Args:
        k: Current step (1-indexed)
        N: Total number of steps
        alpha_max: Maximum learning rate value
        beta: Linear tail coefficient
        a: Beta distribution shape parameter
        b: Beta distribution shape parameter
        
    Returns:
        Learning rate value for step k
    """
    x = (k - 1) / (N - 1)
    hump = lr_hump_beta(k, N, alpha_max - beta, a, b)  # Same shape but reduced amplitude
    tail = beta * x
    return hump + tail


def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Forward process in flow-matching.
    
    Args:
        scheduler: Diffusion scheduler
        sample: Input sample tensor
        timestep: Current timestep
        noise: Optional noise tensor
        
    Returns:
        Scaled sample tensor
    """
    scheduler._init_step_index(timestep)
    sigma = scheduler.sigmas[scheduler.step_index]
    sample = sigma * noise + (1.0 - sigma) * sample
    return sample


def calc_v_sd3(pipe, src_tgt_latent_model_input: torch.Tensor, 
               src_tgt_prompt_embeds: torch.Tensor, 
               src_tgt_pooled_prompt_embeds: torch.Tensor, 
               src_guidance_scale: float, tgt_guidance_scale: float, 
               t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate velocity predictions for SD3 model.
    
    Args:
        pipe: Diffusion pipeline
        src_tgt_latent_model_input: Concatenated source and target latent inputs
        src_tgt_prompt_embeds: Concatenated prompt embeddings
        src_tgt_pooled_prompt_embeds: Concatenated pooled prompt embeddings
        src_guidance_scale: Source guidance scale
        tgt_guidance_scale: Target guidance scale
        t: Timestep tensor
        
    Returns:
        Tuple of (source_velocity, target_velocity)
    """
    # Broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(src_tgt_latent_model_input.shape[0])
    
    with torch.no_grad():
        # Predict the noise for the source and target prompts
        noise_pred_src_tgt = pipe.transformer(
            hidden_states=src_tgt_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tgt_prompt_embeds,
            pooled_projections=src_tgt_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        # Perform guidance
        if pipe.do_classifier_free_guidance:
            src_noise_pred_uncond, src_noise_pred_text, tgt_noise_pred_uncond, tgt_noise_pred_text = noise_pred_src_tgt.chunk(4)
            noise_pred_src = src_noise_pred_uncond + src_guidance_scale * (src_noise_pred_text - src_noise_pred_uncond)
            noise_pred_tgt = tgt_noise_pred_uncond + tgt_guidance_scale * (tgt_noise_pred_text - tgt_noise_pred_uncond)

    return noise_pred_src, noise_pred_tgt


def DVRF(
    pipe,
    scheduler,
    x_src: torch.Tensor,
    src_prompt: str,
    tgt_prompt: str,
    negative_prompt: str,
    T_steps: int = 50,
    B: int = 1,
    src_guidance_scale: float = 6,
    tgt_guidance_scale: float = 16.5,
    num_steps: int = 50,
    eta: float = 1.0,
    scheduler_strategy: str = "descending",
    lr: Union[float, str] = 0.02,
    optim: str = 'SGD',
) -> Tuple[torch.Tensor, list, list]:
    """
    DVRF text-to-image optimization for SD3 and SD3.5 models.
    
    Args:
        pipe: Diffusion pipeline
        scheduler: Diffusion scheduler
        x_src: Source latent tensor
        src_prompt: Source text prompt
        tgt_prompt: Target text prompt
        negative_prompt: Negative text prompt
        T_steps: Number of diffusion timesteps
        B: Batch size for averaging
        src_guidance_scale: Source guidance scale
        tgt_guidance_scale: Target guidance scale
        num_steps: Number of optimization steps
        eta: Eta parameter for trajectory modification
        scheduler_strategy: Strategy for timestep scheduling ("random" or "descending")
        lr: Learning rate (float or string for adaptive)
        optim: Optimizer type
        
    Returns:
        Tuple of (optimized_latent, velocities, trajectories)
    """
    zt_edit = x_src.float().clone().requires_grad_(True)
    
    # Initialize optimizer
    if optim == 'SGD':
        if type(lr) == float:
            optimizer = torch.optim.SGD([zt_edit], lr=lr)
        else:
            optimizer = torch.optim.SGD([zt_edit], lr=0.02)
    elif optim == 'SGD_Nesterov':
        optimizer = torch.optim.SGD([zt_edit], lr=lr, momentum=0.9, nesterov=True)
    elif optim == 'RMSprop':
        optimizer = torch.optim.RMSprop([zt_edit], lr=lr, alpha=0.9)
    elif optim == 'AdamW':
        optimizer = torch.optim.AdamW([zt_edit], lr=lr)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam([zt_edit], lr=lr)
    else:
        raise ValueError(f'Optimizer {optim} not supported.')
    
    device = x_src.device
    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)
    pipe._num_timesteps = len(timesteps)
    
    # Prompt encoding
    pipe._guidance_scale = src_guidance_scale
    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )
    
    pipe._guidance_scale = tgt_guidance_scale
    (
        tgt_prompt_embeds,
        tgt_negative_prompt_embeds,
        tgt_pooled_prompt_embeds,
        tgt_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tgt_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )
    
    src_tgt_prompt_embeds = torch.cat(
        [src_negative_prompt_embeds, src_prompt_embeds, tgt_negative_prompt_embeds, tgt_prompt_embeds], dim=0
    )
    src_tgt_pooled_prompt_embeds = torch.cat(
        [
            src_negative_pooled_prompt_embeds,
            src_pooled_prompt_embeds,
            tgt_negative_pooled_prompt_embeds,
            tgt_pooled_prompt_embeds,
        ],
        dim=0,
    )
    
    # Initialize tracking variables
    velocities = []
    trajectories = [zt_edit.detach().clone()]
    alpha_T_steps = (timesteps[T_steps-2]/1000 - timesteps[T_steps-1] / 1000) / 1.6
    alpha_max, beta = alpha_T_steps / 1.6, alpha_T_steps / 4
    
    # Optimization loop
    if scheduler_strategy == "random":
        pbar = tqdm(range(num_steps), desc="DVRF Optimization (Random)", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for i in pbar:
            V_delta_avg = torch.zeros_like(x_src)
            for k in range(B):
                ind = torch.randint(2, T_steps - 1, (1,)).item()
                t = timesteps[ind]
                t_i = t / 1000
                alpha_i = 2.2 * lr_hump_tail_beta(i+1, T_steps+28, alpha_max, beta, a=10, b=8)
                eta_i = eta * i / T_steps
                
                fwd_noise = torch.randn_like(x_src, device=device)
                zt_src = (1 - t_i) * x_src + t_i * fwd_noise
                c_t_i = eta_i * t_i
                zt_tgt = (1 - t_i) * zt_edit + t_i * fwd_noise + c_t_i * (zt_edit - x_src)  # Eq. 6
                src_tgt_latent_model_input = (
                    torch.cat([zt_src, zt_src, zt_tgt, zt_tgt])
                    if pipe.do_classifier_free_guidance
                    else (zt_src, zt_tgt)
                )
                
                # Use inference mode and cast latent input to half precision
                with torch.inference_mode():
                    src_tgt_latent_model_input_fp16 = src_tgt_latent_model_input.half()
                    Vt_src, Vt_tgt = calc_v_sd3(
                        pipe,
                        src_tgt_latent_model_input_fp16,
                        src_tgt_prompt_embeds,
                        src_tgt_pooled_prompt_embeds,
                        src_guidance_scale,
                        tgt_guidance_scale,
                        t,
                    )
                V_delta_avg += (Vt_tgt - Vt_src) / B
            
            current_lr = alpha_i
            if type(lr) == str:
                optimizer.param_groups[0]['lr'] = current_lr
            
            # Update progress bar with current learning rate
            pbar.set_postfix({'lr': f'{current_lr:.6f}', 'eta': f'{eta_i:.3f}'})
            
            velocities.append(V_delta_avg)
            grad = V_delta_avg + (1 - eta_i) * (zt_edit - x_src)  # Eq. 8
            loss = (zt_edit * grad.detach()).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trajectories.append(zt_edit.detach().clone())
    else:  # descending
        # Filter timesteps for descending strategy
        active_timesteps = [(i, t) for i, t in enumerate(timesteps) if T_steps - i <= num_steps]
        
        pbar = tqdm(active_timesteps, desc="DVRF Optimization (Descending)", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for i, t in pbar:
            t_i = t / 1000
            alpha_i = 2.2 * lr_hump_tail_beta(i+1, T_steps+28, alpha_max, beta, a=10, b=8)
            eta_i = eta * i / T_steps
            V_delta_avg = torch.zeros_like(x_src)
            
            for k in range(B):
                fwd_noise = torch.randn_like(x_src, device=device)
                zt_src = (1 - t_i) * x_src + t_i * fwd_noise
                zt_tgt = (1 - t_i) * zt_edit + t_i * fwd_noise + eta_i * t_i * (zt_edit - x_src)  # Eq. 6
                src_tgt_latent_model_input = (
                    torch.cat([zt_src, zt_src, zt_tgt, zt_tgt])
                    if pipe.do_classifier_free_guidance
                    else (zt_src, zt_tgt)
                )
                
                with torch.inference_mode():
                    src_tgt_latent_model_input_fp16 = src_tgt_latent_model_input.half()
                    Vt_src, Vt_tgt = calc_v_sd3(
                        pipe,
                        src_tgt_latent_model_input_fp16,
                        src_tgt_prompt_embeds,
                        src_tgt_pooled_prompt_embeds,
                        src_guidance_scale,
                        tgt_guidance_scale,
                        t,
                    )
                V_delta_avg += (Vt_tgt - Vt_src) / B
            
            current_lr = alpha_i
            if type(lr) == str:
                optimizer.param_groups[0]['lr'] = current_lr
            
            # Update progress bar with current timestep and learning rate
            pbar.set_postfix({'step': i, 't': f'{t_i:.3f}', 'lr': f'{current_lr:.6f}', 'eta': f'{eta_i:.3f}'})
            
            velocities.append(V_delta_avg)
            grad = V_delta_avg + (1 - eta_i) * (zt_edit - x_src)  # Eq. 8
            # loss = 0.5 * grad.pow(2).sum()  # Equivalent loss
            loss = (zt_edit * grad.detach()).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trajectories.append(zt_edit.detach().clone())
    
    return zt_edit, velocities, trajectories
