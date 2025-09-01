import torch
from diffusers import StableDiffusion3Pipeline, FluxPipeline
from PIL import Image
import argparse
import random
import numpy as np
import yaml
import os
import shutil
from models.DVRF import DVRF

def concatenate_images(image1, image2):
    """Concatenate two images side by side."""
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Ensure both images have the same height
    new_height = max(height1, height2)
    new_width = width1 + width2

    concatenated_image = Image.new("RGB", (new_width, new_height))
    concatenated_image.paste(image1, (0, 0))
    concatenated_image.paste(image2, (width1, 0))

    return concatenated_image


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_number", type=int, default=0, help="device number to use")
    parser.add_argument("--exp_yaml", type=str, default="exp.yaml", help="experiment yaml file")
    parser.add_argument("--exp_name", type=str, help="override exp_name")
    parser.add_argument("--eta", type=float, default=1.0, help="override eta")
    parser.add_argument("--num_steps", type=int, help="override num_steps")
    parser.add_argument("--src_guidance_scale", type=float, help="override src_guidance_scale")
    parser.add_argument("--tgt_guidance_scale", type=float, help="override tgt_guidance_scale")

    args = parser.parse_args()

    # Set device
    device_number = args.device_number
    device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")

    # Load experiment YAML file
    exp_yaml = args.exp_yaml
    if not os.path.exists(exp_yaml):
        raise FileNotFoundError(f"Experiment YAML file not found: {exp_yaml}")
    
    with open(exp_yaml) as file:
        exp_configs = yaml.load(file, Loader=yaml.FullLoader)

    # Apply CLI overrides to each experiment config
    overrides = {
        "exp_name": args.exp_name,
        "eta": args.eta,
        "num_steps": args.num_steps,
        "src_guidance_scale": args.src_guidance_scale,
        "tgt_guidance_scale": args.tgt_guidance_scale,
    }
    for exp in exp_configs:
        for key, val in overrides.items():
            if val is not None:
                exp[key] = val

    model_type = exp_configs[0]["model_type"]

    # Load model
    if model_type == 'FLUX':
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
    elif model_type == 'SD3':
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
    elif model_type.startswith('SD3.5'):
        SD35_MODELS = {
            'large': 'stabilityai/stable-diffusion-3.5-large',
            'large-turbo': 'stabilityai/stable-diffusion-3.5-large-turbo',
            'medium': 'stabilityai/stable-diffusion-3.5-medium'
        }
        # Handle SD 3.5 variants
        if model_type == 'SD3.5':
            # Default to medium if no specific variant is specified
            model_variant = 'medium'
        else:
            # Extract variant from model_type (e.g., 'SD3.5-large', 'SD3.5-large-turbo')
            model_variant = model_type.split('-', 1)[1] if '-' in model_type else 'medium'

        if model_variant not in SD35_MODELS:
            raise ValueError(f"Unknown SD 3.5 variant: {model_variant}. Available variants: {list(SD35_MODELS.keys())}")
        
        model_id = SD35_MODELS[model_variant]
        print(f"Loading SD 3.5 model: {model_id}")
        pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")
    
    scheduler = pipe.scheduler
    pipe = pipe.to(device)

    for exp_dict in exp_configs:
        exp_name = exp_dict["exp_name"]
        T_steps = exp_dict["T_steps"]
        B = exp_dict["B"]
        src_guidance_scale = exp_dict["src_guidance_scale"]
        tgt_guidance_scale = exp_dict["tgt_guidance_scale"]
        num_steps = exp_dict["num_steps"]
        seed = exp_dict["seed"]
        eta=exp_dict["eta"]
        scheduler_strategy=exp_dict["scheduler_strategy"]
        lr=exp_dict["lr"]
        optim=exp_dict["optimizer"]

        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        dataset_yaml = exp_dict["dataset_yaml"]
        if not os.path.exists(dataset_yaml):
            raise FileNotFoundError(f"Dataset YAML file not found: {dataset_yaml}")
            
        with open(dataset_yaml) as file:
            dataset_configs = yaml.load(file, Loader=yaml.FullLoader)
        for data_dict in dataset_configs:
            src_prompt = data_dict["source_prompt"]
            tgt_prompts = data_dict["target_prompts"]
            image_src_path = data_dict["input_img"]

            # Check if source image exists
            if not os.path.exists(image_src_path):
                raise FileNotFoundError(f"Source image not found: {image_src_path}")

            # Load source image
            image = Image.open(image_src_path).convert("RGB")
            image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
            image_src = pipe.image_processor.preprocess(image)

            image_src = image_src.to(device).half()

            with torch.autocast("cuda"), torch.inference_mode():
                x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()

            x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            x0_src = x0_src.to(device)
            for tgt_num, tgt_prompt in enumerate(tgt_prompts):

                if model_type == 'SD3' or model_type.startswith('SD3.5'):
                    print(src_prompt, tgt_prompt)
                    x0_tgt, _, trajectories = DVRF(pipe, scheduler, x0_src, src_prompt, tgt_prompt, "", T_steps, B, src_guidance_scale, tgt_guidance_scale, num_steps, eta, scheduler_strategy, lr, optim)

                else:
                    raise NotImplementedError(f"Sampler type {model_type} not implemented") 

                
                x0_tgt_denorm = (x0_tgt / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                with torch.autocast("cuda"), torch.inference_mode():
                    image_tgt = pipe.vae.decode(x0_tgt_denorm, return_dict=False)[0]

                image_tgt = pipe.image_processor.postprocess(image_tgt)[0]

                # Concatenate source and generated image
                concatenated_image = concatenate_images(image, image_tgt)

                src_prompt_txt = data_dict["input_img"].split("/")[-1].split(".")[0]
                tgt_prompt_txt = str(tgt_num)

                # Ensure directory exists
                save_dir = f"outputs/{exp_name}/{model_type}/src_{src_prompt_txt}/tgt_{tgt_prompt_txt}"
                os.makedirs(save_dir, exist_ok=True)

                # Save concatenated image
                output_filename = f"{save_dir}/{lr}_eta_{eta}_{scheduler_strategy}{optim}T_steps_{T_steps}_num_steps_{num_steps}_cfg_enc_{src_guidance_scale}_cfg_dec{tgt_guidance_scale}_seed{seed}.png"
                concatenated_image.save(output_filename)

                # Save intermediate trajectory images every 5 steps
                if trajectories is not None and len(trajectories) > 0:
                    trajectory_dir = f"{save_dir}/trajectories"
                    
                    # Delete the trajectory directory if it already exists
                    if os.path.exists(trajectory_dir):
                        shutil.rmtree(trajectory_dir)
                    
                    # Create a fresh trajectory directory
                    os.makedirs(trajectory_dir, exist_ok=True)
                    
                    for i, traj_latent in enumerate(trajectories):
                        if i % 4 == 0 or i == len(trajectories) - 1:  # Save every 5th image and the last one
                            traj_latent_denorm = (traj_latent / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                            with torch.autocast("cuda"), torch.inference_mode():
                                traj_image = pipe.vae.decode(traj_latent_denorm, return_dict=False)[0]
                            
                            traj_image = pipe.image_processor.postprocess(traj_image)[0]
                            traj_filename = f"{trajectory_dir}/step_{i:03d}.png"
                            traj_image.save(traj_filename)

                # Save source and target prompts
                with open(f"{save_dir}/prompts.txt", "w") as f:
                    f.write(f"Source prompt: {src_prompt}\n")
                    f.write(f"Target prompt: {tgt_prompt}\n")
                    f.write(f"Seed: {seed}\n")
                    f.write(f"Sampler type: {model_type}\n")

    print("Done")
