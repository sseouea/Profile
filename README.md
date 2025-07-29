import os
import numpy as np
from tqdm import tqdm

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    DDIMScheduler,
    UniPCMultistepScheduler
)

# from peft import (
#     LoraConfig,
#     get_peft_model
# )

# from accelerate import Accelerator

from loss import *


def train(config, writer, train_loader):
    # Get config
    # fsdp_plugin = FSDPPlugin(sharing_strategy="FULL_SHARD")
    # accelerator = Accelerator(mixed_precision="fp16")
    # print("device check:", accelerator.device)

    # config["device"] = accelerator.device
    if config["device"] == "cuda":
        config["device"] = "cuda:0"
    device = config["device"]
    print("Device check:", device)

    lr = config["lr"]
    num_epochs = config["epoch"]
    save_every = config["save_every"]
    save_best = config["save_best"]
    output_path = config["output_path"]
    folder_name = config["folder_name"]
    
    # Load model
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_inpaint",
        torch_dtype=torch.float16
        )

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
        ).to(device)
    
    # peft_config = LoraConfig(
    #     r=2,
    #     lora_alpha=8,
    #     lora_dropout=0.1,
    #     bias="none",
    #     task_type="INPAINT",
    #     target_modules=[
    #         "mid_block.attn2.to_q",
    #         "mid_block.attn2.to_v",
    #         ]
    #     )
    # pipe.unet = get_peft_model(pipe.unet, peft_config)

    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing(slice_size="auto")

    # Set mode
    unet = pipe.unet.train()
    vae = pipe.vae.eval()
    controlnet = pipe.controlnet.eval()
    text_encoder = pipe.text_encoder.eval()
    tokenizer = pipe.tokenizer
    
    # Schuduler, Optimizer
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    # scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(scheduler.config.num_train_timesteps)
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=lr)
    criterion = OvenLoss(config)

    # unet, optimizer, train_loader = accelerator.prepare(unet, optimizer, train_loader)

    best_loss = float("inf")
    global_step = 0
    for epoch in range(num_epochs):
        for batch in tqdm(train_loader, desc=f"[Epoch {epoch+1}]"):
            # Data
            image = batch["image"].to(device)       # [B, 3, H, W]
            mask = batch["mask"].to(device)         # [B, 1, H, W]
            control = batch["control"].to(device)   # [B, 3, H, W]
            prompts = batch["prompt"]               # list[str]
            prompts = [prompts] if isinstance(prompts, str) else prompts
            names = batch["name"]                   # list[str]
            names = [names] if isinstance(names, str) else names

            # Text embedding
            with torch.no_grad():
                inp = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt"
                    ).to(device)
                text = text_encoder(inp.input_ids.to(device))[0]

            # Image, Noise Latent encoding
            """
            mask: preserve(0), inpaiting(1)
            origin_latents: used for preservation
            masked_latents: used for inpainting
            """

            with torch.no_grad():
                origin_latents = vae.encode(image).latent_dist.sample() * pipe.vae.config.scaling_factor

                masked_img = image * (1.0 - mask)
                masked_latents = vae.encode(masked_img).latent_dist.sample() * pipe.vae.config.scaling_factor

            # Add noise
            down = mask.shape[-1] // origin_latents.shape[-1]
            mask_latent = F.interpolate(mask, scale_factor=1/down, mode="nearest") # downscaling mask to resolution of latent (vae)
            mask_latent = mask_latent.to(device)
            latent_input = masked_latents * mask_latent + origin_latents  * (1 - mask_latent) 
            
            noise = torch.randn_like(origin_latents)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (origin_latents.shape[0],),
                device=device
                ).long()
            noisy_latents = scheduler.add_noise(latent_input, noise, timesteps).to(device) # revised on ver 2.0

            # Forward
            down_res, mid_res = controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text,
                controlnet_cond=control,
                return_dict=False
            )

            out = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res
            )
            noise_pred = out.sample # backward possible

            #noise_pred.sum().backward() #okay
            #print("noise")

            # Denoise
            # out = scheduler.step(
            #     model_output=noise_pred,   
            #     timestep=timesteps.tolist(),        
            #     sample=noisy_latents,
            #     return_dict=True
            # )

            # pred_x0 = out.pred_original_sample  # Tensor [B, C, h, w]

            #noise_pred.sum().backward() okay
            #print("noise")

            alphas = scheduler.alphas_cumprod.to(device)
            alpha_t = alphas[timesteps].view(-1,1,1,1)
            beta_t = 1 - alpha_t
            pred_x0 = (noisy_latents - beta_t.sqrt() * noise_pred) / alpha_t.sqrt()
            pred_x0 = pred_x0.half()

            #noise_pred.sum().backward()
            #print("noise")

            decoded = vae.decode(pred_x0 / pipe.vae.config.scaling_factor).sample

            noise_pred.sum().backward()
            print("noise")

            decode_image = (decoded / 2 + 0.5).clamp(0, 1).to(device)  # [B,3,H,W] in [0,1]

            noise_pred.sum().backward()
            print("noise")

            
            # Debug (Inverse)
            # img_np = decode_image.detach().cpu().permute(0, 2, 3, 1).numpy()
            # img_np = (img_np * 255).round().astype("uint8") 
            # img_pil = [Image.fromarray(_img_np) for _img_np in img_np]
            # img_pil[0].save(f"{output_path}/{folder_name}/{epoch}_{names[0]} (decoded image).png")
            
            # Loss
            noise_pred.sum().backward()
            print("noise")

            loss_mse = F.mse_loss(noise_pred * mask_latent, noise * mask_latent)

            noise_pred.sum().backward()
            print("noise")

            mask_latent.sum().backward()
            print("mask")

            (noise_pred * mask_latent).sum().backward()
            print("pred")
            
            loss_mse.backward()
            print("mse")

            ret = [criterion(f'train/{n}', out) for n, out in zip(names, decode_image)]
            # loss_pcd = torch.stack(ret).mean()
            loss_pcd = torch.stack([l[0] for l in ret]).mean()
            loss_dir = torch.stack([l[1] for l in ret]).mean()
            loss_con = torch.stack([l[2] for l in ret]).mean()

            loss = loss_mse + loss_pcd
            loss = loss.to(device)
            losses = {
                "MSE": loss_mse.item(),
                "PCD": loss_pcd.item(),
                "DIR": loss_dir.item(),
                "CON": loss_con.item(),
                "Total": loss.item()
            }
            print(f'[Epoch {epoch+1:2d}] MSE: {losses["MSE"]:.4f}\tPCD: {losses["PCD"]:.4f}\tDIR: {losses["DIR"]:.4f}\tCON: {losses["CON"]:.4f}\tTotal: {losses["Total"]:.4f}')

            # Tensorboard logging
            writer.add_scalar("Loss/total", loss.item(), global_step)
            for key, val in losses.items():
                if key != "total":
                    writer.add_scalar(f"Loss/{key}", val, global_step)
            global_step += 1
            
            # Backward
            optimizer.zero_grad()
            # accelerator.backward(loss)
            loss.backward()
            optimizer.step()

            # Save
            if epoch == 1 or (epoch + 1) % save_every == 0:
                pipe.save_pretrained(f"{output_path}/{folder_name}/pipe/base_dir/{epoch}_{losses['Total']:.4f}")
                # pipe.unet.load_attn_procs(f"{output_path}/{folder_name}/pipe/lora_adapter/{epoch}_{losses['Total']:.4f}")
                torch.save({
                    "epoch": epoch,
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "config": config
                }, f"{output_path}/{folder_name}/pipe/{epoch}")

            if save_best and losses["Total"] < best_loss:
                best_loss = losses["Total"]
                pipe.save_pretrained(f"{output_path}/{folder_name}/best_pipe/base_dir/{epoch}_{losses['Total']:.4f}")
                # pipe.unet.load_attn_procs(f"{output_path}/{folder_name}/best_pipe/lora_adapter/{epoch}_{losses['Total']:.4f}")
                torch.save({
                    "epoch": epoch,
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "config": config
                }, f"{output_path}/{folder_name}/pipe/{epoch}")
