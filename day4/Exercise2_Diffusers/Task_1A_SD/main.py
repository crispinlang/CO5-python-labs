import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import modal
import torch
import io

# 1. DEFINE THE ENVIRONMENT (The "Docker Container")
# We start with a slim Linux image and install the libraries we need.
image = (
    modal.Image.debian_slim()
    .pip_install("diffusers", "transformers", "torch", "accelerate", "Pillow")
)

# 2. CACHE THE MODELS
# We define a helper function to download models during the "build" phase.
# This ensures the 4GB+ models are baked into the image and don't download every time you run.
def download_models():
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

    repo_id = "CompVis/stable-diffusion-v1-4"
    
    # Just trigger the downloads to cache them
    AutoencoderKL.from_pretrained(repo_id, subfolder="vae")
    UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet")
    CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    PNDMScheduler.from_pretrained(repo_id, subfolder="scheduler")

# Attach the download step to the image
image = image.run_function(download_models)

# 3. DEFINE THE APP
app = modal.App(name="stable-diffusion-manual", image=image)

# 4. THE REMOTE FUNCTION (Runs on the Cloud GPU)
# We request a GPU (gpu="any" gives you a T4 or A10, typically sufficient)
@app.function(gpu="any", timeout=600)
def run_stable_diffusion():
    # --- YOUR ORIGINAL IMPORTS ---
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
    from PIL import Image
    from tqdm.auto import tqdm

    # Set device to CUDA (Modal always provides NVIDIA GPUs)
    device = "cuda"

    # --- YOUR ORIGINAL SETUP CODE ---
    repo_id = "CompVis/stable-diffusion-v1-4"
    
    vae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    scheduler = PNDMScheduler.from_pretrained(repo_id, subfolder="scheduler")


    # Parameters
    prompt = "A stunning panoramic view of Zurich, Switzerland. The Limmat river flowing through the historic Old Town (Altstadt), iconic twin towers of the Grossmünster cathedral, ancient stone bridges, guild houses along the water quay."
    batch_size = 1
    height = 512
    width = 512
    steps = 50
    guidance_scale = 7.5

    # --- YOUR ORIGINAL PIPELINE LOGIC ---
    
    # 1. Encode Text
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    # Replace your current uncond_input line with this:
    negative_prompt = "low quality, bad quality, blurry, pixelated, low resolution, watermark, text, signature, distorted, ugly, flat lighting, oversaturated"

    uncond_input = tokenizer(
        negative_prompt, 
        padding="max_length", 
        max_length=tokenizer.model_max_length, 
        return_tensors="pt"
    )

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # 2. Latents
    latents = torch.randn((batch_size, unet.config.in_channels, height // 8, width // 8), device=device)
    latents = latents * scheduler.init_noise_sigma

    # 3. Denoising Loop
    scheduler.set_timesteps(steps)
    for t in tqdm(scheduler.timesteps):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # 4. Decode
    latents = latents / 0.18215
    with torch.no_grad():
        image = vae.decode(latents).sample

    # 5. Process Image
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")
    pil_image = Image.fromarray(image[0])

    # --- RETURN THE RESULT ---
    # We convert the image to bytes to send it back to your local computer
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

# 5. LOCAL ENTRYPOINT (Runs on your computer)
@app.local_entrypoint()
def main():
    print("Sending script to Modal...")
    image_bytes = run_stable_diffusion.remote()
    
    output_filename = "zurich_neg_prompt1.png"
    with open(output_filename, "wb") as f:
        f.write(image_bytes)
    
    print(f"Done! Image saved to {output_filename}")