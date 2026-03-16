"""
STEP 4: Stable Diffusion 1.5 + LoRA Fine-tuning & Generation
Fine-tunes SD 1.5 on each minority breed using LoRA and generates 500 images.

Requirements:
    pip install diffusers transformers accelerate peft torch torchvision
    # xformers not needed — attention slicing + VAE slicing used instead

Hardware: NVIDIA GPU 6–8 GB VRAM | ~45 mins per class
"""

import os
import gc
import json
import time
import math
import itertools
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model

# ── Breed type lookup (Oxford-IIIT Pet has both cats and dogs) ───────────────
CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
    "Siamese", "Sphynx"
}

def get_animal_type(breed: str) -> str:
    """Returns 'cat' or 'dog' based on breed name."""
    # Normalise breed name for lookup
    normalised = breed.replace(" ", "_")
    return "cat" if normalised in CAT_BREEDS else "dog"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID       = "stable-diffusion-v1-5/stable-diffusion-v1-5"
RESULTS_DIR    = Path("results")
SPLIT_DIR      = Path("data/oxford_pet/splits")
SYNTH_DIR      = Path("data/synthetic/sd_lora")
LORA_DIR       = Path("models/sd_lora")
IMG_SIZE       = 512          # SD 1.5 native resolution
N_GENERATE     = 500
TRAIN_STEPS    = 1000         # LoRA trains fast; 800–1200 steps is optimal for ~30–50 images
BATCH_SIZE     = 1            # Must be 1 for 6 GB VRAM at 512px
GRAD_ACCUM     = 4            # effective batch = 4
LR             = 1e-4
LORA_RANK      = 8
RANDOM_SEED    = 42
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Dataset ───────────────────────────────────────────────────────────────────
class BreedLoRADataset(Dataset):
    """Pairs each image with a text prompt describing the breed."""
    def __init__(self, image_paths: list, breed: str, tokenizer, size=512):
        self.paths     = image_paths
        self.tokenizer = tokenizer
        self.size      = size
        # Trigger word approach — unique token [V] maps to the breed
        # Note: Oxford-IIIT has both cats and dogs — using "pet" is safer
        animal = get_animal_type(breed)
        self.prompt    = f"a photo of a {breed} {animal}, high quality, realistic pet photography"
        self.tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img    = Image.open(self.paths[idx]).convert("RGB")
        pixel  = self.tf(img)
        tokens = self.tokenizer(
            self.prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        return {"pixel_values": pixel, "input_ids": tokens}

# ── LoRA training ─────────────────────────────────────────────────────────────
def train_lora(breed: str, image_paths: list, save_dir: Path):
    print(f"\n  LoRA fine-tuning for breed: {breed} ({len(image_paths)} images)")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load base models
    tokenizer   = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder").to(DEVICE)
    vae          = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").to(DEVICE)
    unet         = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet").to(DEVICE)
    noise_sched  = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Apply LoRA to UNet attention layers only
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Enable gradient checkpointing to save VRAM
    unet.enable_gradient_checkpointing()

    # Dataset & loader
    ds     = BreedLoRADataset(image_paths, breed, tokenizer, size=IMG_SIZE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)  # num_workers=0 faster on Windows

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=LR,
        weight_decay=1e-2
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_STEPS)

    unet.train()
    loader_iter = iter(loader)
    losses = []

    with tqdm(range(1, TRAIN_STEPS + 1), desc=f"  LoRA [{breed}]") as pbar:
        for step in pbar:
            # Accumulate gradients
            optimizer.zero_grad()
            acc_loss = 0.0

            for _ in range(GRAD_ACCUM):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    batch = next(loader_iter)

                pixel_vals = batch["pixel_values"].to(DEVICE, dtype=torch.float32)
                input_ids  = batch["input_ids"].to(DEVICE)

                # Encode images to latent space
                with torch.no_grad():
                    latents = vae.encode(pixel_vals).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Add noise
                noise     = torch.randn_like(latents)
                bsz       = latents.size(0)
                timesteps = torch.randint(0, noise_sched.config.num_train_timesteps,
                                         (bsz,), device=DEVICE).long()
                noisy_lat = noise_sched.add_noise(latents, noise, timesteps)

                # Text conditioning
                with torch.no_grad():
                    enc_hidden = text_encoder(input_ids)[0]

                # Predict noise (AMP float16 for speed)
                with torch.amp.autocast('cuda'):
                    noise_pred = unet(noisy_lat, timesteps, enc_hidden).sample
                    # MSE loss
                    loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
                loss = loss / GRAD_ACCUM
                acc_loss += loss.item()
                loss.backward()

            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, unet.parameters()), 1.0)
            optimizer.step()
            lr_scheduler.step()

            losses.append(acc_loss)
            pbar.set_postfix(loss=f"{acc_loss:.4f}")

            if step % 200 == 0:
                avg = np.mean(losses[-50:])
                print(f"    step {step}/{TRAIN_STEPS} | avg_loss={avg:.4f}")

    # Save LoRA weights — use save_pretrained on the peft model
    # This saves adapter_config.json + adapter_model.safetensors
    unet.save_pretrained(save_dir / "unet_lora")
    print(f"  ✓ LoRA weights saved to {save_dir / 'unet_lora'}")
    # Verify saved files
    saved = list((save_dir / "unet_lora").glob("*"))
    print(f"  Saved files: {[f.name for f in saved]}")

    # Free VRAM
    del unet, vae, text_encoder
    gc.collect()
    torch.cuda.empty_cache()


# ── Generation ────────────────────────────────────────────────────────────────
def generate_images_sd(breed: str, lora_path: Path, out_dir: Path, n: int = 500):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Generating {n} SD+LoRA images for {breed}...")

    # Load pipeline with LoRA
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None,    # disabled for research use
    ).to(DEVICE)

    # Load LoRA weights using PEFT (works with both .bin and .safetensors)
    # Load LoRA weights using PEFT (handles .safetensors format)
    from peft import PeftModel
    pipe.unet = PeftModel.from_pretrained(pipe.unet, str(lora_path / "unet_lora"))
    pipe.enable_attention_slicing(1)   # slice size 1 = maximum VRAM saving
    pipe.enable_vae_slicing()          # saves ~1GB VRAM during decoding
    pipe.enable_vae_tiling()           # extra safety for 8GB cards
    # xformers not used — incompatible with torch 2.7.1+cu118
    # The above three settings provide equivalent VRAM savings

    animal  = get_animal_type(breed)
    prompts = [
        f"a high quality photo of a {breed} {animal}, natural lighting, sharp focus",
        f"a realistic photograph of a {breed} {animal} outdoors",
        f"a {breed} {animal} portrait, professional photo, highly detailed",
        f"a {breed} {animal}, cute, realistic pet photography",
    ]

    count = 0
    with torch.no_grad():
        while count < n:
            for prompt in prompts:
                if count >= n:
                    break
                result = pipe(
                    prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    num_images_per_prompt=4,
                    generator=torch.manual_seed(count),
                ).images

                for img in result:
                    if count >= n:
                        break
                    img = img.resize((224, 224), Image.LANCZOS)
                    img.save(out_dir / f"{breed}_sdlora_{count:04d}.jpg")
                    count += 1

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  ✓ Saved {count} images to {out_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    with open(RESULTS_DIR / "metadata.json") as f:
        meta = json.load(f)
    minority_breeds = meta["minority_breeds"]

    train_df = pd.read_csv(SPLIT_DIR / "train.csv")
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    LORA_DIR.mkdir(parents=True, exist_ok=True)

    timing = {}
    for breed in minority_breeds:
        breed_paths = train_df[train_df["breed"] == breed]["path"].tolist()
        print(f"\n{'='*60}")
        print(f"Breed: {breed}  |  Training images: {len(breed_paths)}")
        print(f"{'='*60}")

        t0         = time.time()
        lora_path  = LORA_DIR / breed.replace(" ", "_")
        synth_dir  = SYNTH_DIR / breed.replace(" ", "_")

        # Train LoRA if not already done
        if (lora_path / "unet_lora").exists():
            print(f"  [skip] LoRA for {breed} already trained")
        else:
            train_lora(breed, breed_paths, lora_path)

        # Generate images
        if synth_dir.exists() and len(list(synth_dir.glob("*.jpg"))) >= N_GENERATE:
            print(f"  [skip] Images already generated for {breed}")
        else:
            generate_images_sd(breed, lora_path, synth_dir, N_GENERATE)

        timing[breed] = round((time.time() - t0) / 60, 1)
        print(f"  Time for {breed}: {timing[breed]} mins")

    with open(RESULTS_DIR / "sdlora_timing.json", "w") as f:
        json.dump(timing, f, indent=2)

    print("\n✓ SD+LoRA fine-tuning and generation complete.")
    print(f"  Synthetic images saved to: {SYNTH_DIR}")


if __name__ == "__main__":
    main()
