"""
STEP 3: FastGAN Training & Synthetic Image Generation (FIXED v2)
Proper FastGAN architecture with:
  - Skip-Layer channel-wise Excitation (SLE) in Generator
  - Spectral normalisation on Discriminator
  - Self-supervised reconstruction loss in Discriminator
  - Lower learning rate (1e-4) for stability
  - Gradient clipping
  - Full checkpoint resume every 5k iterations

Requirements:
    pip install torch torchvision tqdm pillow pandas
Hardware: NVIDIA GPU 6-8 GB VRAM, ~2-3 hrs per class
"""

import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
SPLIT_DIR   = Path("data/oxford_pet/splits")
SYNTH_DIR   = Path("data/synthetic/fastgan")
MODELS_DIR  = Path("models/fastgan")
IMG_SIZE    = 256
N_GENERATE  = 500
BATCH_SIZE  = 8
N_ITER      = 50000
LR          = 1e-4        # KEY FIX: reduced from 2e-4 for stability
NZ          = 256
NGF         = 64
NDF         = 64
RANDOM_SEED = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Weights init ──────────────────────────────────────────────────────────────
def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname and hasattr(m, 'weight') and m.weight is not None:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# ── GLU activation ────────────────────────────────────────────────────────────
class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        return x[:, :nc//2] * torch.sigmoid(x[:, nc//2:])

# ── Skip-Layer channel-wise Excitation (SLE) ─────────────────────────────────
# Core FastGAN innovation: low-res features modulate high-res features
# This is what makes FastGAN work well on small datasets
class SLE(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(ch_in, ch_out, 4, 1, 0, bias=False),
            nn.SiLU(),
            nn.Conv2d(ch_out, ch_out, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)

# ── Generator blocks ──────────────────────────────────────────────────────────
def up_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_ch, out_ch * 2, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_ch * 2),
        GLU()
    )

# ── Generator with SLE connections ───────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, nz=NZ, ngf=NGF, nc=3):
        super().__init__()
        self.init = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True)
        )
        # Upsampling: 4->8->16->32->64->128->256
        self.up1 = up_block(ngf * 16, ngf * 8)
        self.up2 = up_block(ngf * 8,  ngf * 4)
        self.up3 = up_block(ngf * 4,  ngf * 2)
        self.up4 = up_block(ngf * 2,  ngf)
        self.up5 = up_block(ngf,      ngf // 2)
        self.up6 = up_block(ngf // 2, ngf // 4)

        # SLE: low-res feature maps excite high-res ones
        self.sle1 = SLE(ngf * 16, ngf * 2)   # 4px  -> 32px
        self.sle2 = SLE(ngf * 8,  ngf)        # 8px  -> 64px
        self.sle3 = SLE(ngf * 4,  ngf // 2)   # 16px -> 128px

        self.to_rgb = nn.Sequential(
            nn.Conv2d(ngf // 4, nc, 3, 1, 1),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, z):
        x0 = self.init(z.view(z.size(0), -1, 1, 1))  # 4x4
        x1 = self.up1(x0)                              # 8x8
        x2 = self.up2(x1)                              # 16x16
        x3 = self.up3(x2)                              # 32x32
        x3 = self.sle1(x0, x3)                        # SLE
        x4 = self.up4(x3)                              # 64x64
        x4 = self.sle2(x1, x4)                        # SLE
        x5 = self.up5(x4)                              # 128x128
        x5 = self.sle3(x2, x5)                        # SLE
        x6 = self.up6(x5)                              # 256x256
        return self.to_rgb(x6)

# ── Discriminator with spectral norm + decoder head ──────────────────────────
def sn_conv(in_ch, out_ch, k=4, s=2, p=1):
    return nn.utils.spectral_norm(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
    )

def down_block(in_ch, out_ch):
    return nn.Sequential(
        sn_conv(in_ch, out_ch),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True)
    )

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=NDF):
        super().__init__()
        self.from_rgb = nn.Sequential(
            sn_conv(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.d1 = down_block(ndf,      ndf * 2)
        self.d2 = down_block(ndf * 2,  ndf * 4)
        self.d3 = down_block(ndf * 4,  ndf * 8)
        self.d4 = down_block(ndf * 8,  ndf * 16)

        self.to_logit = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ndf * 16, 1),
        )
        # Decoder head for self-supervised reconstruction loss
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ndf * 16, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ndf * 4, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, x, return_recon=False):
        f = self.from_rgb(x)
        f = self.d1(f)
        f = self.d2(f)
        f = self.d3(f)
        f = self.d4(f)
        logit = self.to_logit(f).squeeze(1)
        if return_recon:
            recon = self.decoder(f)
            return logit, recon
        return logit

# ── Dataset ───────────────────────────────────────────────────────────────────
class BreedDataset(Dataset):
    def __init__(self, image_paths, size=256):
        self.paths = image_paths
        self.tf = transforms.Compose([
            transforms.Resize((int(size * 1.15), int(size * 1.15))),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        return self.tf(Image.open(self.paths[idx]).convert("RGB"))

# ── Adaptive training params based on dataset size ───────────────────────────
def get_training_params(n_images: int) -> dict:
    """
    Automatically adjusts iterations and batch size based on
    how many real training images are available for the breed.

    Key insight:
      - Too many iterations on tiny datasets = memorisation/overfitting
      - Too large a batch on tiny datasets = generator sees same images repeatedly
    """
    if n_images < 30:           # severe minority (e.g. 20 images)
        return {
            "n_iter":     20000,
            "batch_size": 4,
            "label":      "severe minority",
        }
    elif n_images < 80:         # moderate minority (e.g. 50 images)
        return {
            "n_iter":     35000,
            "batch_size": 6,
            "label":      "moderate minority",
        }
    else:                       # comfortable (100+ images)
        return {
            "n_iter":     N_ITER,   # full 50k
            "batch_size": BATCH_SIZE,
            "label":      "standard",
        }

# ── Training ──────────────────────────────────────────────────────────────────
def train_fastgan(breed: str, image_paths: list, save_dir: Path,
                  n_iter: int, batch_size: int = BATCH_SIZE):
    print(f"\n  Training FastGAN for: {breed} ({len(image_paths)} images)")
    save_dir.mkdir(parents=True, exist_ok=True)

    ds     = BreedDataset(image_paths, size=IMG_SIZE)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=False, drop_last=True)  # num_workers=0 is faster on Windows

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)

    # torch.compile — disabled on Windows (requires Triton which is Linux-only)
    # Uncomment the 4 lines below only if running on Linux
    # G = torch.compile(G)
    # D = torch.compile(D)
    # print("  [speed] torch.compile enabled")

    opt_g = optim.Adam(G.parameters(), lr=LR,     betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=LR * 2, betas=(0.5, 0.999))

    # AMP scaler — enables float16 mixed precision on Tensor Cores (~30-40% speedup)
    scaler_g = torch.amp.GradScaler('cuda')
    scaler_d = torch.amp.GradScaler('cuda')

    criterion       = nn.BCEWithLogitsLoss()
    recon_criterion = nn.L1Loss()
    fixed_z         = torch.randn(16, NZ, device=DEVICE)
    loader_iter     = iter(loader)
    losses          = {"g": [], "d": []}
    start_iter      = 1

    # ── Resume ────────────────────────────────────────────────────────────────
    resume_path = save_dir / "checkpoint_latest.pth"
    if resume_path.exists():
        print(f"  [resume] Loading from {resume_path}")
        ckpt       = torch.load(resume_path, map_location=DEVICE)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        if "scaler_g" in ckpt: scaler_g.load_state_dict(ckpt["scaler_g"])
        if "scaler_d" in ckpt: scaler_d.load_state_dict(ckpt["scaler_d"])
        start_iter = ckpt["iter"] + 1
        losses     = ckpt.get("losses", {"g": [], "d": []})
        print(f"  [resume] Continuing from iter {start_iter}/{n_iter}")
    else:
        print(f"  [fresh] Starting from scratch")

    # ── Loop ──────────────────────────────────────────────────────────────────
    for i in tqdm(range(start_iter, n_iter + 1), desc=f"  GAN [{breed}]",
                  initial=start_iter - 1, total=n_iter):

        try:
            real = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            real = next(loader_iter)

        real = real.to(DEVICE)
        bs   = real.size(0)

        # ── Discriminator (AMP float16) ───────────────────────────────────────
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                fake = G(torch.randn(bs, NZ, device=DEVICE))

        with torch.amp.autocast('cuda'):
            real_logit, real_recon = D(real, return_recon=True)
            fake_logit              = D(fake)

            real_labels = torch.ones(bs,  device=DEVICE) * 0.9
            fake_labels = torch.zeros(bs, device=DEVICE)

            loss_d_adv   = criterion(real_logit, real_labels) + \
                           criterion(fake_logit, fake_labels)
            real_down    = F.interpolate(real, size=real_recon.shape[-2:],
                                         mode="bilinear", align_corners=False)
            loss_d_recon = recon_criterion(real_recon, real_down)
            loss_d       = loss_d_adv + 0.1 * loss_d_recon

        opt_d.zero_grad()
        scaler_d.scale(loss_d).backward()
        scaler_d.unscale_(opt_d)
        nn.utils.clip_grad_norm_(D.parameters(), 1.0)
        scaler_d.step(opt_d)
        scaler_d.update()

        # ── Generator (AMP float16) ───────────────────────────────────────────
        with torch.amp.autocast('cuda'):
            fake   = G(torch.randn(bs, NZ, device=DEVICE))
            loss_g = criterion(D(fake), torch.ones(bs, device=DEVICE))

        opt_g.zero_grad()
        scaler_g.scale(loss_g).backward()
        scaler_g.unscale_(opt_g)
        nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        scaler_g.step(opt_g)
        scaler_g.update()

        losses["d"].append(loss_d.item())
        losses["g"].append(loss_g.item())

        # Checkpoint every 5k
        if i % 5000 == 0:
            torch.save({
                "iter":    i,
                "G":       G.state_dict(),
                "D":       D.state_dict(),
                "opt_g":   opt_g.state_dict(),
                "opt_d":   opt_d.state_dict(),
                "scaler_g": scaler_g.state_dict(),
                "scaler_d": scaler_d.state_dict(),
                "losses":  losses,
            }, resume_path)
            torch.save(G.state_dict(), save_dir / f"G_{i}.pth")
            with torch.no_grad():
                vutils.save_image(
                    G(fixed_z) * 0.5 + 0.5,
                    save_dir / f"preview_{i}.png", nrow=4
                )
            avg_g = np.mean(losses["g"][-500:])
            avg_d = np.mean(losses["d"][-500:])
            print(f"    iter {i}/{n_iter} | G={avg_g:.3f} D={avg_d:.3f} [saved]")

    torch.save(G.state_dict(), save_dir / "G_final.pth")
    if resume_path.exists():
        resume_path.unlink()
    print(f"  ✓ Done: {breed}")
    return G

# ── Generation ────────────────────────────────────────────────────────────────
def generate_images(G, breed: str, out_dir: Path, n: int = 500):
    out_dir.mkdir(parents=True, exist_ok=True)
    G.eval()
    count = 0
    print(f"  Generating {n} images for {breed}...")
    with torch.no_grad():
        while count < n:
            imgs = (G(torch.randn(16, NZ, device=DEVICE)) * 0.5 + 0.5).cpu()
            for img_t in imgs:
                if count >= n: break
                img = transforms.ToPILImage()(img_t.clamp(0, 1))
                img.resize((224, 224), Image.LANCZOS).save(
                    out_dir / f"{breed}_fastgan_{count:04d}.jpg")
                count += 1
    print(f"  ✓ Saved {count} images to {out_dir}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    with open(RESULTS_DIR / "metadata.json") as f:
        meta = json.load(f)
    minority_breeds = meta["minority_breeds"]
    train_df = pd.read_csv(SPLIT_DIR / "train.csv")
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    timing = {}
    for breed in minority_breeds:
        breed_paths = train_df[train_df["breed"] == breed]["path"].tolist()
        print(f"\n{'='*60}")
        print(f"Breed: {breed}  |  Images: {len(breed_paths)}")
        print(f"{'='*60}")

        t0        = time.time()
        model_dir = MODELS_DIR / breed.replace(" ", "_")
        synth_dir = SYNTH_DIR  / breed.replace(" ", "_")

        # Automatically adjust iterations and batch size for this breed
        params = get_training_params(len(breed_paths))
        print(f"  Training profile : {params['label']}")
        print(f"  Iterations       : {params['n_iter']:,}")
        print(f"  Batch size       : {params['batch_size']}")

        if (model_dir / "G_final.pth").exists():
            print(f"  [skip] Already trained")
            G = Generator().to(DEVICE)
            G.load_state_dict(torch.load(model_dir / "G_final.pth",
                                         map_location=DEVICE))
        else:
            G = train_fastgan(breed, breed_paths, model_dir,
                              n_iter=params["n_iter"],
                              batch_size=params["batch_size"])

        if synth_dir.exists() and len(list(synth_dir.glob("*.jpg"))) >= N_GENERATE:
            print(f"  [skip] Images already generated")
        else:
            generate_images(G, breed, synth_dir, N_GENERATE)

        timing[breed] = round((time.time() - t0) / 60, 1)
        print(f"  Time: {timing[breed]} mins")

    with open(RESULTS_DIR / "fastgan_timing.json", "w") as f:
        json.dump(timing, f, indent=2)
    print("\n✓ All breeds complete.")
    print("  Check preview images in models/fastgan/<breed>/ to verify quality.")

if __name__ == "__main__":
    main()
