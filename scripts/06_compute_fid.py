"""
STEP 6: Compute FID Scores for Synthetic Image Quality
Measures how realistic the GAN and SD-generated images are
compared to real held-out images.

Requirements:
    pip install pytorch-fid torch torchvision
    (or: pip install clean-fid)
"""

import os
import json
import shutil
import subprocess
import pandas as pd
from pathlib import Path

SPLIT_DIR   = Path("data/oxford_pet/splits")
SYNTH_TRAD  = Path("data/synthetic/traditional")
SYNTH_GAN   = Path("data/synthetic/fastgan")
SYNTH_SD    = Path("data/synthetic/sd_lora")
RESULTS_DIR = Path("results")
TMP_DIR     = Path("data/tmp_fid")

# ── Prepare real reference images per breed ───────────────────────────────────
def prepare_real_refs(minority_breeds: list):
    """Copy and resize real test images to 299x299 for FID computation.
    pytorch-fid requires all images to be the same size.
    """
    from PIL import Image as PILImage
    test_df = pd.read_csv(SPLIT_DIR / "test.csv")
    ref_dirs = {}
    for breed in minority_breeds:
        ref_dir = TMP_DIR / "real" / breed.replace(" ", "_")
        ref_dir.mkdir(parents=True, exist_ok=True)
        breed_paths = test_df[test_df["breed"] == breed]["path"].tolist()
        for p in breed_paths:
            dst = ref_dir / Path(p).name
            if not dst.exists():
                img = PILImage.open(p).convert("RGB").resize((299, 299))
                img.save(dst)
        ref_dirs[breed] = str(ref_dir)
        print(f"  Real ref for {breed}: {len(breed_paths)} images (resized to 299x299)")
    return ref_dirs

def prepare_synth_refs(synth_dir: Path, minority_breeds: list) -> dict:
    """Resize synthetic images to 299x299 for FID computation."""
    from PIL import Image as PILImage
    resized_dirs = {}
    for breed in minority_breeds:
        safe      = breed.replace(" ", "_")
        src_dir   = synth_dir / safe
        if not src_dir.exists():
            continue
        dst_dir = TMP_DIR / synth_dir.name / safe
        dst_dir.mkdir(parents=True, exist_ok=True)
        imgs = list(src_dir.glob("*.jpg"))
        for p in imgs:
            dst = dst_dir / p.name
            if not dst.exists():
                img = PILImage.open(p).convert("RGB").resize((299, 299))
                img.save(dst)
        resized_dirs[breed] = str(dst_dir)
    return resized_dirs

# ── Run FID via pytorch-fid CLI ───────────────────────────────────────────────
def compute_fid(real_dir: str, fake_dir: str) -> float:
    """
    Uses the pytorch-fid command-line tool.
    Install: pip install pytorch-fid
    """
    result = subprocess.run(
        ["python", "-m", "pytorch_fid", real_dir, fake_dir,
         "--device", "cuda", "--num-workers", "0",
         "--batch-size", "16"],
        capture_output=True, text=True
    )
    # Output looks like: "FID:  42.37"
    for line in result.stdout.splitlines():
        if "FID" in line:
            try:
                return float(line.split()[-1])
            except ValueError:
                pass
    print(f"  [WARN] Could not parse FID output:\n{result.stdout}\n{result.stderr}")
    return -1.0

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    with open(RESULTS_DIR / "metadata.json") as f:
        meta = json.load(f)
    minority_breeds = meta["minority_breeds"]

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    ref_dirs = prepare_real_refs(minority_breeds)

    # Pre-resize all synthetic images to 299x299
    print("\nResizing synthetic images to 299x299 for FID...")
    trad_resized = prepare_synth_refs(SYNTH_TRAD, minority_breeds)
    gan_resized  = prepare_synth_refs(SYNTH_GAN,  minority_breeds)
    sd_resized   = prepare_synth_refs(SYNTH_SD,   minority_breeds)

    fid_results = {}

    for breed in minority_breeds:
        fid_results[breed] = {}
        print(f"\nComputing FID for breed: {breed}")

        # Traditional aug FID
        if breed in trad_resized:
            fid_trad = compute_fid(ref_dirs[breed], trad_resized[breed])
            fid_results[breed]["traditional"] = round(fid_trad, 2)
            print(f"  Traditional FID: {fid_trad:.2f}")
        else:
            print(f"  [skip] Traditional images not found for {breed}")

        # GAN FID
        if breed in gan_resized:
            fid_gan = compute_fid(ref_dirs[breed], gan_resized[breed])
            fid_results[breed]["fastgan"] = round(fid_gan, 2)
            print(f"  FastGAN  FID: {fid_gan:.2f}")
        else:
            print(f"  [skip] FastGAN images not found for {breed}")

        # SD FID
        if breed in sd_resized:
            fid_sd = compute_fid(ref_dirs[breed], sd_resized[breed])
            fid_results[breed]["sd_lora"] = round(fid_sd, 2)
            print(f"  SD+LoRA  FID: {fid_sd:.2f}")
        else:
            print(f"  [skip] SD+LoRA images not found for {breed}")

    # Average FID across breeds
    avg_fid = {}
    for method in ["traditional", "fastgan", "sd_lora"]:
        scores = [fid_results[b].get(method, None) for b in minority_breeds]
        scores = [s for s in scores if s is not None and s >= 0]
        avg_fid[method] = round(sum(scores) / len(scores), 2) if scores else -1

    fid_results["average"] = avg_fid
    print(f"\nAverage FID — Traditional: {avg_fid.get('traditional')}  |  FastGAN: {avg_fid.get('fastgan')}  |  SD+LoRA: {avg_fid.get('sd_lora')}")

    out_path = RESULTS_DIR / "fid_scores.json"
    with open(out_path, "w") as f:
        json.dump(fid_results, f, indent=2)
    print(f"\n✓ FID scores saved to {out_path}")

    # Cleanup temp dir
    shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
