# Bias Benchmark: GAN vs. Diffusion for Bias Correction
## Oxford-IIIT Pet Dataset | Local PC | No Cloud | No Paid Tools

---

## Project Structure

```
bias_benchmark/
├── scripts/
│   ├── 01_prepare_dataset.py             — Download data, build splits, identify minority breeds
│   └── 01b_introduce_imbalance.py        — include the imbalance in the dataset
│   └── 01c_traditional_augmentation.py   — performs traditional augmentation of the real images(flip, rotate,..)
│   ├── 02_train_baseline.py              — Train ResNet-50 on real data only
│   ├── 03_train_fastgan.py               — Train FastGAN per minority breed + generate images
│   ├── 04_train_sd_lora.py               — LoRA fine-tune SD 1.5 per breed + generate images
│   ├── 05_train_augmented.py             — Train ResNet-50 on GAN / SD / Hybrid augmented data
│   ├── 06_compute_fid.py                 — Compute FID scores for image quality
│   └── 07_compare_results.py             — Final comparison table + paper-ready figures
│   └── 08_significance_tests.py          — Performs significance test across 3 seeds
│   └── 09_tsne_analysis.py               — Performs tsne analysis accross 3 augementation tecniques
├── data/                                 — Created automatically
│   ├── oxford_pet/                       — Raw dataset + train/test splits
│   └── synthetic/                        — Generated images (fastgan/ and sd_lora/)
├── models/                               — Saved model weights
└── results/                              — JSON metrics, figures, summary CSV
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 6 GB    | 8 GB        |
| RAM       | 16 GB   | 32 GB       |
| Disk      | 30 GB   | 50 GB       |
| CUDA      | 11.8+   | 12.x        |

---

## Setup

```bash
# 1. Create conda environment
conda create -n bias_bench python=3.10 -y
conda activate bias_bench

# 2. Install PyTorch (CUDA 11.8 — adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install all other dependencies
pip install diffusers transformers accelerate peft
pip install scikit-learn pandas matplotlib seaborn
pip install pytorch-fid requests tqdm pillow

# 4. (Optional but recommended for 6 GB VRAM)
pip install xformers
```

---

## Running — Step by Step

Run scripts in order. Each step is independent and resumable (skips already-completed work).

```bash
cd bias_benchmark

# Day 1-2: Download dataset, build splits, check bias
python scripts/01_prepare_dataset.py
python scripts/01b_introduce_imbalance.py
python scripts/01c_traditional_augmentation.py

# Day 3-4: Train baseline (real data only) — ~1-2 hrs
python scripts/02_train_baseline.py

# Day 5-6: Train FastGAN per breed (~2-3 hrs/breed = ~12-15 hrs total)
python scripts/03_train_fastgan.py

# Day 7: Train GAN-augmented classifier — ~1-2 hrs
# Day 8-9: LoRA fine-tune SD 1.5 per breed (~45 min/breed = ~4 hrs total)
python scripts/04_train_sd_lora.py

# Day 10-11: Train SD+LoRA and Hybrid classifiers — ~2-4 hrs
python scripts/05_train_augmented.py

# Day 11: Compute FID scores (~30 min)
python scripts/06_compute_fid.py

# Day 12: Generate all comparison figures and summary table
python scripts/07_compare_results.py
python scripts/08_significance_tests.py
python scripts/09_tsne_analysis.py
```

---

## Output Files

After running all steps, `results/` will contain:

| File | Description |
|------|-------------|
| `metadata.json` | Dataset info, minority breeds |
| `label_map.json` | Breed → integer label mapping |
| `metrics_baseline.json` | Baseline accuracy + fairness metrics |
| `metrics_fastgan.json` | FastGAN augmented results |
| `metrics_sd_lora.json` | SD+LoRA augmented results |
| `metrics_hybrid.json` | Hybrid augmented results |
| `fid_scores.json` | FID per breed per method |
| `summary_table.csv` | Paper-ready comparison table |
| `figures/fig1_macro_f1.png` | Bar chart: Macro F1 by condition |
| `figures/fig2_minority_majority.png` | Grouped bar: minority vs majority accuracy |
| `figures/fig3_bias_gap.png` | Bias gap reduction |
| `figures/fig4_heatmap.png` | Per-class accuracy heatmap |
| `figures/fig5_fid.png` | FID scores by method and breed |

---

## Key Design Decisions

- **Test set is NEVER touched during augmentation** — synthetic images only go into training
- **Fixed random seeds** throughout for full reproducibility
- **Same ResNet-50 backbone** across all conditions — only the training data changes
- **Gradient checkpointing** enabled for SD 1.5 LoRA training to fit within 6 GB VRAM
- **+500 synthetic images per minority class** — consistent across all augmentation conditions

---

## Common Issues

| Issue | Fix |
|-------|-----|
| CUDA out of memory (SD LoRA) | Reduce `BATCH_SIZE` to 1, enable `xformers` |
| CUDA out of memory (FastGAN) | Reduce `IMG_SIZE` to 128 or `BATCH_SIZE` to 4 |
| SD model download slow | Pre-download: `huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5` |
| FID script not found | `pip install pytorch-fid` |

---

## Citation

If you use this codebase, please cite the paper:
Syntethic data avialble at : https://ieee-dataport.org/documents/pet-breed-generative-augmentation-dataset (https://dx.doi.org/10.21227/tqn0-zz39) 

All code released under MIT License.
