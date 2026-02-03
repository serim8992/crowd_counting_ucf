# Head Scale Adaptive Lightweight Crowd Counting

This repository provides a lightweight crowd counting framework designed for real-time deployment  
in resource-constrained embedded environments.

The model adopts a RepViT M1.1 encoder with a compact single-pass decoder, and introduces  
Relative Head-Scale Adaptive Sigma together with Head-Scale Weighted Bayesian Loss to improve  
scale robustness during training without adding inference overhead.

> ⚠️ Note: Full implementation and training datasets are maintained in a private laboratory repository  
> due to institutional policy. This public repository provides an overview of the method and configuration.

---

## 🚀 Key Features

- **Lightweight RepViT Backbone**
  RepViT M1.1 encoder optimized for real-time embedded inference.

- **Single-Pass Decoder**
  Compact multi-scale fusion with low-latency design.

- **Relative Head-Scale Adaptive Sigma**
  Per-instance adaptive Gaussian kernels derived from annotation geometry (training only).

- **Head-Scale Weighted Bayesian Loss**
  Scale-aware supervision that emphasizes large near-field heads while preserving probabilistic density learning.

- **Embedded Deployment Ready**
  Validated on Jetson Orin Nano / Orin NX with stable near real-time throughput.

---

## 🧠 Method Overview

The framework predicts crowd density maps using a lightweight encoder–decoder architecture.

Main components:

- RepViT M1.1 Backbone Encoder
- Scale Pyramid Module (single global context injection)
- Progressive Single-Pass Decoder
- Density Regression Head
- Relative Head-Scale Adaptive Bayesian Supervision (training only)

During training:

1. Head scale is estimated using nearest-neighbor distances between annotation points.
2. Each annotation receives an adaptive Gaussian sigma.
3. Bayesian loss is re-weighted using normalized head scale.

These mechanisms operate only at supervision level and introduce **zero inference overhead**.

---

## 🛠 Installation

```bash
git clone https://github.com/serim8992/crowd_counting.git
cd crowd_counting

pip install torch torchvision numpy opencv-python tqdm scikit-learn matplotlib
```

---

## 🏃 Usage (Example)

```bash
# Standard training
python train.py --dataset qnrf --model repvit_m1_1

# Enable relative head-scale adaptive supervision
python train.py --adaptive_sigma True --head_weight True

# Embedded-oriented lightweight configuration
python train.py --model repvit_m1_1 --batch_size 5
```

### Key Arguments

| Argument | Description | Default |
|---------|-------------|---------|
| --model | backbone architecture | repvit_m1_1 |
| --dataset | qnrf / shanghaiA / shanghaiB | qnrf |
| --epochs | Training epochs | 1200 |
| --batch_size | Batch size | 5 |
| --lr | learning rate | 5e-4 |
| --adaptive_sigma | relative head-scale adaptive sigma | True |
| --head_weight | head-scale weighted Bayesian loss | True |

---

## 📂 Project Structure

```
crowd_counting/
├── datasets/
├── models/
├── losses/
├── training/
├── tools/
├── configs/
├── train.py
└── test.py

```
---
## 📊 Benchmarks

UCF-QNRF: MAE 81.89 @ 24.43 GFLOPs

Jetson Orin Nano / Orin NX: approximately 10–14 FPS

---

## 📄 Paper

A Head Scale Adaptive Lightweight Crowd Counter for Real-Time Embedded Deployment
(Currently under review)

---

## 👤 Maintainer

Selim Jeong  Seung-Min Jeong Yeongje Park
University Research Lab

---

⭐ If you find this project useful, a star is always appreciated.
```
