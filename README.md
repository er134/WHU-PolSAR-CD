# WHU-PolSAR-CD: A Benchmark Dataset for PolSAR Change Detection

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **A Change Proportion-Driven Small-Sample Learning Framework and an Open-Source Dataset for PolSAR Change Detection**
>
> Bo Shen, Jie Yang, Weidong Sun, Lei Shi, and Lingli Zhao
>

## 📋 Overview

**WHU-PolSAR-CD** is a multi-scenario benchmark dataset for **fully polarimetric SAR (PolSAR) change detection**. It contains **2,468 co-registered L-band UAVSAR image pairs** spanning various land-cover change scenes, with carefully curated pixel-level change masks and rigorous geometric–radiometric consistency checks.

This dataset is designed to facilitate reproducible research in deep learning-based PolSAR change detection, enabling fair comparison of algorithms under diverse imaging conditions and temporal baselines. Using this dataset, the impact of several representative input types (e.g., real/imaginary parts, linear/logarithmic amplitude and phase, complex/real-valued elements) on CD performance has been evaluated. Under the stricter spatially disjoint split that eliminates spatial leakage, the logarithmic intensity-phase combination (LPC) achieves the best overall performance, yielding an F1 score of 79.33% and a Kappa coefficient of 73.54%. Furthermore, we propose a novel Change Proportion-driven Small-Sample Learning (CPSSL) framework for efficiently adapting models to change regions of interest. This framework leverages the global change proportion estimated by the Bartlett distance as weak supervision. CPSSL achieves an F1-score of 77.62% while being trained on only 1% annotated pixels, outperforming existing weak- and pseudo-label methods, and showing strong resistance to speckle noise.

## ✨ Key Features

- 🛰️ **Full-polarimetric L-band data**: Quad-pol (HH, HV, VH, VV) UAVSAR acquisitions at ~5 m (range) × 7 m (azimuth) resolution
- 🌍 **Various land-cover change scenes**: Farmland, wetlands, glaciers, oil spills, urban areas, forests, and bare soil
- 📐 **Standardized patches**: All images cropped to 128×128 pixels for DL-friendly training
- 🎯 **High-quality annotations**: Pixel-level change masks with cross-checking and auxiliary data verification
- 🔧 **Preprocessed & ready-to-use**: Radiometrically calibrated, polarimetrically calibrated, geocoded, speckle-filtered, and co-registered (<1 pixel RMSE)
- 📊 **Benchmark results**: Baseline performance for multiple input representations (RIC, APC, LPC, CRC) and network architectures (U-Net, DeepLabv3+, ChangeViT)
- 🧠 **CPSSL framework**: Change proportion-driven small-sample learning achieving F1=77.62% with only 1% labeled pixels

## 📥 Download

| Platform | Link | Size | Notes |
|----------|------|------|-------|
| **Kaggle** | [🔗 Download on Kaggle](https://www.kaggle.com/datasets/er1345/whu-polsar-cd) | 
| **Baidu Netdisk** | [🔗 Baidu Netdisk](https://pan.baidu.com/s/1gTpzR4QwtZ-bbVSprJCtLQ) Code: `75xt` | 


## 📊 Dataset Statistics

| Property | Value |
|----------|-------|
| **Total image pairs** | 2,468 |
| **Patch size** | 128 × 128 pixels |
| **Polarization** | Full-pol (HH, HV, VH, VV) |
| **Band** | L-band (1.26 GHz) |
| **Spatial resolution** | ~5 m (range) × 7 m (azimuth) |
| **Data format** | Flattened coherency vector (9-channel, `float32`) |
| **Annotation format** | Binary change mask (PNG, 0/255) |
| **Change types** | Agricultural rotation, seasonal flooding, freeze-thaw changes, oil spill diffusion, vegetation disappearance, etc. |

### Scene Distribution

| Region | Location | Land Cover | Temporal Baseline | Dominant Change Type |
|--------|----------|-----------|------------------|---------------------|
| CV | Central Valley, CA | Farmland | 4.0 years | Agricultural Rotation |
| Cochis | Cochise, AZ | Agricultural/Bare soil | 2.2 years | Agricultural Rotation |
| Iceland | Iceland | Glaciers | 1.6 years | Freeze-thaw Changes |
| SJV | San Joaquin Valley, CA | Farmland | 4.1 years | Agricultural Rotation |
| PAD | Peace Athabasca Delta, Canada | Wetlands | 4.9 years | Seasonal Flooding |
| YFW | Yukon Flats West, AK | River basins/Forests | 2.0 years | vegetation disappear |
| SB | Santa Barbara, CA | Marine oil spills | 2.0 days | Oil Spill Diffusion |

## 📁 Directory Structure

```
WHU-PolSAR-CD/
├── README.md
├── requirements.txt
├── data/
│   ├── pre/          # Time-1 images (.npy)
│   ├── next/         # Time-2 images (.npy)
│   └── gt/           # Ground truth masks (.png)
|
├── tools/
│   ├── data_convert/
│   │   ├── polarization_mode_convert.py   # c3toc2: T3→C2 conversion
│   │   ├── plosar_format_convert.py       # Format utilities
│   │   └── split_dataset.py              # Dataset splitting scripts
│   └── file_operation/
│       └── path_process.py                # Path handling utilities
├── datasets/
│   └── polsar_cd.py      # PyTorch Dataset implementations
|
└── docs/
    ├── data_format.md    # Detailed data format specification
    └── representation_guide.md # Input encoding schemes
```

## 🔧 Data Format

### Coherency Vector (`.npy` files in `pre/` and `next/`)

Each `.npy` file contains a **9-channel float32 array** representing the independent elements of the 3×3 Hermitian coherency matrix **T₃**, stored in the following order:

```python
import numpy as np

# Load a sample
data = np.load('data/train/pre/pair_0001.npy')  # Shape: (9, 128, 128), dtype: float32

# Channel mapping:
# [0] T11          : Real, diagonal element
# [1] T22          : Real, diagonal element  
# [2] T33          : Real, diagonal element
# [3] T12_R        : Real part of T12 (off-diagonal)
# [4] T13_R        : Real part of T13 (off-diagonal)
# [5] T23_R        : Real part of T23 (off-diagonal)
# [6] T12_I        : Imaginary part of T12
# [7] T13_I        : Imaginary part of T13
# [8] T23_I        : Imaginary part of T23
```

Due to Hermitian symmetry (T₃ = T₃ᴴ), only 9 independent elements exist: 3 real diagonals + 3 complex off-diagonals = 9 real values.

### Change Mask (`gt/*.png`)

- Single-channel PNG image, shape: (128, 128)
- Pixel values: `0` (unchanged) or `255` (changed)
- Ambiguous regions (water, soil, or oil-spill boundaries) excluded from evaluation

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/er134/WHU-PolSAR-CD.git
cd WHU-PolSAR-CD

# Install dependencies
pip install -r requirements.txt
```

### Dataset Statistics Computation (Recommended)

Before training, compute the channel-wise mean and std for normalization:

```bash
python scripts/compute_stats.py --data-root data/ --split train
# Output: mean.npy, std.npy (9-channel vectors)
```

### Loading Data (PyTorch)

```python
from pathlib import Path
import numpy as np
from datasets.polsar_cd import get_dataset

# data_path should contain {train,valid,test}/{pre,next,gt}/
# after running split_dataset.py, or {pre,next,gt}/ for unsplit data
data_root = Path('data/crops_no_border_split')

# Option 1: Real-valued input (RIC: Real+Imaginary)
real_dataset = get_dataset(
    data_path=data_root,
    mode='train',           # 'train' | 'valid' | 'test'
    data_type='real',       # 'real' | 'complex' | 'ap' | 'ap_db' | 'ap_c2_db'
    mean=np.load('mean.npy'),
    std=np.load('std.npy')
)

# Option 2: Complex-valued input (for complex-valued networks)
complex_dataset = get_dataset(
    data_path=data_root,
    mode='train',
    data_type='complex',
    mean=np.load('mean.npy'),
    std=np.load('std.npy')
)

# Option 3: Amplitude-Phase with dB scaling (LPC, recommended)
ap_db_dataset = get_dataset(
    data_path=data_root,
    mode='train',
    data_type='ap_db',      # Applies: 10*log10(|T|+ε) for amplitude channels
    mean=np.load('mean.npy'),
    std=np.load('std.npy')
)

# --- Single-stream output (concatenated bi-temporal) ---
sample = real_dataset[0]
data = sample['data']   # Shape: (18, 128, 128) = [pre(9) + next(9)]
gt = sample['gt']       # Shape: (1, 128, 128), values in {0, 1}
name = sample['name']   # Pair identifier

# --- Dual-stream output (separate pre/next for Siamese networks) ---
dual_dataset = get_dataset(
    data_path=data_root,
    mode='train',
    data_type='ap_db',
    mean=np.load('mean.npy'),
    std=np.load('std.npy'),
    dual_stream=True       # Returns separate 'pre' and 'next' tensors
)
sample = dual_dataset[0]
pre = sample['pre']     # Shape: (9, 128, 128)
next = sample['next']   # Shape: (9, 128, 128)
gt = sample['gt']       # Shape: (1, 128, 128)
```

### Input Representation Schemes

We provide four input encoding strategies via the `data_type` parameter:

| `data_type` | Description | Output Channels | Transformation | Best For |
|-------------|-------------|----------------|----------------|----------|
| `'real'` | Real-Imaginary Combination (RIC) | 18 (9+9) | Direct normalization | General-purpose, lossless |
| `'complex'` | Complex-valued tensor (CRC) | 18 complex | `sequence_to_complex()` + norm | Complex-valued networks |
| `'ap'` | Amplitude-Phase (APC) | 18 (9+9) | `sequence_to_vector(is_db=False)` | Physical interpretability |
| `'ap_db'` ⭐ | Log-Amplitude-Phase (LPC) | 18 (9+9) | `sequence_to_vector(is_db=True)` | **Best overall** (F1=90.71%) |
| `'ap_c2_db'` | LPC + C2 conversion | 10 (5+5) | APC → `c3toc2()` reduction | Compact representation |

#### Amplitude-Phase Conversion Details

For `'ap'` / `'ap_db'` modes, the 9-channel input is transformed as:
```
[T11, T22, T33, |T12|, ∠T12, |T13|, ∠T13, |T23|, ∠T23]
```

When `is_db=True` (LPC mode):
```
[T11_dB, T22_dB, T33_dB, |T12|_dB, ∠T12, |T13|_dB, ∠T13, |T23|_dB, ∠T23]
where X_dB = 10·log₁₀(|X| + ε), ε=1e-6
```

### Data Augmentation

Training mode (`mode='train'`) automatically applies random augmentations:

- Horizontal flip (p=0.5)
- Vertical flip (p=0.5)
- Rotation within ±30°
- Gaussian perturbations to simulate speckle variations

Augmentations are applied **consistently** to both temporal inputs and the ground truth mask.

### Data Splitting

We provide two splitting strategies via `tools/data_convert/split_dataset.py`:

#### 1. Spatially Disjoint Split

Each scene is divided into five equal parts by patch index order (corresponding to spatial position). The **2nd fifth → test**, **4th fifth → valid**, and the remaining (1st, 3rd, 5th) → **train**. This eliminates spatial leakage between subsets.

Output structure:
```
crops_no_border_split/
├── train/{pre,next,gt}/
├── valid/{pre,next,gt}/
└── test/{pre,next,gt}/
```

```bash
python tools/data_convert/split_dataset.py spatial \
    --src /path/to/crops_no_border \
    --dst /path/to/crops_no_border_split
```

#### 2. Few-Shot Split (1% Labeled Pixels)

23 training images (~1% of 2,468) are randomly selected for supervised fine-tuning; all remaining images are used as the test set. A `train_list.txt` records the selected sample names.

Output structure:
```
0.01_seq_notest_2/
├── train/{pre,next,gt}/   # 23 samples
├── test/{pre,next,gt}/    # 2,445 samples
└── train_list.txt
```

```bash
python tools/data_convert/split_dataset.py fewshot \
    --src /path/to/crops_no_border \
    --dst /path/to/0.01_seq_notest_2 \
    --n-train 23 --seed 42
```

### Training Simple Example

```python
from torch.utils.data import DataLoader
import torch.nn as nn

# Create data loader
train_loader = DataLoader(
    ap_db_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)

# Simple U-Net baseline (9-channel input → 18-channel bi-temporal)
model = UNet(in_channels=18, out_channels=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(100):
    for batch in train_loader:
        data, gt = batch['data'], batch['gt']  # data: [B,18,H,W], gt: [B,1,H,W]
        
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, gt)
        loss.backward()
        optimizer.step()
```

## 📈 Benchmark Results



> **Note**: LPC (Logarithmic Intensity-Phase Combination) achieves the best overall performance across CNN-based, complex-valued, and Transformer-based architectures. Under spatially disjoint split, ChangeViT+LPC obtains the best overall performance, achieving an F1 score of 79.33% and Kappa coefficient of 73.54%.

### Few-Shot Learning (1% Labeled Pixels — 23 training images)

| Method | Precision (%) | Recall (%) | F1 (%) | Kappa (%) |
|--------|---------------|------------|--------|-----------|
| Traditional (Bartlett+KI) | 71.94 | 72.61 | 72.28 | 65.84 |
| Supervised (1% labels) | 71.16 | 69.21 | 70.17 | 63.39 |
| Pseudo-label Training | 74.04 | 73.12 | 73.58 | 67.52 |
| Pseudo-label Fine-tuning | 75.07 | 65.21 | 69.79 | 63.41 |
| **CPSSL (Ours)** | **75.27** | **80.14** | **77.62** | **72.26** |

> **Note**: CPSSL outperforms existing weak- and pseudo-label methods, with particularly strong improvement in recall (+10.93% over supervised baseline).

## 🔬 Preprocessing Pipeline

All data undergo the following standardized preprocessing before distribution:

1. **Radiometric Calibration**: Absolute calibration performed by the UAVSAR processing team
2. **Polarimetric Calibration**: Standard polarimetric calibration provided in official UAVSAR products
3. **Geocoding**: Ground-range geometry in WGS-84 equiangular (EQA) coordinate system
4. **Speckle Filtering**: Refined Lee filter (5×5 window) preserving edges and point targets
5. **Co-Registration**: Polynomial-based sub-pixel alignment (RMSE < 1 pixel)
6. **Patch Extraction**: 128×128 patches with ROI selection ensuring representative changed regions
7. **Vectorization**: 3×3 Hermitian T₃ → 9-channel real vector `[T11,T22,T33,T12_R,T13_R,T23_R,T12_I,T13_I,T23_I]`

See [`docs/data_format.md`](docs/data_format.md) for implementation details of `sequence_to_vector`, `sequence_to_complex`, and `c3toc2`.

## 📚 Citation

If you use this dataset or the CPSSL framework in your research, please cite:

```bibtex
@article{bo2026polsarcd,
  title={A Change Proportion-Driven Small-Sample Learning Framework and an Open-Source Dataset for PolSAR Change Detection},
}
```

Also consider citing the UAVSAR data source:
```bibtex
@INPROCEEDINGS{uavsar,
  author={Rosen, P.A. and Hensley, S. and Wheeler, K. and Sadowy, G. and Miller, T. and Shaffer, S. and Muellerschoen, R. and Jones, C. and Zebker, H. and Madsen, S.},
  booktitle={2006 IEEE Conference on Radar}, 
  title={UAVSAR: a new NASA airborne SAR system for science and technology research}, 
  year={2006},
  doi={10.1109/RADAR.2006.1631770}
}
```

## 🪪 License

- **Dataset**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Code**: [MIT License](LICENSE)


Under the following terms:
- 🔹 **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made.

## 📬 Contact

- **Dataset & Code Issues**: [GitHub Issues](https://github.com/er134/WHU-PolSAR-CD/issues)
- **Scientific Inquiries**: `shenbo94@whu.edu.cn`

---

> **Funding**: This work was supported by the National Natural Science Foundation of China under Grant 62471337 and Grant 42201416.

*Last updated: June 2026*