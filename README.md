# WHU-PolSAR-CD: A Benchmark Dataset for PolSAR Change Detection

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **A Change Proportion-Driven Small-Sample Learning Framework and an Open-Source Dataset for PolSAR Change Detection**

## 📋 Overview

**WHU-PolSAR-CD** is the multi-scenario benchmark dataset for **fully polarimetric SAR (PolSAR) change detection**. It contains **2,468 co-registered L-band UAVSAR image pairs** spanning seven representative land-cover types, with carefully curated pixel-level change masks and rigorous geometric–radiometric consistency checks.

This dataset is designed to facilitate reproducible research in deep learning-based PolSAR change detection, enabling fair comparison of algorithms under diverse imaging conditions and temporal baselines.

## ✨ Key Features

- 🛰️ **Full-polarimetric L-band data**: Quad-pol (HH, HV, VH, VV) UAVSAR acquisitions at ~5m×7m resolution
- 🌍 **Seven representative scenes**: Farmland, wetlands, glaciers, oil spills, urban areas, forests, and bare soil
- 📐 **Standardized patches**: All images cropped to 128×128 pixels for DL-friendly training
- 🎯 **High-quality annotations**: Pixel-level change masks with cross-validation and auxiliary data verification
- 🔧 **Preprocessed & ready-to-use**: Radiometrically calibrated, geometrically corrected, speckle-filtered, and co-registered
- 📊 **Benchmark results**: Baseline performance for multiple input representations and network architectures

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
| **Change types** | Vegetation phenology, water expansion, glacier retreat, oil diffusion, urban expansion, etc. |

### Scene Distribution

| Region | Location | Land Cover | Temporal Baseline |
|--------|----------|-----------|------------------|
| CV | Central Valley, CA | Farmland | ~4 years |
| Cochis | Cochise, AZ | Agricultural/Bare soil | ~2 years |
| Iceland | Iceland | Glaciers | ~1.5 years |
| SJV | San Joaquin Valley, CA | Farmland | ~4 years |
| PAD | Peace Athabasca Delta, Canada | Wetlands | ~5 years |
| YFW | Yukon Flats West, AK | River basins/Forests | ~2 years |
| SB | Santa Barbara, CA | Marine oil spills | ~2 days |

## 📁 Directory Structure

```
WHU-PolSAR-CD/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── data/
│   ├── pre/          # Time-1 images (.npy)
│   ├── next/         # Time-2 images (.npy)
│   └── gt/           # Ground truth masks (.png)
|
├── tools/
│   ├── data_convert/
│   │   ├── polarization_mode_convert.py   # c3toc2: T3→C2 conversion
│   │   └── plosar_format_convert.py       # Format utilities
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
- Ambiguous regions (~3.2% of pixels) excluded from evaluation

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/WHU-PolSAR-CD.git
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
from datasets.polsar_cd import get_dataset

# Load dataset with different input representations
data_root = Path('data')

# Option 1: Real-valued input (RIC: Real+Imaginary)
real_dataset = get_dataset(
    data_path=data_root,
    mode='train',           # 'train' | 'val' | 'test'
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

# Get a sample
sample = real_dataset[0]
data = sample['data']   # Shape: (18, 128, 128) = [pre(9) + next(9)]
gt = sample['gt']       # Shape: (1, 128, 128), values in {0, 1}
name = sample['name']   # Pair identifier
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
- Rotation by 90° or 180° (p=0.5)

Augmentations are applied **consistently** to both temporal inputs and the ground truth mask.

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

### Input Representation Comparison (U-Net-80, Full Supervision)

| Representation | Precision (%) | Recall (%) | F1 (%) | Kappa (%) |
|---------------|---------------|------------|--------|-----------|
| RIC (`real`) | 89.80 ± 1.77 | 86.10 ± 1.73 | 87.90 ± 1.49 | 85.34 ± 1.52 |
| APC (`ap`) | 89.59 ± 2.29 | 87.54 ± 1.58 | 88.52 ± 0.80 | 86.04 ± 0.80 |
| **LPC (`ap_db`)** | **90.37 ± 1.50** | **90.43 ± 1.93** | **90.38 ± 0.72** | **88.27 ± 0.70** |
| CRC (`complex`)* | 92.16 ± 2.70 | 83.85 ± 0.90 | 87.79 ± 1.45 | 85.28 ± 1.53 |

*\*CRC results use Complex U-Net with comparable parameter count.*

### Few-Shot Learning (1% Labeled Pixels)

| Method | Precision (%) | Recall (%) | F1 (%) | Kappa (%) |
|--------|---------------|------------|--------|-----------|
| Traditional (Bartlett+KI) | 71.94 | 72.61 | 72.28 | 65.84 |
| Supervised (1% labels) | 71.16 | 69.21 | 70.17 | 63.39 |
| Pseudo-label Training | 74.04 | 73.12 | 73.58 | 67.52 |
| **CPSSL (Ours)** | **75.27** | **80.14** | **77.62** | **72.26** |

## 🔬 Preprocessing Pipeline

All data undergo the following standardized preprocessing before distribution:

1. **Radiometric Calibration**: Absolute calibration by UAVSAR team + incidence-angle normalization
2. **Geometric Correction**: Range-Doppler terrain correction → ground-range geometry (WGS84/UTM)
3. **Speckle Filtering**: Refined Lee filter (5×5 window) preserving edges and point targets
4. **Co-Registration**: Polynomial-based sub-pixel alignment (<1 pixel RMSE)
5. **Patch Extraction**: 128×128 sliding window with 50% overlap, filtered for valid change content
6. **Vectorization**: 3×3 Hermitian T₃ → 9-channel real vector `[T11,T22,T33,T12_R,T13_R,T23_R,T12_I,T13_I,T23_I]`

See [`docs/data_format.md`](docs/data_format.md) for implementation details of `sequence_to_vector`, `sequence_to_complex`, and `c3toc2`.

## 📚 Citation

If you use this dataset or the CPSSL framework in your research, please cite:

```bibtex
@article{bo2026polsarcd,
  title={A Change Proportion-Driven Small-Sample Learning Framework and an Open-Source Dataset for PolSAR Change Detection}, 
```

Also consider citing the UAVSAR data source:
```bibtex
@INPROCEEDINGS{uavsar,
  author={Rosen, P.A. and Hensley, S. and Wheeler, K. and Sadowy, G. and Miller, T. and Shaffer, S. and Muellerschoen, R. and Jones, C. and Zebker, H. and Madsen, S.},
  booktitle={2006 IEEE Conference on Radar}, 
  title={UAVSAR: a new NASA airborne SAR system for science and technology research}, 
  year={2006},
  volume={},
  number={},
  pages={8 pp.-},
  keywords={NASA;Space technology;Radar tracking;Synthetic aperture radar;Seismic measurements;Radar polarimetry;Unmanned aerial vehicles;Directive antennas;Propulsion;Laboratories},
  doi={10.1109/RADAR.2006.1631770}
}
```

## 🪪 License

- **Dataset**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Code**: [MIT License](LICENSE)

You are free to:
- ✅ Share — copy and redistribute the material in any medium or format
- ✅ Adapt — remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:
- 🔹 **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made.

## 📬 Contact

- **Dataset & Code Issues**: [GitHub Issues](https://github.com/er134/WHU-PolSAR-CD/issues)
- **Scientific Inquiries**: `shenbo94@whu.edu.cn`

---

> **Funding**: This work was supported by the National Natural Science Foundation of China under Grant 62471337 and Grant 42201416.

*Last updated: January 2026*