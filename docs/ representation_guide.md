# WHU-PolSAR-CD: Input Representation Guide

This document provides a comprehensive guide to the polarimetric feature encoding schemes supported by the **WHU-PolSAR-CD** dataset. It covers mathematical formulations, channel mappings, implementation details, and practical recommendations for deep learning-based change detection.

---

## 🎯 Why Input Representation Matters in PolSAR CD

Polarimetric SAR data is inherently represented as a **3×3 Hermitian complex coherency matrix** $\mathbf{T}_3$. Standard convolutional neural networks (CNNs) expect real-valued, fixed-channel tensors. The choice of how to flatten, transform, and normalize $\mathbf{T}_3$ directly impacts:

- 📉 **Gradient stability** during backpropagation
- 🌊 **Dynamic range compression** for weak vs. strong scatterers
- 🔄 **Phase relationship preservation** across polarimetric channels
- 🛡️ **Robustness to multiplicative speckle noise**
- ⚖️ **Precision-Recall trade-off** in final change masks

Our experiments show that input representation can shift F1 scores by **>3%** and significantly alter model convergence behavior. This guide helps you choose the right scheme for your use case.

---

## 📦 Supported Representation Schemes

The dataset provides utilities to convert the raw 9-channel coherency vector into four primary encoding strategies, plus a compact C2-reduced variant.

| `data_type` | Name | Channels/Frame | Output Type | Recommended For |
|:-----------:|:-----|:--------------:|:-----------:|:----------------|
| `'real'` | Real-Imaginary (RIC) | 9 | `float32` | Baselines, lossless conversion |
| `'ap'` | Amplitude-Phase (APC) | 9 | `float32` | Physical interpretability, phase analysis |
| `'ap_db'` ⭐ | Log-Phase (LPC) | 9 | `float32` | **Default choice** (Best accuracy & stability) |
| `'complex'` | Complex Tensor (CRC) | 6 | `complex64` | Complex-valued networks (CV-CNN) |
| `'ap_c2_db'` | C2-Reduced LPC | 5 | `float32` | Lightweight models, dimensionality ablation |

> 💡 **Bi-temporal Input**: All schemes concatenate time-1 and time-2 tensors along the channel dimension. Network input shape = `(C×2, 128, 128)`.

---

## 🔬 Detailed Scheme Breakdown

### 1. Real-Imaginary Combination (RIC)
**Description**: Direct decomposition of independent $\mathbf{T}_3$ elements into real and imaginary parts.

**Mathematical Form**:
$$
\mathcal{F}_{\text{RIC}} = \big[T_{11}, T_{22}, T_{33}, \Re(T_{12}), \Re(T_{13}), \Re(T_{23}), \Im(T_{12}), \Im(T_{13}), \Im(T_{23})\big]
$$

**Channel Order**:
```
[0] T11_R   [3] T12_R   [6] T12_I
[1] T22_R   [4] T13_R   [7] T13_I
[2] T33_R   [5] T23_R   [8] T23_I
```

✅ **Pros**: Mathematically lossless, trivial to implement, compatible with any real-valued CNN.  
⚠️ **Cons**: Linear scale causes strong scatterers to dominate gradients; weaker change signals may be suppressed.  
🎯 **Use When**: Establishing baselines, debugging network architecture, or feeding into complex-valued preprocessing.

---

### 2. Amplitude-Phase Combination (APC)
**Description**: Converts off-diagonal complex elements to polar coordinates, explicitly decoupling scattering intensity and phase structure.

**Mathematical Form**:
$$
\mathcal{F}_{\text{APC}} = \big[T_{11}, T_{22}, T_{33}, |T_{12}|, \angle T_{12}, |T_{13}|, \angle T_{13}, |T_{23}|, \angle T_{23}\big]
$$
where $\angle T_{ij} = \operatorname{atan2}(\Im(T_{ij}), \Re(T_{ij})) \in [-\pi, \pi]$.

✅ **Pros**: Physically interpretable; phase channels capture dielectric/structural variations; amplitude channels reflect backscatter energy.  
⚠️ **Cons**: Amplitude dynamic range spans 30–50 dB; standard min-max or Z-score normalization may still saturate weak targets.  
🎯 **Use When**: Analyzing phase sensitivity, studying scattering mechanisms, or training phase-aware attention modules.

---

### 3. Logarithmic Intensity-Phase Combination (LPC) ⭐ *Recommended*
**Description**: Applies decibel scaling to all intensity/amplitude channels while preserving raw phase values.

**Mathematical Form**:
$$
\mathcal{F}_{\text{LPC}} = \big[10\log_{10}(T_{ii}+\epsilon),\; 10\log_{10}(|T_{ij}|+\epsilon),\; \angle T_{ij}\big]
$$
where $\epsilon = 10^{-6}$ prevents numerical singularities.

**Channel Order** (same as APC, but dB-scaled):
```
[0] T11_dB   [3] |T12|_dB   [6] ∠T13
[1] T22_dB   [4] ∠T12       [7] |T23|_dB
[2] T33_dB   [5] |T13|_dB   [8] ∠T23
```

✅ **Pros**: Compresses dynamic range into a Gaussian-like distribution; balances gradient flow across channels; empirically achieves **highest F1 (90.71%)** and lowest variance.  
⚠️ **Cons**: Requires careful $\epsilon$ selection; dB values can be negative (handled naturally by normalization).  
🎯 **Use When**: **Default choice for production models**, few-shot learning, or any task requiring robust change detection across heterogeneous scenes.

---

### 4. Complex/Real-valued Combination (CRC)
**Description**: Preserves the native complex structure for Complex-Valued Deep Learning (CV-DL) architectures.

**Mathematical Form**:
$$
\mathcal{F}_{\text{CRC}} = \big[T_{11}, T_{22}, T_{33}, T_{12}, T_{13}, T_{23}\big] \in \mathbb{C}^6
$$

✅ **Pros**: Maintains intrinsic wave interference and phase rotation algebra; avoids artificial real/imag decomposition artifacts.  
⚠️ **Cons**: Requires specialized CV-Conv, CV-BN, and zReLU-like activations; higher memory footprint; tends to yield high precision but lower recall due to amplitude sensitivity.  
🎯 **Use When**: Researching complex-valued networks, theoretical studies on polarimetric phase algebra, or benchmarking CV-CNNs.

---

### 5. C2-Reduced Representation
**Description**: Projects the 9-channel $\mathbf{T}_3$ space into a 5-channel $\mathbf{C}_2$ subspace by retaining dominant scattering components.

**Channel Order**:
```
[T11, T22, |T12| (or dB), ∠T12, T33]
```

✅ **Pros**: Reduces parameters by ~44%; faster inference; removes redundant polarimetric correlations.  
⚠️ **Cons**: Loses subtle cross-polarization interactions; may degrade performance on fine-grained changes.  
🎯 **Use When**: Edge deployment, ablation studies on polarimetric dimensionality, or training lightweight U-Nets.

---

## 💻 Implementation Guide

### Loading Data via `get_dataset`
The dataset provides a unified factory function to instantiate PyTorch `Dataset` objects:

```python
from pathlib import Path
from datasets.polsar_cd import get_dataset

data_root = Path('data/')

# Recommended: LPC with dB scaling
dataset = get_dataset(
    data_path=data_root,
    mode='train',           # 'train' | 'val' | 'test'
    data_type='ap_db',      # Choose: 'real', 'ap', 'ap_db', 'complex', 'ap_c2_db'
    mean=np.load('stats/mean.npy'),
    std=np.load('stats/std.npy')
)

# DataLoader handles bi-temporal concatenation automatically
sample = dataset[0]
x = sample['data']  # Shape: (18, 128, 128) for ap_db; (12, 128, 128) complex for 'complex'
y = sample['gt']    # Shape: (1, 128, 128)
```

### Custom Transform Pipeline
If you need to apply representations manually, the core utilities are located in `tools/data_convert/`:

```python
import numpy as np
from tools.data_convert.plosar_format_convert import sequence_to_vector, sequence_to_complex
from tools.data_convert.polarization_mode_convert import c3toc2

# 1. Raw load (9, 128, 128)
raw = np.load('data/train/pre/pair_0001.npy')

# 2. Convert to LPC
lpc_features = sequence_to_vector(raw, is_db=True)

# 3. Convert to C2-reduced LPC
c2_lpc = c3toc2(lpc_features)  # (5, 128, 128)

# 4. Convert to Complex (for CV-CNN)
complex_features = sequence_to_complex(raw)  # (6, 128, 128) complex64
```

### Normalization Best Practices
Always compute statistics **on the training split only**, and **after** representation conversion:

```bash
# Compute mean/std for LPC
python scripts/compute_stats.py --data-root data/ --split train --mode ap_db
```

Apply via `torchvision.transforms.Normalize`:
```python
from torchvision.transforms import Normalize
normalize = Normalize(mean=mean.tolist(), std=std.tolist())
# Applied inside RealDataset / APDataset automatically
```

---

## 📊 Practical Recommendations & Decision Flow

| Your Goal | Recommended Scheme | Network Suggestion | Key Hyperparameter |
|:----------|:------------------|:-------------------|:-------------------|
| 🏆 Best overall accuracy & robustness | `ap_db` (LPC) | U-Net / DeepLabv3+ | Standard Adam, lr=1e-4 |
| 🔬 Complex-valued network research | `complex` (CRC) | CUNet / Complex Transformer | Complex BN, zReLU activation |
| 🧪 Baseline / Reproducibility check | `real` (RIC) | Any standard CNN | None |
| 🌊 Phase-sensitive analysis | `ap` | Attention-based models | Phase-aware loss weighting |

> 💡 **Rule of Thumb**: Start with `ap_db`. If training diverges or recall is low, verify normalization stats and consider increasing batch size. Only switch to `complex` if explicitly studying CV-CNN theory.

---

📬 **Need help?** Open an issue on GitHub or contact `shenbo94@whu.edu.cn` with your `data_type`, network architecture, and error logs.

*Document version: 1.1 | Last updated: January 2026*