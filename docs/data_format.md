# WHU-PolSAR-CD Data Format Specification

This document provides detailed specifications for the data formats used in the **WHU-PolSAR-CD** dataset, including file structures, channel mappings, coordinate conventions, and code examples for loading and transforming data.

---

## 📦 File Organization

```
WHU-PolSAR-CD/
├── data/
    ├── pre/          # Time-1 images (.npy)
    ├── next/         # Time-2 images (.npy)
    └── gt/           # Ground truth masks (.png)
```

> **Note**: Filename consistency is enforced across splits: `pre/{name}.npy`, `next/{name}.npy`, and `gt/{name}.png` refer to the same image pair.

---

## 🔢 Core Data Format: 9-Channel Coherency Vector

### Storage Format
- **File type**: NumPy `.npy` (little-endian, C-order)
- **Array shape**: `(9, 128, 128)` — channels first (PyTorch convention)
- **Data type**: `float32`

### Channel Mapping

The 9 channels represent the independent elements of the 3×3 Hermitian coherency matrix **T₃** in Pauli basis:

| Channel Index | Symbol | Physical Meaning | Value Type |
|--------------|--------|-----------------|------------|
| 0 | `T11` | ⟨\|S_HH + S_VV\|²⟩ / 2 | Real, ≥ 0 |
| 1 | `T22` | ⟨\|S_HH - S_VV\|²⟩ / 2 | Real, ≥ 0 |
| 2 | `T33` | ⟨2\|S_HV\|²⟩ | Real, ≥ 0 |
| 3 | `T12_R` | Re(⟨(S_HH+S_VV)(S_HH-S_VV)*⟩ / 2) | Real |
| 4 | `T13_R` | Re(⟨(S_HH+S_VV)(2S_HV)*⟩ / 2) | Real |
| 5 | `T23_R` | Re(⟨(S_HH-S_VV)(2S_HV)*⟩ / 2) | Real |
| 6 | `T12_I` | Im(⟨(S_HH+S_VV)(S_HH-S_VV)*⟩ / 2) | Real |
| 7 | `T13_I` | Im(⟨(S_HH+S_VV)(2S_HV)*⟩ / 2) | Real |
| 8 | `T23_I` | Im(⟨(S_HH-S_VV)(2S_HV)*⟩ / 2) | Real |

### Reconstructing the Full T₃ Matrix

```python
import numpy as np

def vector_to_T3(vec: np.ndarray) -> np.ndarray:
    """
    Convert 9-channel vector to 3×3 complex coherency matrix.
    
    Args:
        vec: np.ndarray of shape (9, H, W) or (9,)
    
    Returns:
        T3: np.ndarray of shape (3, 3, H, W) or (3, 3), dtype=complex64
    """
    if vec.ndim == 2:  # (9, H*W)
        H, W = vec.shape[1], vec.shape[2] if vec.ndim == 3 else 1
        vec = vec.reshape(9, -1)
    
    T11, T22, T33 = vec[0], vec[1], vec[2]
    T12 = vec[3] + 1j * vec[6]
    T13 = vec[4] + 1j * vec[7]
    T23 = vec[5] + 1j * vec[8]
    
    T3 = np.zeros((3, 3) + vec.shape[1:], dtype=np.complex64)
    T3[0, 0] = T11
    T3[1, 1] = T22
    T3[2, 2] = T33
    T3[0, 1] = T12; T3[1, 0] = np.conj(T12)
    T3[0, 2] = T13; T3[2, 0] = np.conj(T13)
    T3[1, 2] = T23; T3[2, 1] = np.conj(T23)
    
    return T3
```

---

## 🔄 Input Representation Schemes

The dataset provides utilities to transform the raw 9-channel vector into four widely-used input formats for deep learning. All transformations are implemented in `tools/data_convert/plosar_format_convert.py`.

### 1. Real-Imaginary Combination (RIC) — `data_type='real'`

**Description**: Direct use of the stored 9-channel vector (3 real diagonals + 3 complex off-diagonals split into real/imaginary parts).

**Output shape**: `(9, 128, 128)` per timestamp → `(18, 128, 128)` after concatenating bi-temporal inputs.

**Usage**:
```python
from datasets.polsar_cd import get_dataset
dataset = get_dataset('data/', mode='train', data_type='real')
```

**Best for**: General-purpose training; lossless representation compatible with standard real-valued CNNs.

---

### 2. Complex-Valued Combination (CRC) — `data_type='complex'`

**Description**: Reconstructs complex off-diagonal elements for use with complex-valued neural networks.

**Transformation**:
```python
# tools/data_convert/plosar_format_convert.py
def sequence_to_complex(seq: np.ndarray) -> np.ndarray:
    """
    Convert 9-channel real sequence to complex tensor.
    
    Input:  (9, H, W) real float32
    Output: (6, H, W) complex64
            [T11, T22, T33, T12, T13, T23] where last 3 are complex
    """
    real_parts = seq[:6]  # T11, T22, T33, T12_R, T13_R, T23_R
    imag_parts = np.zeros_like(real_parts)
    imag_parts[3:] = seq[6:]  # T12_I, T13_I, T23_I
    complex_tensor = real_parts + 1j * imag_parts
    return complex_tensor
```

**Output shape**: `(6, 128, 128)` complex64 per timestamp → `(12, 128, 128)` complex64 after concatenation.

**Usage**:
```python
dataset = get_dataset('data/', mode='train', data_type='complex')
# Returns: sample['data'].dtype == torch.complex64
```

**Best for**: Complex-valued networks (e.g., CUNet) that natively support complex arithmetic.

---

### 3. Amplitude-Phase Combination (APC) — `data_type='ap'`

**Description**: Converts complex off-diagonals to polar coordinates (magnitude + phase).

**Transformation**:
```python
def sequence_to_vector(seq: np.ndarray, is_db: bool = False) -> np.ndarray:
    """
    Convert 9-channel sequence to amplitude-phase representation.
    
    Args:
        seq: (9, H, W) real float32 [T11,T22,T33,T12_R,T13_R,T23_R,T12_I,T13_I,T23_I]
        is_db: if True, apply 10*log10(|x|+ε) to amplitude channels
    
    Returns:
        vec: (9, H, W) real float32
             [T11, T22, T33, |T12|, ∠T12, |T13|, ∠T13, |T23|, ∠T23]
             or dB-scaled amplitudes if is_db=True
    """
    amp = np.sqrt(seq[3]**2 + seq[6]**2)  # |T12|
    phase = np.arctan2(seq[6], seq[3])     # ∠T12
    # Repeat for T13, T23...
    
    if is_db:
        eps = 1e-6
        amp = 10 * np.log10(amp + eps)  # dB scaling
    
    return np.stack([seq[0], seq[1], seq[2], amp12, phase12, amp13, phase13, amp23, phase23])
```

**Output channel order**:
```
[0] T11 (or T11_dB)
[1] T22 (or T22_dB)
[2] T33 (or T33_dB)
[3] |T12| (or |T12|_dB)
[4] ∠T12 ∈ [-π, π]
[5] |T13| (or |T13|_dB)
[6] ∠T13 ∈ [-π, π]
[7] |T23| (or |T23|_dB)
[8] ∠T23 ∈ [-π, π]
```

**Usage**:
```python
# APC (linear amplitude)
dataset_ap = get_dataset('data/', mode='train', data_type='ap')

# LPC (logarithmic amplitude, recommended)
dataset_lpc = get_dataset('data/', mode='train', data_type='ap_db')
```

**Best for**: 
- `ap`: Physical interpretability, phase-sensitive tasks
- `ap_db` ⭐: **Recommended** — best empirical performance (F1=90.71%), robust to dynamic range

---

### 4. C2-Reduced LPC — `data_type='ap_c2_db'`

**Description**: Applies polarization mode reduction (T₃ → C₂) after LPC conversion, yielding a more compact 5-channel representation.

**Transformation**:
```python
# tools/data_convert/polarization_mode_convert.py
def c3toc2(vec_ap: np.ndarray) -> np.ndarray:
    """
    Reduce 9-channel APC/LPC vector to 5-channel C2 representation.
    
    Input:  (9, H, W) in APC/LPC format
    Output: (5, H, W) [T11, T22, |T12|, ∠T12, T33] or dB-scaled
    """
    # Keep diagonal terms and first off-diagonal pair
    return np.stack([vec_ap[0], vec_ap[1], vec_ap[3], vec_ap[4], vec_ap[2]])
```

**Output shape**: `(5, 128, 128)` per timestamp → `(10, 128, 128)` after bi-temporal concatenation.

**Best for**: Lightweight models, ablation studies on polarization dimensionality.

---

## 🏷️ Ground Truth Format

### File Specification
- **Format**: PNG (8-bit grayscale)
- **Shape**: `(128, 128)` — height × width (no channel dimension)
- **Pixel values**:
  - `0`: Unchanged
  - `255`: Changed
  - *Note*: ~3.2% of pixels marked as ambiguous are excluded from evaluation (value=128, ignored in loss)

### Loading Example
```python
import cv2
import torch
from torchvision.transforms import functional as TF

gt = cv2.imread('data/train/gt/pair_0001.png', cv2.IMREAD_GRAYSCALE)  # (128, 128), uint8
gt_tensor = TF.to_tensor(gt)  # (1, 128, 128), float32, values in {0.0, 1.0}
# Convert 255→1.0 automatically; ambiguous pixels (128) should be masked in loss
```

### Loss Masking (Recommended)
```python
def masked_bce_loss(pred, gt, ignore_value=0.5):  # 0.5 corresponds to 128/255
    mask = (gt != ignore_value).float()
    loss = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
    return (loss * mask).sum() / (mask.sum() + 1e-8)
```

---

## 📐 Normalization Statistics

### Channel-wise Z-score Normalization
All real-valued inputs should be normalized using dataset-wide statistics:

```python
# Load pre-computed stats (computed on training set only)
mean = np.load('stats/mean.npy')  # Shape: (9,)
std = np.load('stats/std.npy')    # Shape: (9,)

# Apply normalization (PyTorch-style, channel-wise)
from torchvision.transforms import Normalize
normalize = Normalize(mean=mean.tolist(), std=std.tolist())
```

### Computing Statistics (If Needed)
```bash
python scripts/compute_stats.py --data-root data/ --split train --output stats/
```

**Important**: 
- Statistics are computed **only on the training split** to avoid data leakage
- For `'ap_db'` mode, stats are computed *after* dB conversion
- For `'complex'` mode, real and imaginary parts are normalized independently

---

## 🧭 Coordinate Conventions

### Spatial Dimensions
- **Order**: `(channels, height, width)` — PyTorch convention
- **Origin**: Top-left corner (standard image coordinates)
- **Resolution**: ~5 m (range) × 7 m (azimuth); *not square pixels*

### Geographic Referencing
- **Coordinate system**: WGS84 / UTM (zone varies by region)
- **Geocoding**: Ground-range geometry (not slant-range)
- **Metadata**: See `meta/pairs.json` for per-pair centroid coordinates

```json
{
  "pair_0001": {
    "region": "CV",
    "center_utm": {"easting": 321456.7, "northing": 3887234.1, "zone": "11S"},
    "center_wgs84": {"lon": -119.3012, "lat": 35.1045},
    "acq1_date": "2018-11-27",
    "acq2_date": "2022-11-30",
    "incidence_angle_deg": 42.3
  }
}
```

---

## 🛠️ Utility Functions Reference

### `tools/data_convert/plosar_format_convert.py`

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `matrix_to_sequence(T3)` | `(3,3,H,W)` complex | `(9,H,W)` float | Flatten Hermitian T₃ to 9-channel vector |
| `sequence_to_matrix(vec)` | `(9,H,W)` float | `(3,3,H,W)` complex | Reconstruct T₃ from vector |
| `sequence_to_vector(seq, is_db)` | `(9,H,W)` float | `(9,H,W)` float | RIC → APC/LPC conversion |
| `sequence_to_complex(seq)` | `(9,H,W)` float | `(6,H,W)` complex | RIC → CRC conversion |

### `tools/data_convert/polarization_mode_convert.py`

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `c3toc2(vec_ap)` | `(9,H,W)` APC/LPC | `(5,H,W)` APC/LPC | Reduce T₃-based features to C₂ subspace |
| `c2toc3(vec_c2)` | `(5,H,W)` APC/LPC | `(9,H,W)` APC/LPC | Expand C₂ back to T₃ (zero-padding off-diagonals) |

---

## 🧪 Validation Checklist

When loading data, verify:

```python
import numpy as np

def validate_npy(path: str):
    data = np.load(path)
    assert data.shape == (9, 128, 128), f"Unexpected shape: {data.shape}"
    assert data.dtype == np.float32, f"Unexpected dtype: {data.dtype}"
    assert np.isfinite(data).all(), "Contains NaN/Inf values"
    
    # Check physical constraints (diagonals should be non-negative)
    assert np.all(data[0] >= -1e-6), "T11 has negative values"
    assert np.all(data[1] >= -1e-6), "T22 has negative values"
    assert np.all(data[2] >= -1e-6), "T33 has negative values"
    print("✓ Validation passed")
```

---

## 📬 Support

- **Data format issues**: [GitHub Issues](https://github.com/your-org/WHU-PolSAR-CD/issues)

*Document version: 1.0 | Last updated: January 2026*