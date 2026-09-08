RAPIDS_cuDF_cuML_Guide.md
# RAPIDS cuDF & cuML: Beginner's Guide

## What is RAPIDS?

RAPIDS is an **open-source GPU-accelerated data science framework** that makes data analysis and ML 10-100x faster by leveraging NVIDIA GPUs. It provides familiar Python APIs (pandas, scikit-learn) without requiring CUDA programming knowledge.

**Architecture:**
```
Python APIs (cuDF/cuML) → Apache Arrow → CUDA → GPU Hardware
```

---

## cuDF: GPU-Accelerated DataFrames

**What**: GPU-accelerated pandas for data manipulation.

**Performance**: 78x faster GroupBy, 72x faster value_counts, 16x faster string ops.

**How it works**: Columnar Apache Arrow format + parallel CUDA kernels on GPU.

### Basic Usage

```python
import cudf

# Same API as pandas
df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
result = df.groupby('a').sum()
```

### Zero-Code Acceleration with cudf.pandas

```python
# Add this line at top of notebook (affects entire session)
%load_ext cudf.pandas

import pandas as pd  # Auto-accelerated on GPU!
df = pd.read_csv('data.csv')  # Runs on GPU
df.groupby('col').mean()  # GPU-accelerated
```

**Note**: `%load_ext` is an IPython magic that loads the extension into the kernel. Once loaded, it affects **all cells** in the notebook. Best practice: load in first cell before any imports.

---

## cuML: GPU-Accelerated Machine Learning

**What**: GPU-accelerated scikit-learn with 50+ algorithms.

**Performance**: 27x faster Linear Regression, 24x faster PCA, 6x faster K-Means.

**Algorithms**: Clustering (K-Means, DBSCAN), Classification (RandomForest, SVM), Regression (Linear, Ridge), Dimensionality Reduction (PCA, UMAP, t-SNE).

### Basic Usage

```python
from cuml.cluster import DBSCAN
from cuml.datasets import make_blobs

X, y = make_blobs(n_samples=10000, centers=5, n_features=10)
dbscan = DBSCAN(eps=1.0, min_samples=5)
dbscan.fit(X)
print(f"Found {dbscan.labels_.max() + 1} clusters")
```

### Drop-in Replacement

```python
# Just change import - same code!
from cuml.ensemble import RandomForestClassifier  # Instead of sklearn
clf = RandomForestClassifier()
clf.fit(X_train, y_train)  # 10-50x faster
```

---

## Complete Pipeline Example

```python
import cudf
from cuml.preprocessing import StandardScaler
from cuml.linear_model import LogisticRegression
from cuml.model_selection import train_test_split

# Load, split, scale, train - all on GPU
df = cudf.read_csv('dataset.csv')
X, y = df.drop('target', axis=1), df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2f}")
```

---

## Installation & Setup

### System Requirements

- **GPU**: NVIDIA with Compute Capability **7.0+** (Volta or newer)
- **OS**: Linux (Ubuntu 20.04+), Windows 11 WSL2 (macOS not supported)
- **CUDA**: 12.0+ (Driver 525.60.13+) or 13.0+ (Driver 580.65.06+)

**Check compatibility:**
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
# Output example: Tesla T4, 7.5 ✅ (supported)
```

---

### Supported GPU Models

| Category | GPU Models | Compute Cap | Status |
|----------|-----------|-------------|--------|
| **Cloud/Data Center** | T4, V100, A100, H100, L4, L40 | 7.0 - 10.3 | ✅ Supported |
| **Consumer (RTX)** | RTX 20/30/40/50 series | 7.5 - 12.0 | ✅ Supported |
| **Workstation** | RTX A-series, Quadro RTX | 7.5 - 8.9 | ✅ Supported |
| **Edge** | Jetson AGX/Orin | 8.7 | ✅ Supported |
| **Older GPUs** | GTX 10 series, K80, P100 | < 7.0 | ❌ Not Supported |

**Popular Choices**:
- **Cloud**: T4 (AWS g4dn, GCP, Azure), A100 (premium)
- **Desktop**: RTX 3060/4060+ (8GB+ VRAM recommended)
- **Free**: Google Colab (T4 free tier)

---

### Installation Methods

#### 1. Conda (Recommended)

```bash
# Install Miniforge (if needed)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh && source ~/.bashrc

# Create environment (CUDA 12)
conda create -n rapids-24.12 -c rapidsai -c conda-forge -c nvidia \
    rapids=24.12 python=3.11 cuda-version=12.0

# Or minimal install (just cuDF + cuML)
conda create -n rapids-minimal -c rapidsai -c conda-forge -c nvidia \
    cudf=24.12 cuml=24.12 python=3.11 cuda-version=12.0

conda activate rapids-24.12

# Optimize conda performance
conda config --set channel_priority flexible
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

#### 2. pip (Alternative)

```bash
python3 -m venv rapids-env && source rapids-env/bin/activate

# CUDA 12.x
pip install cudf-cu12 cuml-cu12

# CUDA 11.x
pip install cudf-cu11 cuml-cu11
```

#### 3. Docker

```bash
# Install NVIDIA Container Toolkit first
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Run RAPIDS
docker pull rapidsai/base:24.12-cuda12.0-py3.11
docker run --gpus all --rm -it -p 8888:8888 rapidsai/base:24.12-cuda12.0-py3.11

# With Jupyter notebooks
docker pull rapidsai/notebooks:24.12-cuda12.0-py3.11
docker run --gpus all --rm -it -p 8888:8888 -v $(pwd):/workspace \
    rapidsai/notebooks:24.12-cuda12.0-py3.11
```

#### 4. Google Colab (Free GPU)

```python
!git clone https://github.com/rapidsai/rapidsai-csp-utils.git
!python rapidsai-csp-utils/colab/pip-install.py
import cudf
```

---

### Quick Reference Table

| Use Case | Method | Command |
|----------|--------|---------|
| Production | Conda | `conda create -n rapids -c rapidsai rapids` |
| Development | Conda minimal | `conda create -n rapids cudf cuml` |
| Learning | Google Colab | Use Colab notebook |
| Container | Docker | `docker pull rapidsai/base` |
| Existing CUDA | pip | `pip install cudf-cu12 cuml-cu12` |

---

### Verify Installation

```python
import cudf, cuml, cupy as cp

# Test cuDF
df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print(f"cuDF {cudf.__version__}: {len(df)} rows")

# Test cuML
from cuml.datasets import make_blobs
X, y = make_blobs(100, 5)
print(f"cuML {cuml.__version__}: Created {len(X)} samples")

# Check GPU
print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
print("✅ RAPIDS working!")
```

---

### Troubleshooting

**Issue**: CUDA driver insufficient
```bash
sudo apt install nvidia-driver-535 && sudo reboot
```

**Issue**: libcuda.so.1 not found
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

**Issue**: Out of Memory
```python
import cupy as cp
cp.get_default_memory_pool().set_limit(size=8*1024**3)  # 8GB limit
```

**Issue**: Conda slow
```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

---

## When to Use RAPIDS

**✅ Great for:**
- Large datasets (1M+ rows)
- ETL pipelines
- ML training on big data
- Real-time analytics

**❌ Skip if:**
- Small datasets (< 100K rows)
- No NVIDIA GPU available
- Operations unsupported by RAPIDS

---

## Key Takeaways

1. **cuDF** = GPU pandas (78x faster groupby)
2. **cuML** = GPU sklearn (27x faster regression)
3. **Same API** = Minimal code changes
4. **cudf.pandas** = Zero-code GPU acceleration
5. **Open-source** = Free, active community

---

## Resources

- [Official Docs](https://docs.rapids.ai/)
- [GitHub](https://github.com/rapidsai)
- [Installation Guide](https://rapids.ai/start.html)
- [Example Notebooks](https://github.com/rapidsai/notebooks)

**Next Steps**: Try `%load_ext cudf.pandas` in a notebook for instant GPU acceleration!
