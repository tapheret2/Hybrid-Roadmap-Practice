# Environment Setup Guide

## Quick Start

### Python (Data Science & Backend)
```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install all packages
pip install -r requirements.txt

# Or install only core packages (faster)
pip install numpy pandas matplotlib scikit-learn jupyter
```

### Node.js (Frontend & Express)
```bash
# Install dependencies
npm install

# Or using yarn
yarn install
```

---

## Package Groups

### Core DS (Minimum)
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### ML Extended
```bash
pip install xgboost lightgbm optuna mlflow
```

### Deep Learning (Large, Optional)
```bash
pip install torch torchvision
# OR
pip install tensorflow
```

### API Development
```bash
pip install fastapi uvicorn pydantic
```

---

## Verify Installation

```python
# test_setup.py
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print("✅ Core packages installed successfully!")
```

---

## Troubleshooting

### Windows: Visual C++ Build Tools
Some packages need C++ compiler:
- Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/

### macOS: Xcode Command Line Tools
```bash
xcode-select --install
```

### GPU Support (CUDA)
```bash
# PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
