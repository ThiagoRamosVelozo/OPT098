"""Script to verify python environment setup and dependencies."""

import sys

def verify():
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")

    try:
        import torch
        print(f"[OK] PyTorch version: {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
        if torch.cuda.is_available():
            print(f"     GPU: {torch.cuda.get_device_name(0)}")
        
        # Test basic tensor computation
        x = torch.randn(2, 3)
        y = torch.matmul(x, x.T)
        print(f"[OK] PyTorch tensor test passed: shape {y.shape}")
    except Exception as e:
        print(f"[FAIL] PyTorch check failed: {e}")
        return False

    try:
        import scipy
        import scipy.io
        print(f"[OK] SciPy version: {scipy.__version__}")
    except Exception as e:
        print(f"[FAIL] SciPy check failed: {e}")
        return False

    try:
        import mne
        print(f"[OK] MNE version: {mne.__version__}")
    except Exception as e:
        print(f"[FAIL] MNE check failed: {e}")
        return False

    try:
        import numpy as np
        import pandas as pd
        import matplotlib
        import seaborn as sns
        import sklearn
        print(f"[OK] NumPy {np.__version__}, Pandas {pd.__version__}, Matplotlib {matplotlib.__version__}, Seaborn {sns.__version__}, Scikit-Learn {sklearn.__version__}")
    except Exception as e:
        print(f"[FAIL] Auxiliary libraries check failed: {e}")
        return False

    print("\n>>> All environment dependencies verified successfully! <<<")
    return True

if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
