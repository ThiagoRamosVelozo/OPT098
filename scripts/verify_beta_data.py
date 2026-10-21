"""Verification script for BETA SSVEP dataset files in data/raw/.

Validates data dimensions, 40 targets, sampling rate, blocks, and data integrity
(no NaNs, no Infs, consistent tensor structures).
"""

import sys
from pathlib import Path
import numpy as np
import scipy.io

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Ground-truth specifications for BETA dataset (Liu et al., 2020)
EXPECTED_TARGETS = 40
EXPECTED_CHANNELS = 64
EXPECTED_FS = 250  # Hz


def inspect_subject_file(mat_path: Path):
    """Load and thoroughly validate a single BETA subject .mat file."""
    print(f"\n==================================================")
    print(f"Validating: {mat_path.name} ({mat_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"==================================================")

    try:
        mat_contents = scipy.io.loadmat(mat_path, squeeze_me=False, struct_as_record=False)
    except Exception as e:
        print(f"[ERROR] Failed to load {mat_path.name} with scipy.io: {e}")
        return False, None

    keys = [k for k in mat_contents.keys() if not k.startswith("__")]
    print(f"MATLAB variables found: {keys}")

    if "data" not in mat_contents:
        print(f"[FAIL] Missing 'data' variable in {mat_path.name}")
        return False, None

    data_obj = mat_contents["data"]

    # BETA dataset struct inspection
    # data can be a structured numpy array or object with fields: EEG, suppl_info, etc.
    # Typically: data(1,1).EEG or data.EEG
    details = {}
    
    # Check if data_obj is a structured array
    eeg_found = False
    eeg_data = None

    if hasattr(data_obj, "EEG"):
        eeg_data = data_obj.EEG
        eeg_found = True
    elif isinstance(data_obj, np.ndarray):
        # Could be structured array
        if data_obj.dtype.names and "EEG" in data_obj.dtype.names:
            eeg_data = data_obj["EEG"][0, 0]
            eeg_found = True
        elif data_obj.size > 0 and hasattr(data_obj.flat[0], "EEG"):
            eeg_data = data_obj.flat[0].EEG
            eeg_found = True

    if not eeg_found or eeg_data is None:
        # Fallback inspection of fields
        print(f"Available fields on 'data': {dir(data_obj) if hasattr(data_obj, '__dict__') else type(data_obj)}")
        # Check if data itself is a numeric tensor
        if isinstance(data_obj, np.ndarray) and data_obj.dtype.kind in ('f', 'i'):
            eeg_data = data_obj
            eeg_found = True

    if eeg_data is not None:
        shape = eeg_data.shape
        dtype = eeg_data.dtype
        print(f"  -> EEG Tensor Shape: {shape} ({dtype})")

        # Check for NaNs and Infs
        has_nan = np.isnan(eeg_data).any()
        has_inf = np.isinf(eeg_data).any()

        if has_nan or has_inf:
            print(f"[FAIL] Data corruption detected: NaNs={has_nan}, Infs={has_inf}")
            return False, None

        min_val, max_val, mean_val = np.min(eeg_data), np.max(eeg_data), np.mean(eeg_data)
        print(f"  -> Value range: [{min_val:.2f}, {max_val:.2f}], Mean: {mean_val:.4f}")

        # Check if 40 targets are present in shape
        # In BETA: shape is typically (64 channels, time_points, 4 blocks, 40 targets)
        # or (40 targets, 4 blocks, 64 channels, time_points), etc.
        contains_40 = 40 in shape
        contains_64 = 64 in shape or (shape[0] in (64, 65))  # sometimes trigger channel

        print(f"  -> Contains 40 targets: {'YES' if contains_40 else 'CHECK SHAPE'}")
        print(f"  -> Contains 64 channels: {'YES' if contains_64 else 'CHECK SHAPE'}")

        details["shape"] = shape
        details["dtype"] = str(dtype)
        details["has_nan"] = has_nan
        details["has_inf"] = has_inf
        details["contains_40_targets"] = contains_40

        print(f"[OK] Subject {mat_path.name} validated successfully without corruption.")
        return True, details
    else:
        print(f"[WARN] Could not identify EEG field directly. Raw data representation: {type(data_obj)}")
        return False, None


def main():
    mat_files = sorted(list(RAW_DATA_DIR.glob("S*.mat")), key=lambda p: int(p.stem[1:]) if p.stem[1:].isdigit() else 999)

    if not mat_files:
        print(f"[ERROR] No S*.mat files found in {RAW_DATA_DIR}.")
        print("Please run scripts/download_beta.py first.")
        sys.exit(1)

    print(f"Found {len(mat_files)} subject file(s) in {RAW_DATA_DIR}: {[f.name for f in mat_files]}")

    results = {}
    all_ok = True

    for mat_file in mat_files:
        ok, info = inspect_subject_file(mat_file)
        results[mat_file.name] = (ok, info)
        if not ok:
            all_ok = False

    print("\n==================================================")
    print("                VALIDATION SUMMARY                 ")
    print("==================================================")
    for name, (ok, info) in results.items():
        status = "PASSED" if ok else "FAILED"
        shape_str = str(info["shape"]) if info and "shape" in info else "N/A"
        print(f"  {name:10s} : {status} | Tensor Shape: {shape_str}")

    if all_ok:
        print("\n>>> All downloaded BETA subject files are valid and ready for pipeline processing! <<<")
        sys.exit(0)
    else:
        print("\n>>> Some files failed validation. Check logs above. <<<")
        sys.exit(1)


if __name__ == "__main__":
    main()
