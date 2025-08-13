#!/usr/bin/env python3
"""
promote_model.py
Copies a trained model artifact from logs/experiments to the registry
for production serving.
"""

import shutil
from pathlib import Path
import sys
#-----For use in .py files-----#
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#-----For use in .ipynb files-----#
#PROJECT_ROOT = Path().resolve().parents[0]

#sys.path.append(str(PROJECT_ROOT / 'src'))

# ==== USER CONFIG (edit these for your setup) ====
MODEL_NAME = "HousePrices_regressor"  # descriptive name for your model
MODEL_FILENAME = "pipeline.joblib"  # name of your serialized model in logs
EXP_CONFIG_FILENAME = "experiment_config.json"  # name of the experiment config file
VAL_RESULTS_FILENAME = "val_results.json"  # name of the validation results file
# ================================================

def copy_item(src, dest):
    """
    Copies a file or directory from src to dest.
    If src is a directory, it copies the entire directory recursively.
    """
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dest)
    print(f"Copied {src} to {dest}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python promote_model.py <experiment_dir> <version>")
        print("Example: python promote_model.py 2025-08-12_134455 v1")
        sys.exit(1)

    FILENAMES = [
        MODEL_FILENAME,
        EXP_CONFIG_FILENAME,
        VAL_RESULTS_FILENAME
    ]
    exp_dir = PROJECT_ROOT / "logs" / sys.argv[1]
    version = sys.argv[2]

    src_model_path = exp_dir / MODEL_FILENAME
    if not src_model_path.exists():
        print(f"❌ Model file not found: {src_model_path}")
        sys.exit(1)
    dest_dir = PROJECT_ROOT / "registry" / MODEL_NAME / version
    dest_dir.mkdir(parents=True, exist_ok=True)
    for filename in FILENAMES:
        src_path = exp_dir / filename
        dest_path = dest_dir / filename
        copy_item(src_path, dest_path)

    print(f"✅ All files copied model to {dest_dir}")

if __name__ == "__main__":
    main()

