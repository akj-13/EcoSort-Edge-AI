"""
Dataset Download & Extraction Helper for EcoSort Edge-AI Challenge.

Usage:
    python starter_kit/download_data.py
"""

import os
import shutil
import sys
import zipfile

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
PUBLIC_ZIP = os.path.join(DATA_DIR, "ecosort_dataset_public.zip")
TARGET_DIR = os.path.join(DATA_DIR, "public")


def unpack_dataset():
    print("=" * 60)
    print("📦 EcoSort Dataset Setup")
    print("=" * 60)

    train_dir = os.path.join(TARGET_DIR, "train")
    val_dir = os.path.join(TARGET_DIR, "val")

    if os.path.exists(train_dir) and os.path.exists(val_dir):
        print(f"Dataset already ready at: {TARGET_DIR}")
        print_dataset_stats(train_dir, val_dir)
        return

    if os.path.exists(PUBLIC_ZIP):
        print(f"Extracting local archive {PUBLIC_ZIP}...")
        os.makedirs(TARGET_DIR, exist_ok=True)
        with zipfile.ZipFile(PUBLIC_ZIP, "r") as zip_ref:
            zip_ref.extractall(TARGET_DIR)
        print("Extraction complete!")
        print_dataset_stats(train_dir, val_dir)
        return

    print("Generating local benchmark dataset directly via prepare_dataset.py...")
    prep_script = os.path.join(DATA_DIR, "prepare_dataset.py")
    if os.path.exists(prep_script):
        import subprocess
        subprocess.run([sys.executable, prep_script], check=True)
        print("Dataset generated successfully!")
        print_dataset_stats(train_dir, val_dir)
    else:
        print(f"Error: Neither {PUBLIC_ZIP} nor {prep_script} found.")
        sys.exit(1)


def print_dataset_stats(train_dir: str, val_dir: str):
    print("\nDataset Summary:")
    for split_name, s_dir in [("Train", train_dir), ("Validation", val_dir)]:
        if os.path.exists(s_dir):
            classes = sorted(os.listdir(s_dir))
            total = sum(len(os.listdir(os.path.join(s_dir, c))) for c in classes if os.path.isdir(os.path.join(s_dir, c)))
            print(f"  • {split_name} Split: {total:,} images across {len(classes)} classes {classes}")
    print("\nYou are ready to start training with: python train_template.py\n")


if __name__ == "__main__":
    unpack_dataset()
