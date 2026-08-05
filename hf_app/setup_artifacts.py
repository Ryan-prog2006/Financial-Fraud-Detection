"""Utility script to copy model and RAG artifacts to hf_app/artifacts/."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Paths to source files
SRC_MODEL = Path("data/models/xgboost_tuned.pkl")
SRC_SCALER = Path("data/processed/scaler.pkl")
SRC_VECTORSTORE = Path("data/vectorstore")
SRC_POLICY = Path("data/policy_docs")

def get_dir_size(path: Path) -> int:
    total = 0
    if path.is_file():
        return path.stat().st_size
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total

def copy_resource(src: Path, dest: Path) -> None:
    if not src.exists():
        print(f"Warning: Source path {src} does not exist. Skipping.")
        return
        
    if dest.exists():
        if dest.is_dir():
            shutil.rmtree(dest)
        else:
            dest.unlink()
            
    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        shutil.copy2(src, dest)
        
    size_mb = get_dir_size(dest) / (1024 * 1024)
    print(f"Copied: {src} -> {dest} (Size: {size_mb:.2f} MB)")

def main() -> None:
    print("Setting up Hugging Face app artifacts...")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # Copy resources
    copy_resource(SRC_MODEL, ARTIFACTS_DIR / "model.pkl")
    copy_resource(SRC_SCALER, ARTIFACTS_DIR / "scaler.pkl")
    copy_resource(SRC_VECTORSTORE, ARTIFACTS_DIR / "vectorstore")
    copy_resource(SRC_POLICY, ARTIFACTS_DIR / "policy_docs")
    
    # Create gitkeep
    with open(ARTIFACTS_DIR / ".gitkeep", "w") as f:
        f.write("")
        
    print("Hugging Face app artifacts setup completed successfully.")

if __name__ == "__main__":
    main()
