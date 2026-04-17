"""
Download TextZoom dataset.

TextZoom GitHub: https://github.com/WenjiaWang0312/TextZoom
Dataset is stored in LMDB format with train/test splits.

Expected structure after download:
  data/TextZoom/
    train/
      easy/   -> LMDB database
      medium/ -> LMDB database
      hard/   -> LMDB database
    test/
      easy/   -> LMDB database
      medium/ -> LMDB database
      hard/   -> LMDB database
"""

import os
import sys
import gdown
import zipfile
import argparse
from pathlib import Path


# Google Drive file IDs for TextZoom
# Source: https://github.com/WenjiaWang0312/TextZoom
TEXTZOOM_FILES = {
    "train_easy":   "1nAKjK9FKNO0BQMKKO77ME9pMBqVLBqfH",
    "train_medium": "1EomEsX0bzPAizpKv1GkKsxTcj7M2GVzU",
    "train_hard":   "1JBD0lDRNvYrGHvw0LQ6J1JHE8iX9Q8PQ",
    "test_easy":    "16UDnCCMO8j7AkGD5CjHVFcABl3sO4bsV",
    "test_medium":  "1L0A-lIrOWqnkQGQgFaRuhliYdZVHoSKy",
    "test_hard":    "18GBR3BOBGr_dGpP-PYuJpGVRNVEPPXBK",
}

# Alternative: download the full dataset as single archive
FULL_DATASET_ID = "1NxoZcO8J9gzxhScg6xRHoqFqrExG1-gA"


def download_from_gdrive(file_id: str, output_path: str):
    """Download a file from Google Drive."""
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)


def download_textzoom(data_root: str, split: str = "all"):
    """
    Download TextZoom dataset.

    Args:
        data_root: Root directory to save dataset
        split: "train", "test", or "all"
    """
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    print(f"Downloading TextZoom dataset to {data_root}")
    print("=" * 60)
    print("NOTE: TextZoom is hosted on Google Drive.")
    print("If gdown fails, please manually download from:")
    print("https://github.com/WenjiaWang0312/TextZoom")
    print("=" * 60)

    splits_to_download = []
    if split in ("train", "all"):
        splits_to_download.extend(["train_easy", "train_medium", "train_hard"])
    if split in ("test", "all"):
        splits_to_download.extend(["test_easy", "test_medium", "test_hard"])

    for key in splits_to_download:
        split_name, difficulty = key.split("_")
        out_dir = data_root / split_name / difficulty
        out_dir.mkdir(parents=True, exist_ok=True)

        lmdb_path = out_dir / "data.mdb"
        if lmdb_path.exists():
            print(f"  [SKIP] {key} already exists at {out_dir}")
            continue

        print(f"\n  Downloading {key}...")
        zip_path = str(data_root / f"{key}.zip")
        try:
            download_from_gdrive(TEXTZOOM_FILES[key], zip_path)
            print(f"  Extracting {key}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(str(out_dir))
            os.remove(zip_path)
            print(f"  Done: {out_dir}")
        except Exception as e:
            print(f"  ERROR downloading {key}: {e}")
            print(f"  Please download manually and place LMDB files in {out_dir}")

    print("\nDownload complete!")
    print_dataset_info(data_root)


def print_dataset_info(data_root: Path):
    """Print info about the downloaded dataset."""
    import lmdb
    print("\n=== TextZoom Dataset Info ===")
    for split in ["train", "test"]:
        for difficulty in ["easy", "medium", "hard"]:
            lmdb_dir = data_root / split / difficulty
            if not lmdb_dir.exists():
                continue
            try:
                env = lmdb.open(str(lmdb_dir), readonly=True, lock=False)
                with env.begin() as txn:
                    n = txn.stat()["entries"]
                print(f"  {split}/{difficulty}: {n // 2} samples")  # Each sample has hr+lr keys
                env.close()
            except Exception:
                print(f"  {split}/{difficulty}: (exists but can't read)")


def manual_download_instructions():
    """Print manual download instructions."""
    print("""
Manual Download Instructions for TextZoom:
==========================================

1. Visit: https://github.com/WenjiaWang0312/TextZoom

2. Download the dataset from the links in the README:
   - BaiduYun: https://pan.baidu.com/s/1KSNLv4EY3zFWHpBYlpFoDA (code: m6uk)
   - Google Drive: See links in the GitHub README

3. After downloading, organize as:
   data/TextZoom/
     train/
       easy/    (LMDB files: data.mdb, lock.mdb)
       medium/  (LMDB files: data.mdb, lock.mdb)
       hard/    (LMDB files: data.mdb, lock.mdb)
     test/
       easy/    (LMDB files: data.mdb, lock.mdb)
       medium/  (LMDB files: data.mdb, lock.mdb)
       hard/    (LMDB files: data.mdb, lock.mdb)

4. Run: python scripts/download_textzoom.py --check

TextZoom Statistics:
  Train: ~17,367 image pairs (easy/medium/hard splits)
  Test:  ~3,021 image pairs
  LR size: varies, typically 32×128 pixels
  HR size: varies, typically 2× LR resolution
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download TextZoom dataset")
    parser.add_argument("--data_root", default="./data/TextZoom", help="Download destination")
    parser.add_argument("--split", default="all", choices=["train", "test", "all"])
    parser.add_argument("--check", action="store_true", help="Only check existing data")
    parser.add_argument("--manual", action="store_true", help="Print manual download instructions")
    args = parser.parse_args()

    if args.manual:
        manual_download_instructions()
    elif args.check:
        print_dataset_info(Path(args.data_root))
    else:
        download_textzoom(args.data_root, args.split)
