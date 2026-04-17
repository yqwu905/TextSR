"""
Download TextZoom dataset.

TextZoom GitHub: https://github.com/WenjiaWang0312/TextZoom
Dataset is stored in LMDB format with train/test splits.

Expected structure after download:
  data/TextZoom/
    train1/   -> LMDB database
    train2/   -> LMDB database
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
    "train1":    "1nAKjK9FKNO0BQMKKO77ME9pMBqVLBqfH",
    "train2":    "1EomEsX0bzPAizpKv1GkKsxTcj7M2GVzU",
    "test_easy":    "16UDnCCMO8j7AkGD5CjHVFcABl3sO4bsV",
    "test_medium":  "1L0A-lIrOWqnkQGQgFaRuhliYdZVHoSKy",
    "test_hard":    "18GBR3BOBGr_dGpP-PYuJpGVRNVEPPXBK",
}


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

    # Build list of (key, out_dir) to download
    to_download = []
    if split in ("train", "all"):
        to_download.append(("train1", data_root / "train1"))
        to_download.append(("train2", data_root / "train2"))
    if split in ("test", "all"):
        to_download.append(("test_easy",   data_root / "test" / "easy"))
        to_download.append(("test_medium", data_root / "test" / "medium"))
        to_download.append(("test_hard",   data_root / "test" / "hard"))

    for key, out_dir in to_download:
        out_dir.mkdir(parents=True, exist_ok=True)

        if (out_dir / "data.mdb").exists():
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
    # Train: flat train1 / train2
    for name in ["train1", "train2"]:
        lmdb_dir = data_root / name
        if not lmdb_dir.exists():
            continue
        try:
            env = lmdb.open(str(lmdb_dir), readonly=True, lock=False)
            with env.begin() as txn:
                n = txn.stat()["entries"]
            print(f"  {name}: {n // 2} samples")
            env.close()
        except Exception:
            print(f"  {name}: (exists but can't read)")
    # Test: test/easy, test/medium, test/hard
    for difficulty in ["easy", "medium", "hard"]:
        lmdb_dir = data_root / "test" / difficulty
        if not lmdb_dir.exists():
            continue
        try:
            env = lmdb.open(str(lmdb_dir), readonly=True, lock=False)
            with env.begin() as txn:
                n = txn.stat()["entries"]
            print(f"  test/{difficulty}: {n // 2} samples")
            env.close()
        except Exception:
            print(f"  test/{difficulty}: (exists but can't read)")


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
     train1/  (LMDB files: data.mdb, lock.mdb)
     train2/  (LMDB files: data.mdb, lock.mdb)
     test/
       easy/    (LMDB files: data.mdb, lock.mdb)
       medium/  (LMDB files: data.mdb, lock.mdb)
       hard/    (LMDB files: data.mdb, lock.mdb)

4. Run: python scripts/download_textzoom.py --check

TextZoom Statistics:
  Train: ~17,367 image pairs across train1 + train2
  Test:  ~3,021 image pairs (easy/medium/hard)
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
