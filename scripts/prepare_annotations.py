"""
Pre-compute PaddleOCR annotations for TextZoom dataset.

Run this ONCE before training to cache OCR results and speed up data loading.
Saves annotations to data/ocr_annotations/{split}_ocr.json

Usage:
  python scripts/prepare_annotations.py --data_root ./data/TextZoom \\
                                         --output_dir ./data/ocr_annotations \\
                                         --split train \\
                                         --use_gpu
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import LMDBReader, ocr_image, pil_to_numpy


def annotate_split(
    data_root: str,
    split: str,
    output_dir: str,
    difficulties: tuple = ("easy", "medium", "hard"),
    use_gpu: bool = False,
    save_every: int = 500,
):
    """Run OCR on all LR images in a dataset split and cache results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = output_dir / f"{split}_ocr.json"

    # Load existing annotations
    annotations: dict = {}
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        print(f"Loaded {len(annotations)} existing annotations from {cache_file}")

    global_idx = 0
    for diff in difficulties:
        lmdb_path = os.path.join(data_root, split, diff)
        if not os.path.isdir(lmdb_path):
            print(f"[SKIP] {lmdb_path} not found")
            continue

        reader = LMDBReader(lmdb_path)
        n = len(reader)
        print(f"\n--- {split}/{diff}: {n} samples ---")

        for local_idx in tqdm(range(n), desc=f"{split}/{diff}"):
            key = str(global_idx)
            if key in annotations:
                global_idx += 1
                continue

            sample = reader.get(local_idx)
            lr_pil = sample.get("lr")
            gt_label = sample.get("label", "")

            if lr_pil is None:
                annotations[key] = gt_label or ""
                global_idx += 1
                continue

            lr_np = pil_to_numpy(lr_pil)
            text = ocr_image(lr_np, use_gpu=use_gpu)
            annotations[key] = text

            global_idx += 1

            # Periodic save
            if global_idx % save_every == 0:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(annotations, f, ensure_ascii=False)

    # Final save
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(annotations)} annotations to {cache_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data/TextZoom")
    parser.add_argument("--output_dir", default="./data/ocr_annotations")
    parser.add_argument("--split", default="train", choices=["train", "test", "all"])
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    splits = ["train", "test"] if args.split == "all" else [args.split]
    for split in splits:
        annotate_split(
            data_root=args.data_root,
            split=split,
            output_dir=args.output_dir,
            use_gpu=args.use_gpu,
        )
