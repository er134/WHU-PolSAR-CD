"""
Dataset splitting scripts for WHU-PolSAR-CD.

Provides two splitting strategies described in the paper:
  1. Spatially disjoint split (crops_no_border_split):
     - Each scene is split along its spatial extent by patch index order
       into five equal parts; the 2nd fifth → test, the 4th fifth → valid,
       and the remaining (1st, 3rd, 5th) → train
     - Guarantees no spatial leakage between subsets

  2. Few-shot split (0.01_seq_notest_2):
     - 23 training images (~1% of 2,468) are randomly selected and placed
       in train/{pre,next,gt}/; the remaining images go to valid/{pre,next,gt}/
     - train_list.txt records the 23 selected sample names

Usage:
  # Spatially disjoint split
  python tools/data_convert/split_dataset.py spatial \
      --src /path/to/crops_no_border \
      --dst /path/to/crops_no_border_split

  # 1% few-shot split
  python tools/data_convert/split_dataset.py fewshot \
      --src /path/to/crops_no_border \
      --dst /path/to/0.01_seq_notest_2 \
      --n-train 23 --seed 42

  # Both splits at once
  python tools/data_convert/split_dataset.py all \
      --src /path/to/crops_no_border \
      --dst-spatial /path/to/crops_no_border_split \
      --dst-fewshot /path/to/0.01_seq_notest_2 \
      --seed 42
"""

import argparse
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


SCENE_NAMES = [
    "CValle", "Cochis", "Icelnd", "SnJoaq", "Yukon", "padelW",
    "santab1", "santab2",
]

SUBDIRS_DATA = ["pre", "next", "gt"]
EXT_MAP = {"pre": ".npy", "next": ".npy", "gt": ".png"}


def list_samples(src: Path):
    """Return {scene: [sorted_indices]} from the pre/ directory."""
    scene_indices = defaultdict(list)
    for f in sorted(os.listdir(src / "pre")):
        if not f.endswith(".npy"):
            continue
        stem = f[:-4]
        for scene in SCENE_NAMES:
            if stem.startswith(scene + "_"):
                idx = int(stem[len(scene) + 1 :])
                scene_indices[scene].append(idx)
                break
    for scene in scene_indices:
        scene_indices[scene].sort()
    return dict(scene_indices)


def spatial_split(src: Path, dst: Path):
    """Create spatially disjoint train/valid/test split.

    For each scene, patches are sorted by index (which corresponds to
    spatial order in the original large-scale image).  The spatial
    extent is divided into five equal parts:
      - 2nd fifth → test
      - 4th fifth → valid
      - remaining (1st, 3rd, 5th) → train
    This ensures that training, validation, and test regions are
    spatially interleaved rather than contiguous.
    """
    scene_indices = list_samples(src)
    total = sum(len(v) for v in scene_indices.values())
    print(f"Found {total} samples across {len(scene_indices)} scenes")

    for split in ["train", "valid", "test"]:
        for subdir in SUBDIRS_DATA:
            (dst / split / subdir).mkdir(parents=True, exist_ok=True)

    split_stats = {"train": 0, "valid": 0, "test": 0}

    for scene, indices in sorted(scene_indices.items()):
        n = len(indices)
        b1 = n // 5
        b2 = 2 * n // 5
        b3 = 3 * n // 5
        b4 = 4 * n // 5
        train_idx = set(indices[:b1] + indices[b2:b3] + indices[b4:])
        test_idx = set(indices[b1:b2])
        valid_idx = set(indices[b3:b4])

        for idx in indices:
            if idx in train_idx:
                split = "train"
            elif idx in valid_idx:
                split = "valid"
            else:
                split = "test"

            stem = f"{scene}_{idx}"
            for subdir in SUBDIRS_DATA:
                ext = EXT_MAP[subdir]
                src_file = src / subdir / (stem + ext)
                dst_file = dst / split / subdir / (stem + ext)
                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
                else:
                    print(f"  WARNING: {src_file} not found")

            split_stats[split] += 1

        print(
            f"  {scene}: {n} total → train={len(train_idx)}, "
            f"valid={len(valid_idx)}, test={len(test_idx)}"
        )

    print(
        f"\nSpatial split done: train={split_stats['train']}, "
        f"valid={split_stats['valid']}, test={split_stats['test']}"
    )


def fewshot_split(
    src: Path, dst: Path, ratio: float = 0.01, seed: int = 42, n_train: int = None
):
    """Create 1% few-shot split for CPSSL.

    Selects ``n_train`` samples for training (placed in train/), and
    the remaining samples for validation (placed in valid/).
    A train_list.txt records the selected training sample names.
    """
    scene_indices = list_samples(src)
    total = sum(len(v) for v in scene_indices.values())
    if n_train is None:
        n_train = max(1, int(total * ratio))
    print(f"Total samples: {total}, selecting {n_train} for training")

    rng = random.Random(seed)

    all_stems = []
    for scene, indices in sorted(scene_indices.items()):
        for idx in indices:
            all_stems.append(f"{scene}_{idx}")

    train_stems = set(rng.sample(all_stems, n_train))

    for split in ["train", "test"]:
        for subdir in SUBDIRS_DATA:
            (dst / split / subdir).mkdir(parents=True, exist_ok=True)

    for stem in all_stems:
        split = "train" if stem in train_stems else "test"
        for subdir in SUBDIRS_DATA:
            ext = EXT_MAP[subdir]
            src_file = src / subdir / (stem + ext)
            dst_file = dst / split / subdir / (stem + ext)
            if src_file.exists():
                shutil.copy2(src_file, dst_file)

    train_list_path = dst / "train_list.txt"
    with open(train_list_path, "w") as f:
        for stem in sorted(train_stems):
            f.write(stem + "\n")

    scene_counts = defaultdict(int)
    for stem in train_stems:
        for scene in SCENE_NAMES:
            if stem.startswith(scene + "_"):
                scene_counts[scene] += 1
                break

    n_test = total - n_train
    print(f"\nFew-shot split done: train={n_train}, test={n_test}")
    for scene, cnt in sorted(scene_counts.items()):
        print(f"  {scene}: {cnt} train samples")
    print(f"Training list saved to {train_list_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split WHU-PolSAR-CD dataset for training"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sp_spatial = subparsers.add_parser(
        "spatial", help="Spatially disjoint split (3:1:1 by index order)"
    )
    sp_spatial.add_argument("--src", type=Path, required=True)
    sp_spatial.add_argument("--dst", type=Path, required=True)

    sp_fewshot = subparsers.add_parser(
        "fewshot", help="1-pct few-shot split for CPSSL"
    )
    sp_fewshot.add_argument("--src", type=Path, required=True)
    sp_fewshot.add_argument("--dst", type=Path, required=True)
    sp_fewshot.add_argument("--ratio", type=float, default=0.01)
    sp_fewshot.add_argument("--n-train", type=int, default=None,
                            help="Exact number of training samples (overrides --ratio)")
    sp_fewshot.add_argument("--seed", type=int, default=42)

    sp_all = subparsers.add_parser("all", help="Run both splits")
    sp_all.add_argument("--src", type=Path, required=True)
    sp_all.add_argument("--dst-spatial", type=Path, required=True)
    sp_all.add_argument("--dst-fewshot", type=Path, required=True)
    sp_all.add_argument("--ratio", type=float, default=0.01)
    sp_all.add_argument("--n-train", type=int, default=None,
                        help="Exact number of training samples (overrides --ratio)")
    sp_all.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "spatial":
        spatial_split(args.src, args.dst)
    elif args.command == "fewshot":
        fewshot_split(args.src, args.dst, args.ratio, args.seed, args.n_train)
    elif args.command == "all":
        print("=" * 60)
        print("Running spatially disjoint split...")
        print("=" * 60)
        spatial_split(args.src, args.dst_spatial)
        print()
        print("=" * 60)
        print("Running few-shot split...")
        print("=" * 60)
        fewshot_split(args.src, args.dst_fewshot, args.ratio, args.seed, args.n_train)


if __name__ == "__main__":
    main()
