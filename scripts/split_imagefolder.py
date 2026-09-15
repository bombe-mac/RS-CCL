"""Create train/test splits for ImageFolder-style datasets.

Example:
    python scripts/split_imagefolder.py ^
        --source data/UCMerced_LandUse/Images ^
        --output data/UCML ^
        --train-ratio 0.8
"""
#python scripts\split_imagefolder.py --source data\UCMerced_LandUse\Images --output data\UCML --train-ratio 0.8 --overwrite
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split class folders into train and test folders."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/UCMerced_LandUse/Images"),
        help="Directory containing one subdirectory per class.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/UCML"),
        help="Output directory where train/ and test/ will be created.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of each class to place in train.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=5959,
        help="Random seed used for reproducible splits.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output directory before writing.",
    )
    return parser.parse_args()


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_class_files(files: list[Path], train_ratio: float) -> tuple[list[Path], list[Path]]:
    shuffled = files[:]
    random.shuffle(shuffled)
    train_count = int(len(shuffled) * train_ratio)
    return shuffled[:train_count], shuffled[train_count:]


def copy_or_move(files: list[Path], destination: Path, move: bool) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    operation = shutil.move if move else shutil.copy2

    for source_file in files:
        operation(str(source_file), str(destination / source_file.name))


def main() -> None:
    args = parse_args()

    if not args.source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {args.source}")

    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1.")

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {args.output}. Use --overwrite to recreate it."
            )
        shutil.rmtree(args.output)

    random.seed(args.seed)
    class_dirs = sorted(path for path in args.source.iterdir() if path.is_dir())

    total_train = 0
    total_test = 0

    for class_dir in class_dirs:
        files = list_images(class_dir)
        if not files:
            print(f"Skipping {class_dir.name}: no image files found.")
            continue

        train_files, test_files = split_class_files(files, args.train_ratio)
        copy_or_move(train_files, args.output / "train" / class_dir.name, args.move)
        copy_or_move(test_files, args.output / "test" / class_dir.name, args.move)

        total_train += len(train_files)
        total_test += len(test_files)
        print(f"{class_dir.name}: train={len(train_files)} test={len(test_files)}")

    print(f"Done. train={total_train} test={total_test} output={args.output}")


if __name__ == "__main__":
    main()
