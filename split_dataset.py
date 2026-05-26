"""
Dataset splitting utility for IRSTD datasets.
Creates train/val/test split files.
"""

import os
import random
import numpy as np
from typing import List, Tuple


def split_dataset(
    data_root: str,
    img_dir: str = "IRSTD1k_Img",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    output_prefix: str = "",
):
    """
    Split dataset into train/val/test sets and write split files.
    
    Args:
        data_root: Root directory
        img_dir: Image subdirectory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
        output_prefix: Prefix for output files
    """
    random.seed(seed)
    np.random.seed(seed)

    img_path = os.path.join(data_root, img_dir)
    all_files = sorted(os.listdir(img_path))
    all_names = [os.path.splitext(f)[0] for f in all_files]

    random.shuffle(all_names)

    n_total = len(all_names)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_names = sorted(all_names[:n_train])
    val_names = sorted(all_names[n_train : n_train + n_val])
    test_names = sorted(all_names[n_train + n_val :])

    print(f"Dataset split: Train={len(train_names)}, Val={len(val_names)}, Test={len(test_names)}")

    # Write split files
    def write_list(names, filename):
        filepath = os.path.join(data_root, filename)
        with open(filepath, "w") as f:
            f.write("\n".join(names))
        print(f"  Written {filename} ({len(names)} samples)")

    write_list(train_names, f"{output_prefix}train.txt")
    write_list(val_names, f"{output_prefix}val.txt")
    write_list(test_names, f"{output_prefix}test.txt")
    write_list(train_names + val_names, f"{output_prefix}trainval.txt")
    write_list(train_names + val_names + test_names, f"{output_prefix}trainvaltest.txt")

    return train_names, val_names, test_names


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--img_dir", type=str, default="IRSTD1k_Img")
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_dataset(
        args.data_root,
        args.img_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )