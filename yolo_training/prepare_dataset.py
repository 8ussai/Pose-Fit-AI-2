import argparse
import random
from pathlib import Path
import shutil
import sys

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LBL_EXT = ".txt"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Split images+labels into dataset/{images,labels}/{train,val,test} for YOLO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src-images", type=str, default="dataset_raw/images",
                        help="Source folder containing images")
    parser.add_argument("--src-labels", type=str, default="dataset_raw/labels",
                        help="Source folder containing YOLO label .txt files")
    parser.add_argument("--out-dir", type=str, default="dataset",
                        help="Output dataset root (will create images/ and labels/ subfolders)")
    parser.add_argument("--train", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val",   type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test",  type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed",  type=int, default=42, help="Random seed for shuffling")
    act = parser.add_mutually_exclusive_group()
    act.add_argument("--copy", action="store_true", help="Copy files instead of moving")
    act.add_argument("--move", action="store_true", help="Move files instead of copying")
    parser.add_argument("--strict", action="store_true",
                        help="Strict mode: fail if any image is missing its label, or vice versa")
    return parser.parse_args()

def list_images(src_images: Path):
    imgs = []
    for p in src_images.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
    return imgs

def pair_image_label(img_path: Path, src_labels: Path) -> Path | None:
    lbl = src_labels / (img_path.stem + LBL_EXT)
    return lbl if lbl.exists() else None

def ensure_dirs(root: Path):
    for sub in ["images/train","images/val","images/test","labels/train","labels/val","labels/test"]:
        (root / sub).mkdir(parents=True, exist_ok=True)

def copy_or_move(src: Path, dst: Path, do_move: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if do_move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))

def main():
    args = parse_args()
    random.seed(args.seed)

    src_images = Path(args.src_images).resolve()
    src_labels = Path(args.src_labels).resolve()
    out_dir    = Path(args.out_dir).resolve()

    if not src_images.exists() or not src_images.is_dir():
        print(f"[ERROR] Images folder not found: {src_images}")
        sys.exit(1)
    if not src_labels.exists() or not src_labels.is_dir():
        print(f"[ERROR] Labels folder not found: {src_labels}")
        sys.exit(1)

    total_ratio = args.train + args.val + args.test
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"[ERROR] Split ratios must sum to 1.0 (got {total_ratio})")
        sys.exit(1)

    ensure_dirs(out_dir)

    images = list_images(src_images)
    print(f"[INFO] Found {len(images)} image(s) under: {src_images}")

    pairs = []
    missing_labels = []
    for img in images:
        lbl = pair_image_label(img, src_labels)
        if lbl is None:
            missing_labels.append(img)
            if args.strict:
                print(f"[STRICT] Missing label for image: {img}")
        else:
            pairs.append((img, lbl))

    orphan_labels = []
    for lbl in src_labels.rglob(f"*{LBL_EXT}"):
        img_candidate = None
        for ext in IMG_EXTS:
            cand = src_images / (lbl.stem + ext)
            if cand.exists():
                img_candidate = cand
                break
        if img_candidate is None:
            orphan_labels.append(lbl)

    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * args.train)
    n_val   = int(n * args.val)

    splits = {
        "train": pairs[:n_train],
        "val":   pairs[n_train:n_train+n_val],
        "test":  pairs[n_train+n_val:]
    }

    do_move = args.move and not args.copy
    for split_name, items in splits.items():
        for img, lbl in items:
            dst_img = out_dir / "images" / split_name / img.name
            dst_lbl = out_dir / "labels" / split_name / lbl.name
            copy_or_move(img, dst_img, do_move)
            copy_or_move(lbl, dst_lbl, do_move)

if __name__ == "__main__":
    main()