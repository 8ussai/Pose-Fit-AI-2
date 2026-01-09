import argparse
import shutil
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO and export best.pt to new_training/runs/best.pt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", type=str, default="new_training/data.yaml",
        help="Path to data.yaml describing train/val sets and nc/names"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8s.pt",
        help="Base model weights (.pt) to start from"
    )
    parser.add_argument(
        "--task", type=str, choices=["detect", "segment"], default="detect",
        help="YOLO task type (detection or segmentation)"
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="Number of training epochs"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Training image size"
    )
    parser.add_argument(
        "--batch", type=int, default=-1, help="Auto-batch if -1, else set a value"
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="CUDA device id (e.g. '0') or 'cpu'"
    )
    parser.add_argument(
        "--project", type=str, default="new_training/runs/train",
        help="Project directory for YOLO runs (exp, exp2, ... will be created here)"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Run name (default: auto 'exp', 'exp2', ...)"
    )
    parser.add_argument(
        "--exist-ok", action="store_true",
        help="Allow existing project/name without incrementing"
    )
    return parser.parse_args()


def latest_exp_dir(project_dir: Path) -> Path | None:
    if not project_dir.exists():
        return None
    exps = [p for p in project_dir.iterdir() if p.is_dir() and p.name.startswith("exp")]
    if not exps:
        return None
    exps.sort(key=lambda p: p.stat().st_mtime)
    return exps[-1]


def main():
    args = parse_args()

    base_dir = Path(__file__).resolve().parent  # .../new_training
    project_dir = (base_dir / args.project).resolve() if not str(args.project).startswith(str(base_dir)) else Path(args.project)
    project_dir.mkdir(parents=True, exist_ok=True)

    data_path = (base_dir / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)
    model = YOLO(args.model)

    overrides = dict(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project_dir),
        name=args.name,
        exist_ok=args.exist_ok,
    )

    model.train(**overrides)

    run_dir = latest_exp_dir(project_dir)

    best_src = run_dir / "weights" / "best.pt"
    stable_runs_dir = base_dir / "runs"
    stable_runs_dir.mkdir(parents=True, exist_ok=True)
    best_dst = stable_runs_dir / "best.pt"
    shutil.copy2(best_src, best_dst)

if __name__ == "__main__":
    main()