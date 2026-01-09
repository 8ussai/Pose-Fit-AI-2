import torch

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import modules.config as config
from modules.training.squats.dataset import SquatPTDataset
from modules.training.squats.model import TwoStreamMobileNetGRU


@torch.no_grad()
def evaluate(model_path=None):
    config.ensure_dirs()

    hp = config.HP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_dir = config.EVAL_DIR / "squats"
    eval_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_path or (config.MODELS_DIR / "squats" / "squats_twostream_mobilenet_GRU_model.pt")
    ckpt = torch.load(model_path, map_location="cpu")

    model = TwoStreamMobileNetGRU(
        num_classes=len(hp.class_names),
        hidden=hp.gru_hidden,
        layers=hp.gru_layers,
        dropout=hp.dropout,
        freeze_backbone=False
    ).to(device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    ds = SquatPTDataset(config.PROCESSED_DIR, hp, split="val")

    loader = DataLoader(
        ds,
        batch_size=hp.batch_size,
        shuffle=False,
        num_workers=hp.num_workers
    )

    all_preds, all_labels = [], []

    for batch in loader:
        x1 = batch["original"].to(device)
        x2 = batch["mediapipe"].to(device)
        y = batch["label"].cpu().numpy()
        logits = model(x1, x2)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    report = classification_report(all_labels, all_preds, target_names=list(hp.class_names))
    print(report)

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=hp.class_names
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Confusion Matrix")
    cm_path = eval_dir / "confusion_matrix.png"

    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()

    class_acc = cm.diagonal() / cm.sum(axis=1)

    plt.figure(figsize=(6, 4))
    plt.bar(hp.class_names, class_acc)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    plt.xticks(rotation=30)

    acc_path = eval_dir / "per_class_accuracy.png"

    plt.savefig(acc_path, dpi=150, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    evaluate()