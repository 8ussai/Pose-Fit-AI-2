import random
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import modules.config as config
from modules.training.shoulder_press.dataset import ShoulderPressPTDataset
from modules.training.shoulder_press.model import TwoStreamMobileNetGRU

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
class Trainer:
    def __init__(self):
        config.ensure_dirs()

        self.hp = config.SHOULDER_PRESS_HP

        set_seed(config.SEED)

        torch.backends.cudnn.benchmark = True

        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processed_dir = config.SHOULDER_PRESS_PROCESSED_DIR
        self.models_dir = config.MODELS_DIR / "shoulder_press"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        full_train = ShoulderPressPTDataset(self.processed_dir, split="train")
        g = torch.Generator().manual_seed(config.SEED)
        train_size = int(config.TRAIN_SPLIT * len(full_train))
        val_size = len(full_train) - train_size
        train_ds, val_subset = random_split(full_train, [train_size, val_size], generator=g)
        full_val = ShoulderPressPTDataset(self.processed_dir, split="val")
        val_ds = torch.utils.data.Subset(full_val, val_subset.indices)
        nw = int(self.hp.num_workers)
        pin = (self.device.type == "cuda")

        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.hp.batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=pin,
            persistent_workers=(nw > 0),
            prefetch_factor=2 if nw > 0 else None,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.hp.batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=pin,
            persistent_workers=(nw > 0),
            prefetch_factor=2 if nw > 0 else None,
        )

        self.model = TwoStreamMobileNetGRU(
            num_classes=len(self.hp.class_names),
            hidden=self.hp.gru_hidden,
            layers=self.hp.gru_layers,
            dropout=self.hp.dropout,
            freeze_backbone=self.hp.freeze_backbone
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hp.learning_rate,
            weight_decay=self.hp.weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )

        self.use_amp = (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.best_val_acc = 0.0
        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []

    def _accuracy(self, logits, y):
        pred = logits.argmax(dim=1)
        return (pred == y).float().mean().item()

    def train_one_epoch(self):
        self.model.train()

        total_loss, total_acc, n = 0.0, 0.0, 0
        pbar = tqdm(self.train_loader, desc="Training")

        for batch in pbar:
            x1 = batch["original"].to(self.device, non_blocking=True)
            x2 = batch["mediapipe"].to(self.device, non_blocking=True)
            y = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(x1, x2)
                    loss = self.criterion(logits, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(x1, x2)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

            acc = self._accuracy(logits, y)
            bs = y.size(0)
            total_loss += loss.item() * bs
            total_acc += acc * bs
            n += bs

            pbar.set_postfix(loss=total_loss / n, acc=total_acc / n)

        return total_loss / n, total_acc / n

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss, total_acc, n = 0.0, 0.0, 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            x1 = batch["original"].to(self.device, non_blocking=True)
            x2 = batch["mediapipe"].to(self.device, non_blocking=True)
            y = batch["label"].to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(x1, x2)
                    loss = self.criterion(logits, y)
            else:
                logits = self.model(x1, x2)
                loss = self.criterion(logits, y)

            acc = self._accuracy(logits, y)
            bs = y.size(0)
            total_loss += loss.item() * bs
            total_acc += acc * bs
            n += bs

        return total_loss / n, total_acc / n

    def plot_curves(self):
        out_dir = config.MODELS_DIR / "shoulder_press"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / "training_curves.png"

        plt.figure()
        plt.plot(self.train_losses, label="train_loss")
        plt.plot(self.val_losses, label="val_loss")
        plt.legend()
        plt.title("Shoulder Press - Loss Curves")
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()
        out_png2 = out_dir / "accuracy_curves.png"
        plt.figure()
        plt.plot(self.train_accs, label="train_acc")
        plt.plot(self.val_accs, label="val_acc")
        plt.legend()
        plt.title("Shoulder Press - Accuracy Curves")
        plt.savefig(out_png2, bbox_inches="tight")
        plt.close()

    def fit(self):
        best_path = self.models_dir / "shoulder_press_twostream_mobilenet_GRU_model.pt"

        for epoch in range(1, self.hp.num_epochs + 1):
            tr_loss, tr_acc = self.train_one_epoch()
            va_loss, va_acc = self.validate()

            self.scheduler.step(va_loss)
            self.train_losses.append(tr_loss)
            self.val_losses.append(va_loss)
            self.train_accs.append(tr_acc)
            self.val_accs.append(va_acc)

            if va_acc > self.best_val_acc:
                self.best_val_acc = va_acc
                torch.save(
                    {
                        "model_state": self.model.state_dict(),
                        "hp": self.hp.__dict__,
                        "class_names": self.hp.class_names,
                    },
                    best_path
                )

        self.plot_curves()


if __name__ == "__main__":
    Trainer().fit()