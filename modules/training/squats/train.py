import random
import torch
import matplotlib.pyplot as plt

import numpy as np
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import modules.config as config
from modules.training.squats.dataset import SquatPTDataset
from modules.training.squats.model import TwoStreamMobileNetGRU


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
class SquatTrainer:
    def __init__(self, hp=None):
        config.ensure_dirs()

        self.hp = hp if hp is not None else config.HP
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        set_seed(config.SEED)

        self._build_data()
        self._build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.hp.learning_rate,
            weight_decay=self.hp.weight_decay
        )

    def _build_data(self):
        full_train = SquatPTDataset(config.PROCESSED_DIR, self.hp, split="train")
        g = torch.Generator().manual_seed(config.SEED)
        train_size = int(config.TRAIN_SPLIT * len(full_train))
        val_size = len(full_train) - train_size
        train_ds, _ = random_split(full_train, [train_size, val_size], generator=g)
        full_val = SquatPTDataset(config.PROCESSED_DIR, self.hp, split="val")
        _, val_ds = random_split(full_val, [train_size, val_size], generator=g)

        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.hp.batch_size,
            shuffle=True,
            num_workers=self.hp.num_workers,
            pin_memory=(self.device.type == "cuda")
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.hp.batch_size,
            shuffle=False,
            num_workers=self.hp.num_workers,
            pin_memory=(self.device.type == "cuda")
        )

    def _build_model(self):
        self.model = TwoStreamMobileNetGRU(
            num_classes=len(self.hp.class_names),
            hidden=self.hp.gru_hidden,
            layers=self.hp.gru_layers,
            dropout=self.hp.dropout,
            freeze_backbone=self.hp.freeze_backbone
        ).to(self.device)

    def _run_epoch(self, loader, train=True):
        self.model.train(train)
        total_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(loader, desc="Train" if train else "Val", leave=False)

        for batch in pbar:
            x1 = batch["original"].to(self.device)
            x2 = batch["mediapipe"].to(self.device) 
            y = batch["label"].to(self.device)

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(x1, x2)
            loss = self.criterion(logits, y)

            if train:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * y.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            pbar.set_postfix(loss=float(loss.item()), acc=(correct / max(total, 1)))

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        return avg_loss, acc

    def train(self):
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_val = 0.0
        best_path = config.MODELS_DIR / "squats" / "squats_twostream_mobilenet_GRU_model.pt"

        for epoch in range(1, self.hp.num_epochs + 1):
            tr_loss, tr_acc = self._run_epoch(self.train_loader, train=True)
            va_loss, va_acc = self._run_epoch(self.val_loader, train=False)
            train_losses.append(tr_loss)
            val_losses.append(va_loss)
            train_accs.append(tr_acc)
            val_accs.append(va_acc)

            if va_acc > best_val:
                best_val = va_acc
                torch.save({"state_dict": self.model.state_dict(), "hp": self.hp}, best_path)

        out_plot = config.MODELS_DIR / "squats" / "training_curves_squats.png"

        plt.figure()
        plt.plot(train_losses, label="train_loss")
        plt.plot(val_losses, label="val_loss")
        plt.legend()
        plt.title("Loss Curves")
        plt.savefig(out_plot, dpi=150)
        plt.close()
        out_plot2 = config.MODELS_DIR / "squats" / "training_acc_squats.png"
        plt.figure()
        plt.plot(train_accs, label="train_acc")
        plt.plot(val_accs, label="val_acc")
        plt.legend()
        plt.title("Accuracy Curves")
        plt.savefig(out_plot2, dpi=150)
        plt.close()

def main():
    trainer = SquatTrainer(hp=config.HP)
    trainer.train()

if __name__ == "__main__":
    main()