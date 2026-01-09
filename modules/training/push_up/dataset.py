import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path


class PushUpPTDataset(Dataset):
    def __init__(self, processed_dir, split="train"):
        self.processed_dir = Path(processed_dir)
        import modules.config as config
        self.hp = config.PUSH_UP_HP
        self.split = split

        self.pt_dir = self.processed_dir
        self.pt_files = sorted(list(self.pt_dir.glob("*.pt")))
        if len(self.pt_files) == 0:
            raise ValueError(f"No .pt files found in: {self.pt_dir}")

        self._setup_transforms()

    def _setup_transforms(self):
        t = []

        if self.split == "train" and self.hp.use_augmentation:
            t += [
                transforms.RandomHorizontalFlip(p=self.hp.horizontal_flip_prob),
                transforms.ColorJitter(
                    brightness=self.hp.color_jitter_brightness,
                    contrast=self.hp.color_jitter_contrast
                )
            ]

        t += [
            transforms.Resize((self.hp.img_size, self.hp.img_size)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
        self.transform = transforms.Compose(t)

    def __len__(self):
        return len(self.pt_files)

    def _frames_to_tensor(self, frames_uint8_thwc: torch.Tensor):
        # uint8 (T,H,W,C) -> float (T,C,H,W) in [0,1]
        x = frames_uint8_thwc.permute(0, 3, 1, 2).float() / 255.0
        out = [self.transform(x[i]) for i in range(x.shape[0])]
        return torch.stack(out, dim=0)

    def __getitem__(self, idx):
        item = torch.load(self.pt_files[idx], map_location="cpu")
        original = item["original"]
        mediapipe = item["mediapipe"]
        label = item["label"]

        return {
            "original": self._frames_to_tensor(original),
            "mediapipe": self._frames_to_tensor(mediapipe),
            "label": label if torch.is_tensor(label) else torch.tensor(int(label), dtype=torch.long)
        }