import torch

from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
class SquatPTDataset(Dataset):
    def __init__(self, processed_dir, hp, split="train"):
        self.processed_dir = Path(processed_dir)
        self.hp = hp
        self.split = split
        self.pt_files = sorted(list(self.processed_dir.glob("*.pt")))
        self._setup_transforms()

    def _setup_transforms(self):
        t = [transforms.ToPILImage()]

        if self.split == "train" and self.hp.use_augmentation:
            t += [
                transforms.RandomHorizontalFlip(p=self.hp.horizontal_flip_prob),
                transforms.ColorJitter(
                    brightness=self.hp.color_jitter_brightness,
                    contrast=self.hp.color_jitter_contrast
                )
            ]

        t += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
        self.transform = transforms.Compose(t)

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        pt_path = self.pt_files[idx]
        data = torch.load(pt_path, map_location="cpu")
        original_frames = data["original"]     
        mediapipe_frames = data["mediapipe"]    
        label = int(data["label"])

        H = int(original_frames.shape[1])
        W = int(original_frames.shape[2])

        if H != self.hp.img_size or W != self.hp.img_size:
            raise ValueError(
                f"Stored frames are {H}x{W} but hp.img_size={self.hp.img_size}. "
                f"Re-run preprocess_videos.py to save resized frames."
            )

        original_np = original_frames.numpy()
        mediapipe_np = mediapipe_frames.numpy()
        orig_stack, mp_stack = [], []

        for i in range(original_np.shape[0]):
            orig_stack.append(self.transform(original_np[i]))
            mp_stack.append(self.transform(mediapipe_np[i]))

        return {
            "original": torch.stack(orig_stack),
            "mediapipe": torch.stack(mp_stack),
            "label": torch.tensor(label, dtype=torch.long)
        }