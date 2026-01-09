import sys
import cv2
import torch

import numpy as np
import mediapipe as mp_module

from pathlib import Path
from tqdm import tqdm

import modules.config as config

mp = None
mp_pose = None
mp_draw = None

def init_mediapipe():
    global mp, mp_pose, mp_draw
    if mp is None:
        mp = mp_module
        mp_pose = mp.solutions.pose
        mp_draw = mp.solutions.drawing_utils

def process_single_video(video_path: Path, num_frames: int, img_size: int, mp_model_complexity: int = 1):
    if mp_pose is None:
        init_mediapipe()

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=int(mp_model_complexity),
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            cap.release()
            return None, None

        idxs = np.linspace(0, total - 1, num_frames).astype(int)
        original_frames, mediapipe_frames = [], []

        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                continue

            frame_bgr = cv2.resize(frame_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            original_frames.append(rgb)

            results = pose.process(rgb)
            mp_frame = rgb.copy()
            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    mp_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )
            mediapipe_frames.append(mp_frame)

        cap.release()

        if len(original_frames) == num_frames and len(mediapipe_frames) == num_frames:
            return np.asarray(original_frames, dtype=np.uint8), np.asarray(mediapipe_frames, dtype=np.uint8)
        return None, None

def preprocess_all_videos():
    config.ensure_dirs()
    hp = config.PUSH_UP_HP

    data_dir = config.PUSH_UP_DATA_DIR
    processed_dir = config.PUSH_UP_PROCESSED_DIR
    processed_dir.mkdir(exist_ok=True)

    exts = {".mp4", ".avi", ".mov", ".mkv"}
    all_videos = []
    for cls in hp.class_names:
        cls_dir = data_dir / cls
        if cls_dir.exists():
            for video_file in cls_dir.glob("*"):
                if video_file.suffix.lower() in exts:
                    all_videos.append((cls, video_file))

    failed = 0
    for cls, video_path in tqdm(all_videos, desc="Processing"):
        original, mediapipe = process_single_video(
            video_path=video_path,
            num_frames=hp.num_frames,
            img_size=hp.img_size,
            mp_model_complexity=hp.mp_model_complexity
        )

        if original is None or mediapipe is None:
            failed += 1
            continue

        label = hp.class_names.index(cls)
        out_pt = processed_dir / f"{cls}_{video_path.stem}.pt"

        torch.save(
            {
                "original": torch.from_numpy(original).contiguous(),
                "mediapipe": torch.from_numpy(mediapipe).contiguous(),
                "label": torch.tensor(label, dtype=torch.long),
                "meta": {"video_name": video_path.name}
            },
            out_pt
        )

if __name__ == "__main__":
    preprocess_all_videos()