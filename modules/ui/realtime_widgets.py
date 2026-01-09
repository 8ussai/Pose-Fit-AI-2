import sys
from pathlib import Path
import time
import cv2
import torch
import numpy as np
import mediapipe as mp
from torchvision import transforms

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFrame
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import modules.config as config
from modules.training.squats.model import TwoStreamMobileNetGRU as SquatModel
from modules.training.shoulder_press.model import TwoStreamMobileNetGRU as ShoulderModel
from modules.training.push_up.model import TwoStreamMobileNetGRU as PushUpModel

try:
    from modules.ui.yolo_detection_manager import YOLODetectionManager
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLO detection not available")


class CameraManager:
    """Manages camera capture with MediaPipe"""
    def __init__(self, camera_index=0, w=1280, h=720, fps_ms=15):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")

    def read(self):
        """Read frame and process with MediaPipe"""
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        return frame, rgb, results

    def close(self):
        if self.cap:
            self.cap.release()
        if self.pose:
            self.pose.close()


class BaseExerciseWidget(QWidget):
    def __init__(self, camera_manager, hp, model_class, model_path, parent=None):
        super().__init__(parent)
        self.camera = camera_manager
        self.hp = hp
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_manager = None
        if YOLO_AVAILABLE:
            try:
                self.yolo_manager = YOLODetectionManager()
                if not self.yolo_manager.is_available():
                    self.yolo_manager = None
                    print("⚠️  YOLO model not loaded, equipment detection disabled")
            except Exception as e:
                print(f"⚠️  Failed to initialize YOLO: {e}")
                self.yolo_manager = None

        self.model = model_class(
            num_classes=len(hp.class_names),
            hidden=hp.gru_hidden,
            layers=hp.gru_layers,
            dropout=hp.dropout,
            freeze_backbone=False
        ).to(self.device)

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            self.model.load_state_dict(ckpt["model_state"])
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            self.model.load_state_dict(ckpt["state_dict"])
        else:
            self.model.load_state_dict(ckpt)

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((hp.img_size, hp.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.state = "CALIBRATING"
        self.rep_frames = []
        self.went_down = False
        self.went_up = False
        self.last_result = None
        self.last_result_time = 0.0
        self.result_hold_sec = 3.0
        self.calib_start = time.time()
        self.calib_values = []
        self.stand_position = None
        self.UP_THRESHOLD = None
        self.DOWN_THRESHOLD = None
        self.fps = 0.0
        self.fps_start = time.time()
        self.frame_count = 0
        self.correct_reps_count = 0
        self._build_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(15)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background: #1a1a1a;
                border: 2px solid #b31b1b;
                border-radius: 12px;
            }
        """)
        self.video_label.setMinimumSize(800, 600)

        status_frame = QFrame()
        status_frame.setStyleSheet("""
            QFrame {
                background: rgba(0, 0, 0, 0.7);
                border: 1px solid #b31b1b;
                border-radius: 8px;
                padding: 10px;
            }
        """)

        status_layout = QHBoxLayout(status_frame)

        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("color: #F8F9FA; font-size: 16px; font-weight: bold;")
        self.reps_counter_label = QLabel("✅ Correct: 0")
        self.reps_counter_label.setStyleSheet("""
            color: #00ff00; 
            font-size: 18px; 
            font-weight: bold;
            background: rgba(0, 255, 0, 0.1);
            padding: 5px 15px;
            border-radius: 5px;
            border: 2px solid #00ff00;
        """)

        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("color: #00ff00; font-size: 14px;")
        self.yolo_label = QLabel("Equipment: --")
        self.yolo_label.setStyleSheet("color: #FFA500; font-size: 14px; font-weight: bold;")
        self.recalib_btn = QPushButton("🔄 Recalibrate")
        self.recalib_btn.setCursor(Qt.PointingHandCursor)
        self.recalib_btn.setStyleSheet("""
            QPushButton {
                background: #8F1414;
                color: #F8F9FA;
                border: 1px solid #6F0F0F;
                padding: 8px 16px;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #a01818;
                border: 2px solid #c92222;
            }
        """)
        self.recalib_btn.clicked.connect(self._recalibrate)
        self.reset_counter_btn = QPushButton("🔢 Reset Counter")
        self.reset_counter_btn.setCursor(Qt.PointingHandCursor)
        self.reset_counter_btn.setStyleSheet("""
            QPushButton {
                background: #1a5f1a;
                color: #F8F9FA;
                border: 1px solid #0f3f0f;
                padding: 8px 16px;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #2a7f2a;
                border: 2px solid #3a9f3a;
            }
        """)
        self.reset_counter_btn.clicked.connect(self._reset_counter)

        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.reps_counter_label)
        status_layout.addWidget(self.yolo_label)
        status_layout.addWidget(self.fps_label)
        status_layout.addWidget(self.reset_counter_btn)
        status_layout.addWidget(self.recalib_btn)
        layout.addWidget(self.video_label, 1)
        layout.addWidget(status_frame)

    def _recalibrate(self):
        self.state = "CALIBRATING"
        self.calib_start = time.time()
        self.calib_values = []
        self.stand_position = None
        self.UP_THRESHOLD = None
        self.DOWN_THRESHOLD = None
        self.rep_frames = []
        self.went_down = False
        self.went_up = False
        self.last_result = None

    def _reset_counter(self):
        self.correct_reps_count = 0
        self.reps_counter_label.setText(f"✅ Correct: {self.correct_reps_count}")

    def _display_frame(self, frame, status_text):
        h, w = frame.shape[:2]

        if self.yolo_manager and self.frame_count % 10 == 0:
            try:
                self.yolo_manager.detect(frame)
                yolo_text = self.yolo_manager.get_detection_text()
                self.yolo_label.setText(yolo_text)
            except Exception:
                self.yolo_label.setText("Detection error")

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            now = time.time()
            self.fps = 30.0 / (now - self.fps_start)
            self.fps_start = now
            self.fps_label.setText(f"FPS: {self.fps:.1f}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)
        self.status_label.setText(status_text)

    def _sample_uniform(self, frames, n=24):
        if len(frames) == 0:
            return []
        if len(frames) == n:
            return frames
        idx = np.linspace(0, len(frames) - 1, n).astype(int)
        return [frames[i] for i in idx]

    def _process_rep(self):
        sampled = self._sample_uniform(self.rep_frames, self.hp.num_frames)

        if len(sampled) != self.hp.num_frames:
            self.last_result = {"label": "Incomplete", "conf": 0.0}
            self.last_result_time = time.time()
            return

        original_stream = []
        mediapipe_stream = []

        for fr in sampled:
            original_stream.append(self.transform(fr))

            results = self.camera.pose.process(fr)
            drawn = fr.copy()
            if results.pose_landmarks:
                self.camera.mp_draw.draw_landmarks(
                    drawn, results.pose_landmarks,
                    self.camera.mp_pose.POSE_CONNECTIONS
                )
            mediapipe_stream.append(self.transform(drawn))

        orig_t = torch.stack(original_stream).unsqueeze(0).to(self.device)
        mp_t = torch.stack(mediapipe_stream).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(orig_t, mp_t)
            probs = torch.softmax(out, dim=1)
            conf, pred = torch.max(probs, 1)

        label_idx = pred.item()
        label = self.hp.class_names[label_idx].title()
        confv = float(conf.item() * 100.0)

        self.last_result = {"label": label, "conf": confv}
        self.last_result_time = time.time()

        if label.lower() == "correct":
            self.correct_reps_count += 1
            self.reps_counter_label.setText(f"✅ Correct: {self.correct_reps_count}")

class SquatWidget(BaseExerciseWidget):
    def __init__(self, camera_manager, parent=None):
        hp = config.HP
        model_path = config.MODELS_DIR / "squats" / "squats_twostream_mobilenet_GRU_model.pt"
        super().__init__(camera_manager, hp, SquatModel, model_path, parent)

    def _update_frame(self):
        frame, rgb, results = self.camera.read()
        if frame is None:
            return

        now = time.time()

        hip_y = None
        if results and results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            lh = lm[self.camera.mp_pose.PoseLandmark.LEFT_HIP]
            rh = lm[self.camera.mp_pose.PoseLandmark.RIGHT_HIP]
            if lh.visibility > 0.5 and rh.visibility > 0.5:
                hip_y = float((lh.y + rh.y) / 2.0)

        if self.state == "RESULT" and self.last_result is not None:
            label = self.last_result["label"]
            conf = self.last_result["conf"]

            colors = {
                "correct": (0, 255, 0),
                "fast": (0, 165, 255),
                "incomplete": (0, 255, 255),
                "wrong_position": (0, 0, 255)
            }
            color = colors.get(label.lower(), (255, 255, 255))

            cv2.rectangle(frame, (20, 20), (500, 100), (0, 0, 0), -1)
            cv2.rectangle(frame, (20, 20), (500, 100), color, 2)
            cv2.putText(frame, f"Result: {label}", (35, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Confidence: {conf:.1f}%", (35, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            self._display_frame(frame, f"✅ {label} - {conf:.1f}%")

            if (now - self.last_result_time) >= self.result_hold_sec:
                self.last_result = None
                self.state = "READY"
                self.rep_frames.clear()
                self.went_down = False
                self.went_up = False
            return

        status_text = ""

        if self.state == "CALIBRATING":
            if hip_y is not None:
                self.calib_values.append(hip_y)

            elapsed = now - self.calib_start
            remaining = max(0, self.hp.calib_seconds - elapsed)
            status_text = f"⏱️ Calibrating... {remaining:.1f}s - Stand still"

            if elapsed >= self.hp.calib_seconds and len(self.calib_values) >= 10:
                self.stand_position = float(np.median(self.calib_values))
                self.UP_THRESHOLD = self.stand_position + self.hp.up_offset
                self.DOWN_THRESHOLD = self.stand_position + self.hp.down_offset
                self.state = "READY"
                status_text = "✅ Ready! Do squats continuously"

        elif self.state in ["READY", "RECORDING"]:
            if hip_y is not None and self.UP_THRESHOLD is not None:
                if hip_y > self.DOWN_THRESHOLD:
                    if not self.went_down:
                        self.state = "RECORDING"
                    self.went_down = True

                if self.state == "RECORDING":
                    self.rep_frames.append(rgb) 
                    status_text = f"🔴 Recording... ({len(self.rep_frames)} frames)"

                if self.went_down and hip_y < self.UP_THRESHOLD:
                    self.went_up = True

                if self.went_down and self.went_up:
                    self.state = "PROCESSING"
                    self._process_rep()
                    self.state = "RESULT"

                if len(self.rep_frames) > self.hp.num_frames * 3:
                    self.state = "PROCESSING"
                    self._process_rep()
                    self.state = "RESULT"
            else:
                status_text = "⚠️ No pose detected"

        if not status_text:
            status_text = "Ready: Do squats"

        self._display_frame(frame, status_text)


class ShoulderPressWidget(BaseExerciseWidget):
    def __init__(self, camera_manager, parent=None):
        hp = config.SHOULDER_PRESS_HP
        model_path = config.MODELS_DIR / "shoulder_press" / "shoulder_press_twostream_mobilenet_GRU_model.pt"
        super().__init__(camera_manager, hp, ShoulderModel, model_path, parent)

    def _update_frame(self):
        frame, rgb, results = self.camera.read()
        if frame is None:
            return

        now = time.time()

        wrist_y = None
        if results and results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            lw = lm[self.camera.mp_pose.PoseLandmark.LEFT_WRIST]
            rw = lm[self.camera.mp_pose.PoseLandmark.RIGHT_WRIST]
            if lw.visibility > 0.5 and rw.visibility > 0.5:
                wrist_y = float((lw.y + rw.y) / 2.0)

        if self.state == "RESULT" and self.last_result is not None:
            label = self.last_result["label"]
            conf = self.last_result["conf"]

            colors = {
                "correct": (0, 255, 0),
                "fast": (0, 165, 255),
                "incomplete": (0, 255, 255),
                "wrong_position": (0, 0, 255)
            }
            color = colors.get(label.lower(), (255, 255, 255))

            cv2.rectangle(frame, (20, 20), (500, 100), (0, 0, 0), -1)
            cv2.rectangle(frame, (20, 20), (500, 100), color, 2)
            cv2.putText(frame, f"Result: {label}", (35, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Confidence: {conf:.1f}%", (35, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            self._display_frame(frame, f"✅ {label} - {conf:.1f}%")

            if (now - self.last_result_time) >= self.result_hold_sec:
                self.last_result = None
                self.state = "READY"
                self.rep_frames.clear()
                self.went_down = False
                self.went_up = False
            return

        status_text = ""

        if self.state == "CALIBRATING":
            if wrist_y is not None:
                self.calib_values.append(wrist_y)

            elapsed = now - self.calib_start
            remaining = max(0, self.hp.calib_seconds - elapsed)
            status_text = f"⏱️ Calibrating... {remaining:.1f}s - Hold at shoulder"

            if elapsed >= self.hp.calib_seconds and len(self.calib_values) >= 10:
                self.stand_position = float(np.median(self.calib_values))
                self.UP_THRESHOLD = self.stand_position - self.hp.up_offset
                self.DOWN_THRESHOLD = self.stand_position - 0.03
                self.state = "READY"
                status_text = "✅ Ready! Press continuously"

        elif self.state in ["READY", "RECORDING"]:
            if wrist_y is not None and self.UP_THRESHOLD is not None:
                if wrist_y < self.UP_THRESHOLD:
                    if not self.went_up:
                        self.state = "RECORDING"
                    self.went_up = True

                if self.state == "RECORDING":
                    self.rep_frames.append(rgb)
                    status_text = f"🔴 Recording... ({len(self.rep_frames)} frames)"

                if self.went_up and wrist_y > self.DOWN_THRESHOLD:
                    self.went_down = True

                if self.went_up and self.went_down:
                    self.state = "PROCESSING"
                    self._process_rep()
                    self.state = "RESULT"

                if len(self.rep_frames) > self.hp.num_frames * 3:
                    self.state = "PROCESSING"
                    self._process_rep()
                    self.state = "RESULT"
            else:
                status_text = "⚠️ No pose detected"

        if not status_text:
            status_text = "Ready: Press overhead"

        self._display_frame(frame, status_text)


class PushUpWidget(BaseExerciseWidget):
    def __init__(self, camera_manager, parent=None):
        hp = config.PUSH_UP_HP
        model_path = config.MODELS_DIR / "push_up" / "push_up_twostream_mobilenet_GRU_model.pt"
        super().__init__(camera_manager, hp, PushUpModel, model_path, parent)

    def _update_frame(self):
        frame, rgb, results = self.camera.read()
        if frame is None:
            return

        now = time.time()

        shoulder_y = None
        if results and results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            ls = lm[self.camera.mp_pose.PoseLandmark.LEFT_SHOULDER]
            rs = lm[self.camera.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            if ls.visibility > 0.5 and rs.visibility > 0.5:
                shoulder_y = float((ls.y + rs.y) / 2.0)

        if self.state == "RESULT" and self.last_result is not None:
            label = self.last_result["label"]
            conf = self.last_result["conf"]

            colors = {
                "correct": (0, 255, 0),
                "bad_body_alignment": (0, 0, 255),
                "incomplete": (0, 255, 255)
            }
            color = colors.get(label.lower(), (255, 255, 255))

            cv2.rectangle(frame, (20, 20), (500, 100), (0, 0, 0), -1)
            cv2.rectangle(frame, (20, 20), (500, 100), color, 2)
            cv2.putText(frame, f"Result: {label}", (35, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Confidence: {conf:.1f}%", (35, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            self._display_frame(frame, f"✅ {label} - {conf:.1f}%")

            if (now - self.last_result_time) >= self.result_hold_sec:
                self.last_result = None
                self.state = "READY"
                self.rep_frames.clear()
                self.went_down = False
                self.went_up = False
            return

        status_text = ""

        if self.state == "CALIBRATING":
            if shoulder_y is not None:
                self.calib_values.append(shoulder_y)

            elapsed = now - self.calib_start
            remaining = max(0, self.hp.calib_seconds - elapsed)
            status_text = f"⏱️ Calibrating... {remaining:.1f}s - Plank position"

            if elapsed >= self.hp.calib_seconds and len(self.calib_values) >= 10:
                self.stand_position = float(np.median(self.calib_values))
                self.UP_THRESHOLD = self.stand_position + 0.03
                self.DOWN_THRESHOLD = self.stand_position + self.hp.down_offset
                self.state = "READY"
                status_text = "✅ Ready! Do push-ups continuously"

        elif self.state in ["READY", "RECORDING"]:
            if shoulder_y is not None and self.UP_THRESHOLD is not None:
                if shoulder_y > self.DOWN_THRESHOLD:
                    if not self.went_down:
                        self.state = "RECORDING"
                    self.went_down = True

                if self.state == "RECORDING":
                    self.rep_frames.append(rgb)
                    status_text = f"🔴 Recording... ({len(self.rep_frames)} frames)"

                if self.went_down and shoulder_y < self.UP_THRESHOLD:
                    self.went_up = True

                if self.went_down and self.went_up:
                    self.state = "PROCESSING"
                    self._process_rep()
                    self.state = "RESULT"

                if len(self.rep_frames) > self.hp.num_frames * 3:
                    self.state = "PROCESSING"
                    self._process_rep()
                    self.state = "RESULT"
            else:
                status_text = "⚠️ No pose detected"

        if not status_text:
            status_text = "Ready: Do push-ups"

        self._display_frame(frame, status_text)