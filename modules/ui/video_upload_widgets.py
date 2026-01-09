import sys
from pathlib import Path
from collections import deque
import queue

import cv2
import torch
import numpy as np
import mediapipe as mp
from torchvision import transforms

from ultralytics import YOLO

from PySide6.QtCore import Qt, Signal, QThread, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QFrame, QFileDialog, QProgressBar
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import modules.config as config
from modules.training.squats.model import TwoStreamMobileNetGRU as SquatModel
from modules.training.shoulder_press.model import TwoStreamMobileNetGRU as ShoulderModel
from modules.training.push_up.model import TwoStreamMobileNetGRU as PushUpModel

class YOLOPresenceWorker(QThread):
    presence_ready = Signal(bool, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._q = queue.Queue(maxsize=1)  
        self._running = True

        self._yolo = None
        self._ready = False

        try:
            self._yolo = YOLO(str(config.YOLO_WEIGHTS))
            self._ready = True
        except Exception:
            self._ready = False

    @Slot(object)
    def enqueue_frame(self, frame_bgr):
        if not self._ready:
            return
        try:
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except Exception:
                    break
            self._q.put_nowait(frame_bgr)
        except Exception:
            pass

    def stop(self):
        self._running = False
        try:
            self.requestInterruption()
        except Exception:
            pass
        try:
            self._q.put_nowait(None)
        except Exception:
            pass

    def run(self):
        while self._running and not self.isInterruptionRequested():
            try:
                frame = self._q.get(timeout=0.2)
            except queue.Empty:
                continue

            if frame is None:
                break

            if not self._ready or self._yolo is None:
                self.presence_ready.emit(False, False)
                continue

            barbell = False
            dumbbell = False

            try:
                results = self._yolo.predict(
                    frame,
                    conf=0.4,
                    iou=0.5,
                    verbose=False
                )

                if results and results[0].boxes is not None:
                    for c in results[0].boxes.cls:
                        cid = int(c)
                        if cid == config.YOLO_BARBELL_CLASS_ID:
                            barbell = True
                        elif cid == config.YOLO_DUMBELL_CLASS_ID:
                            dumbbell = True

                self.presence_ready.emit(barbell, dumbbell)

            except Exception:
                self.presence_ready.emit(False, False)


class InferenceWorker(QThread):
    result_ready = Signal(int, dict)  
    error_ready = Signal(int, str)

    def __init__(self, model, hp, transform, device, parent=None):
        super().__init__(parent)
        self.model = model
        self.hp = hp
        self.transform = transform
        self.device = device
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self._q = queue.Queue(maxsize=8)
        self._running = True

    @Slot(int, object, object)
    def enqueue_rep(self, rep_number, frames_rgb, landmarks_list):
        try:
            self._q.put_nowait((rep_number, frames_rgb, landmarks_list))
        except queue.Full:
            self.error_ready.emit(rep_number, "Queue full (dropped rep)")

    def stop(self):
        self._running = False
        try:
            self.requestInterruption()
        except Exception:
            pass

        try:
            self._q.put_nowait((-1, None, None))
        except Exception:
            pass

    def run(self):
        while self._running and not self.isInterruptionRequested():
            try:
                rep_number, frames_rgb, landmarks_list = self._q.get(timeout=0.2)
            except queue.Empty:
                continue

            if rep_number == -1:
                break

            try:
                result = self._analyze_rep(frames_rgb, landmarks_list)
                self.result_ready.emit(rep_number, result)
            except Exception as e:
                self.error_ready.emit(rep_number, str(e))

    def _sample_uniform(self, arr, n):
        if not arr:
            return []
        if len(arr) == n:
            return arr
        idx = np.linspace(0, len(arr) - 1, n).astype(int)
        return [arr[i] for i in idx]

    def _draw_mp_frame(self, rgb, pose_landmarks):
        drawn = rgb.copy()
        if pose_landmarks is not None:
            self.mp_draw.draw_landmarks(
                drawn, pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
        return drawn

    def _analyze_rep(self, frames_rgb, landmarks_list):
        frames = self._sample_uniform(frames_rgb, self.hp.num_frames)
        lms = self._sample_uniform(landmarks_list, self.hp.num_frames)

        if len(frames) != self.hp.num_frames:
            return {"label": "Incomplete", "conf": 0.0}

        orig_stream = []
        mp_stream = []

        for fr, lm in zip(frames, lms):
            orig_stream.append(self.transform(fr))
            mp_drawn = self._draw_mp_frame(fr, lm)
            mp_stream.append(self.transform(mp_drawn))

        orig_t = torch.stack(orig_stream).unsqueeze(0).to(self.device)
        mp_t = torch.stack(mp_stream).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(orig_t, mp_t)
            probs = torch.softmax(out, dim=1)
            conf, pred = torch.max(probs, 1)

        label_idx = pred.item()
        label = self.hp.class_names[label_idx].title()
        confv = float(conf.item() * 100.0)
        return {"label": label, "conf": confv}

class VideoProcessor(QThread):

    frame_ready = Signal(object, str)
    rep_detected = Signal(int)           
    rep_to_infer = Signal(int, object, object) 
    progress_update = Signal(int)
    finished = Signal()
    yolo_frame = Signal(object)

    def __init__(self, video_path, hp, exercise_type, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.hp = hp
        self.exercise_type = exercise_type
        self.running = True
        self.mp_pose = mp.solutions.pose
        self.rep_count = 0
        self.current_frames = []
        self.current_landmarks = []
        self.went_down = False
        self.pos_hist = deque(maxlen=60)
        self.th_up = None
        self.th_down = None
        self.min_len = max(8, self.hp.num_frames // 2)
        self.max_len = self.hp.num_frames * 7  # حماية
        self.last_text = "Processing video..."
        self.in_rep = False
        self._warmup = 0
        self._frame_i = 0
        self._yolo_every = 15   
        self._yolo_downscale = 0.6

    def stop(self):
        self.running = False
        try:
            self.requestInterruption()
        except Exception:
            pass

    def set_last_text(self, txt):
        self.last_text = txt
        self.in_rep = False

    def run(self):
        cap = None
        try:
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                self.frame_ready.emit(None, "❌ Cannot open video")
                self.finished.emit()
                return

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            idx = 0

            self.frame_ready.emit(None, self.last_text)

            with self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as pose:

                while self.running and not self.isInterruptionRequested() and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    idx += 1

                    self.progress_update.emit(int(idx / total * 100))
                    self._frame_i += 1

                    if self._frame_i % self._yolo_every == 0:
                        if self._yolo_downscale and self._yolo_downscale < 1.0:
                            frame_small = cv2.resize(
                                frame, (0, 0),
                                fx=self._yolo_downscale,
                                fy=self._yolo_downscale,
                                interpolation=cv2.INTER_LINEAR
                            )
                        else:
                            frame_small = frame
                        self.yolo_frame.emit(frame_small)

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = pose.process(rgb)

                    pos = self._get_pos(res.pose_landmarks)
                    if pos is None:
                        self._emit_frame(frame, idx)
                        continue

                    self._update_thresholds(pos)

                    if self.th_up is None or self.th_down is None:
                        self._emit_frame(frame, idx)
                        continue

                    self._warmup += 1
                    if self._warmup < 10:
                        self._emit_frame(frame, idx)
                        continue

                    self.in_rep = True
                    self.current_frames.append(rgb)
                    self.current_landmarks.append(res.pose_landmarks)

                    if len(self.current_frames) > self.max_len:
                        self.current_frames = []
                        self.current_landmarks = []
                        self.went_down = False
                        self.in_rep = False

                    if self._rep_finished(pos):
                        if len(self.current_frames) >= self.min_len:
                            self.rep_count += 1
                            self.rep_detected.emit(self.rep_count)

                            self.last_text = f"Rep {self.rep_count}: Analyzing..."
                            self.in_rep = False

                            self.rep_to_infer.emit(
                                self.rep_count,
                                list(self.current_frames),
                                list(self.current_landmarks)
                            )

                            show = cv2.cvtColor(self.current_frames[-1], cv2.COLOR_RGB2BGR)
                            self.frame_ready.emit(show, self.last_text)

                        self.current_frames = []
                        self.current_landmarks = []
                        self.went_down = False

                    self._emit_frame(frame, idx)

            self.finished.emit()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.frame_ready.emit(None, f"❌ Error: {e}")
            self.finished.emit()
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def _emit_frame(self, frame_bgr, idx):
        if idx % 2 == 0:
            self.frame_ready.emit(frame_bgr, self.last_text)

    def _get_pos(self, lm):
        if lm is None:
            return None
        p = lm.landmark

        if self.exercise_type == "squat":
            a = p[self.mp_pose.PoseLandmark.LEFT_HIP]
            b = p[self.mp_pose.PoseLandmark.RIGHT_HIP]
        elif self.exercise_type == "shoulder_press":
            a = p[self.mp_pose.PoseLandmark.LEFT_WRIST]
            b = p[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        else:  # push_up
            a = p[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            b = p[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

        if a.visibility > 0.5 and b.visibility > 0.5:
            return (a.y + b.y) / 2.0
        return None

    def _update_thresholds(self, pos):
        self.pos_hist.append(pos)
        if len(self.pos_hist) < 20:
            return

        v = np.array(self.pos_hist, dtype=np.float32)
        m = float(np.median(v))
        s = float(np.std(v) + 1e-6)

        if self.exercise_type == "shoulder_press":
            self.th_up = m - s * 0.5
            self.th_down = m + s * 0.3
        else:
            self.th_down = m + s * 0.5
            self.th_up = m - s * 0.3

    def _rep_finished(self, pos):
        if self.exercise_type == "shoulder_press":
            if pos < self.th_up:
                self.went_down = True
            if self.went_down and pos > self.th_down:
                return True
        else:
            if pos > self.th_down:
                self.went_down = True
            if self.went_down and pos < self.th_up:
                return True
        return False

class BaseVideoUploadWidget(QWidget):
    def __init__(self, hp, model_class, model_path, exercise_type, parent=None):
        super().__init__(parent)
        self.hp = hp
        self.exercise_type = exercise_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        self.processor = None
        self.worker = None
        self.yolo_worker = None 

        self.correct_reps_count = 0
        self.rep_count = 0

        self._build_ui()

    def stop_processing(self):
        if self.processor is not None:
            try:
                if self.processor.isRunning():
                    self.processor.stop()
                    self.processor.wait(3000)
            except Exception:
                pass
            self.processor = None

        if self.worker is not None:
            try:
                if self.worker.isRunning():
                    self.worker.stop()
                    self.worker.wait(3000)
            except Exception:
                pass
            self.worker = None

        if self.yolo_worker is not None:
            try:
                if self.yolo_worker.isRunning():
                    self.yolo_worker.stop()
                    self.yolo_worker.wait(3000)
            except Exception:
                pass
            self.yolo_worker = None

    def closeEvent(self, event):
        self.stop_processing()
        super().closeEvent(event)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setText("📁 Upload a video to analyze")
        self.video_label.setStyleSheet("""
            QLabel {
                background: #1a1a1a;
                border: 2px solid #b31b1b;
                border-radius: 12px;
                color: #F8F9FA;
                font-size: 18px;
            }
        """)

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

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #F8F9FA; font-size: 16px; font-weight: bold;")
        self.yolo_label = QLabel("Barbell ✖️ | Dumbbell ✖️")
        self.yolo_label.setStyleSheet("color: #F8F9FA; font-size: 14px; font-weight: bold;")
        self.rep_counter_label = QLabel("Rep: 0")
        self.rep_counter_label.setStyleSheet("""
            color: #FFA500;
            font-size: 18px;
            font-weight: bold;
            background: rgba(255, 165, 0, 0.1);
            padding: 5px 15px;
            border-radius: 5px;
            border: 2px solid #FFA500;
        """)

        self.correct_counter_label = QLabel("✅ Correct: 0")
        self.correct_counter_label.setStyleSheet("""
            color: #00ff00;
            font-size: 18px;
            font-weight: bold;
            background: rgba(0, 255, 0, 0.1);
            padding: 5px 15px;
            border-radius: 5px;
            border: 2px solid #00ff00;
        """)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #b31b1b;
                border-radius: 5px;
                text-align: center;
                background: #1a1a1a;
                color: #F8F9FA;
                min-width: 200px;
            }
            QProgressBar::chunk {
                background: #b31b1b;
            }
        """)
        self.progress_bar.setVisible(False)
        self.upload_btn = QPushButton("📁 Upload Video")
        self.upload_btn.setCursor(Qt.PointingHandCursor)
        self.upload_btn.setStyleSheet("""
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
            QPushButton:disabled {
                background: #4a4a4a;
                color: #808080;
            }
        """)
        self.upload_btn.clicked.connect(self._upload_video)
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
        status_layout.addSpacing(12)
        status_layout.addWidget(self.yolo_label)
        status_layout.addStretch()
        status_layout.addWidget(self.progress_bar)
        status_layout.addWidget(self.rep_counter_label)
        status_layout.addWidget(self.correct_counter_label)
        status_layout.addWidget(self.reset_counter_btn)
        status_layout.addWidget(self.upload_btn)

        layout.addWidget(self.video_label, 1)
        layout.addWidget(status_frame)

    def _upload_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        if not file_path:
            return
        self._process_video(file_path)

    def _process_video(self, video_path):
        self.stop_processing()

        self.upload_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.correct_reps_count = 0
        self.rep_count = 0
        self.rep_counter_label.setText("Rep: 0")
        self.correct_counter_label.setText("✅ Correct: 0")
        self.status_label.setText("Processing video...")
        self.yolo_label.setText("Barbell ✖️ | Dumbbell ✖️")
        self.yolo_worker = YOLOPresenceWorker()
        self.yolo_worker.presence_ready.connect(self._on_yolo_presence)
        self.yolo_worker.start()
        self.worker = InferenceWorker(
            model=self.model,
            hp=self.hp,
            transform=self.transform,
            device=self.device
        )
        self.worker.result_ready.connect(self._on_worker_result)
        self.worker.error_ready.connect(self._on_worker_error)
        self.worker.start()
        self.processor = VideoProcessor(
            video_path=video_path,
            hp=self.hp,
            exercise_type=self.exercise_type
        )
        self.processor.frame_ready.connect(self._on_frame_ready)
        self.processor.rep_detected.connect(self._on_rep_detected)
        self.processor.rep_to_infer.connect(self.worker.enqueue_rep)
        self.processor.progress_update.connect(self._on_progress_update)
        self.processor.yolo_frame.connect(self.yolo_worker.enqueue_frame)
        self.processor.finished.connect(self._on_processing_finished)
        self.processor.start()

    def _on_frame_ready(self, frame, status_text):
        if frame is not None:
            h, w = frame.shape[:2]
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

    def _on_yolo_presence(self, barbell, dumbbell):
        b = "✔️" if barbell else "✖️"
        d = "✔️" if dumbbell else "✖️"
        self.yolo_label.setText(f"Barbell {b} | Dumbbell {d}")

    def _on_rep_detected(self, rep_number):
        self.rep_count = rep_number
        self.rep_counter_label.setText(f"Rep: {rep_number}")

    def _on_worker_result(self, rep_number, result):
        label = result["label"]
        conf = result["conf"]
        txt = f"Rep {rep_number}: {label} ({conf:.1f}%)"

        if self.processor:
            self.processor.set_last_text(txt)

        self.status_label.setText(txt)

        if label.lower() == "correct":
            self.correct_reps_count += 1
            self.correct_counter_label.setText(f"✅ Correct: {self.correct_reps_count}")

    def _on_worker_error(self, rep_number, err):
        txt = f"Rep {rep_number}: Error ({err})"
        if self.processor:
            self.processor.set_last_text(txt)
        self.status_label.setText(txt)

    def _on_progress_update(self, progress):
        self.progress_bar.setValue(progress)

    def _on_processing_finished(self):
        self.upload_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        if self.processor:
            self.status_label.setText(self.processor.last_text)

        if self.yolo_worker is not None:
            try:
                self.yolo_worker.stop()
                self.yolo_worker.wait(2000)
            except Exception:
                pass

    def _reset_counter(self):
        self.correct_reps_count = 0
        self.rep_count = 0
        self.rep_counter_label.setText("Rep: 0")
        self.correct_counter_label.setText("✅ Correct: 0")


class SquatVideoUploadWidget(BaseVideoUploadWidget):
    def __init__(self, parent=None):
        hp = config.HP
        model_path = config.MODELS_DIR / "squats" / "squats_twostream_mobilenet_GRU_model.pt"
        super().__init__(hp, SquatModel, model_path, "squat", parent)


class ShoulderPressVideoUploadWidget(BaseVideoUploadWidget):
    def __init__(self, parent=None):
        hp = config.SHOULDER_PRESS_HP
        model_path = config.MODELS_DIR / "shoulder_press" / "shoulder_press_twostream_mobilenet_GRU_model.pt"
        super().__init__(hp, ShoulderModel, model_path, "shoulder_press", parent)


class PushUpVideoUploadWidget(BaseVideoUploadWidget):
    def __init__(self, parent=None):
        hp = config.PUSH_UP_HP
        model_path = config.MODELS_DIR / "push_up" / "push_up_twostream_mobilenet_GRU_model.pt"
        super().__init__(hp, PushUpModel, model_path, "push_up", parent)