import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QFrame,
    QHBoxLayout, QVBoxLayout, QGridLayout,
    QGraphicsDropShadowEffect, QPushButton, QButtonGroup
)

from modules.config import APP_LOGO, SQUATS_LOGO, SHOULDER_PRESS_LOGO, PUSHUP_LOGO

ACCENT = "#b31b1b"
BG = "#E7E0DA"
TEXT = "#F8F9FA"
MUTED = "#8F1414"

EXERCISE_APPS = {
    "squats": {
        "title": "Squats",
        "subtitle": "Real-time Squats Analysis and Rep Counting.",
        "icon": SQUATS_LOGO,
    },
    "shoulder_press": {
        "title": "Shoulder Press",
        "subtitle": "Track Form, Tempo, and Range of Motion.",
        "icon": SHOULDER_PRESS_LOGO,
    },
    "pushup": {
        "title": "Push Up",
        "subtitle": "Posture Detection with Live Feedback.",
        "icon": PUSHUP_LOGO,
    },
}


def shadow(widget, blur=20, y=6):
    eff = QGraphicsDropShadowEffect()
    eff.setBlurRadius(blur)
    eff.setXOffset(0)
    eff.setYOffset(y)
    eff.setColor(Qt.black)
    widget.setGraphicsEffect(eff)

class ClickableExerciseCard(QFrame):
    def __init__(self, meta: dict, on_click, parent=None):
        super().__init__(parent)
        self.meta = meta
        self.on_click = on_click

        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(160)

        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(24, 20, 24, 20)
        self.main_layout.setSpacing(24)

        self.icon = QLabel()
        self.icon.setFixedSize(72, 72)
        self.icon.setAlignment(Qt.AlignCenter)

        icon_path = Path(meta.get("icon", ""))
        if icon_path.exists():
            pm = QPixmap(str(icon_path)).scaled(
                72, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.icon.setPixmap(pm)

        self.icon.setStyleSheet("""
            QLabel {
                background: #b31b1b;
                border: 1px solid #b31b1b;
                border-radius: 12px;
            }
        """)

        title = QLabel(meta["title"])
        title.setStyleSheet(f"color:{TEXT}; font-size:20px; font-weight:900;")

        subtitle = QLabel(meta["subtitle"])
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"color:{TEXT}; font-size:14px;")

        text = QVBoxLayout()
        text.addWidget(title)
        text.addWidget(subtitle)
        text.addStretch(1)

        self.main_layout.addWidget(self.icon)
        self.main_layout.addLayout(text)

        self.setStyleSheet(f"""
            QFrame {{
                background: {ACCENT};
                border: 1px solid {ACCENT};
                border-radius: 20px;
            }}
        """)
        shadow(self, 24, 8)

    def enterEvent(self, e):
        self.main_layout.setContentsMargins(24, 12, 24, 28)
        self.setStyleSheet(f"""
            QFrame {{
                background: #c92222;
                border: 2px solid #c92222;
                border-radius: 20px;
            }}
        """)
        self.icon.setStyleSheet("""
            QLabel {
                background: #c92222;
                border: 2px solid #c92222;
                border-radius: 12px;
            }
        """)

    def leaveEvent(self, e):
        self.main_layout.setContentsMargins(24, 20, 24, 20)
        self.setStyleSheet(f"""
            QFrame {{
                background: {ACCENT};
                border: 1px solid {ACCENT};
                border-radius: 20px;
            }}
        """)
        self.icon.setStyleSheet("""
            QLabel {
                background: #b31b1b;
                border: 1px solid #b31b1b;
                border-radius: 12px;
            }
        """)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.on_click(self.meta)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Pose Fit AI")
        self.resize(1200, 700)

        self.root = QWidget()
        self.root.setStyleSheet(f"background:{BG};")
        self.setCentralWidget(self.root)

        main = QVBoxLayout(self.root)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        logo_container = QWidget()
        logo_container.setFixedHeight(120)

        logo_layout = QHBoxLayout(logo_container)
        logo_layout.setContentsMargins(40, 20, 0, 0)
        logo_layout.setSpacing(0)

        logo_rect = QLabel()
        logo_rect.setFixedSize(400, 100)

        pm = QPixmap(str(APP_LOGO)).scaled(
            400, 100,
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation
        )
        logo_rect.setPixmap(pm)
        logo_rect.setScaledContents(True)
        logo_rect.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        shadow(logo_rect, blur=20, y=6)

        logo_layout.addWidget(logo_rect, alignment=Qt.AlignLeft)
        logo_layout.addStretch(1)

        main.addWidget(logo_container)

        content_container = QWidget()
        content_main = QVBoxLayout(content_container)
        content_main.setContentsMargins(40, 30, 40, 30)
        content_main.setSpacing(30)

        content = QHBoxLayout()
        content.setSpacing(40)

        left = QVBoxLayout()
        tag = QLabel("AN AI GYM COACH")
        tag.setAlignment(Qt.AlignCenter)
        tag.setFixedWidth(180)
        tag.setStyleSheet(f"""
            QLabel {{
                color:{ACCENT};
                border:1px solid {ACCENT};
                border-radius:18px;
                padding:6px 0;
                font-weight:900;
            }}
        """)

        headline = QLabel("START YOUR\nFITNESS JOURNEY\nTODAY")
        headline.setStyleSheet(f"""
            color:{ACCENT};
            font-size:50px;
            font-weight:950;
            line-height:1.05;
        """)

        desc = QLabel(
            "Your Smart AI Gym Coach.\n"
            "Choose an Exercise to Start Real-Time Pose Analysis,\n"
            "Rep Counting, and Instant Feedback Using the Camera."
        )
        desc.setStyleSheet(f"color:{MUTED}; font-size:14px;")

        left.addWidget(tag)
        left.addWidget(headline)
        left.addWidget(desc)
        left.addStretch(1)

        right = QVBoxLayout()
        choose = QLabel("Choose Your Exercise")
        choose.setStyleSheet(f"color:{ACCENT}; font-size:40px; font-weight:900;")
        choose.setAlignment(Qt.AlignCenter)

        cards = QGridLayout()
        cards.setSpacing(20)

        def open_exercise(meta):
            self.start_live(meta["title"])

        cards.addWidget(ClickableExerciseCard(EXERCISE_APPS["squats"], open_exercise), 0, 0)
        cards.addWidget(ClickableExerciseCard(EXERCISE_APPS["shoulder_press"], open_exercise), 1, 0)
        cards.addWidget(ClickableExerciseCard(EXERCISE_APPS["pushup"], open_exercise), 2, 0)

        right.addWidget(choose)
        right.addLayout(cards)
        right.addStretch(1)

        content.addLayout(left, 3)
        content.addLayout(right, 2)

        content_main.addLayout(content)
        main.addWidget(content_container)

        self._camera = None
        self._live_widget = None
        self._mode = "live" 

        self.live_container = QFrame(self.root)
        self.live_container.setStyleSheet(f"""
            QFrame {{
                background: {ACCENT};
                border: 2px solid #c92222;
                border-radius: 20px;
            }}
        """)
        shadow(self.live_container, 24, 10)
        self.live_container.hide()

        self.live_layout = QVBoxLayout(self.live_container)
        self.live_layout.setContentsMargins(16, 16, 16, 16)
        self.live_layout.setSpacing(12)

        topbar = QHBoxLayout()
        self.btn_back = QPushButton("← Back")
        self.btn_back.setCursor(Qt.PointingHandCursor)
        self.btn_back.setStyleSheet(f"""
            QPushButton {{
                color:{TEXT};
                background:#8F1414;
                border:1px solid #6F0F0F;
                padding:10px 14px;
                border-radius:12px;
                font-weight:800;
            }}
            QPushButton:hover {{
                background:#a01818;
                border:2px solid #c92222;
            }}
        """)
        self.btn_back.clicked.connect(self.stop_live)
        self.btn_live = QPushButton("🎥 Live Camera")
        self.btn_live.setCursor(Qt.PointingHandCursor)
        self.btn_live.setCheckable(True)
        self.btn_live.setChecked(True)
        self.btn_upload = QPushButton("📁 Upload Video")
        self.btn_upload.setCursor(Qt.PointingHandCursor)
        self.btn_upload.setCheckable(True)

        mode_button_style = f"""
            QPushButton {{
                color:{TEXT};
                background:#8F1414;
                border:1px solid #6F0F0F;
                padding:10px 14px;
                border-radius:12px;
                font-weight:800;
            }}
            QPushButton:hover {{
                background:#a01818;
                border:2px solid #c92222;
            }}
            QPushButton:checked {{
                background:#b31b1b;
                border:2px solid #c92222;
            }}
        """
        self.btn_live.setStyleSheet(mode_button_style)
        self.btn_upload.setStyleSheet(mode_button_style)
        self.btn_live.clicked.connect(lambda: self._switch_mode("live"))
        self.btn_upload.clicked.connect(lambda: self._switch_mode("upload"))
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.btn_live)
        self.mode_group.addButton(self.btn_upload)
        self.live_title = QLabel("")
        self.live_title.setStyleSheet(f"color:{TEXT}; font-size:18px; font-weight:900;")

        topbar.addWidget(self.btn_back)
        topbar.addSpacing(10)
        topbar.addWidget(self.live_title)
        topbar.addStretch(1)
        topbar.addWidget(self.btn_live)
        topbar.addSpacing(5)
        topbar.addWidget(self.btn_upload)
        topbar.addStretch(1)

        self.live_layout.addLayout(topbar)
        self.live_body = QFrame()
        self.live_body.setStyleSheet("background: transparent;")
        self.live_body_layout = QVBoxLayout(self.live_body)
        self.live_body_layout.setContentsMargins(0, 0, 0, 0)
        self.live_body_layout.setSpacing(0)
        self.live_layout.addWidget(self.live_body, 1)

        self._fit_live_container()

    def _fit_live_container(self):
        m = 20
        self.live_container.setGeometry(m, m, self.root.width() - 2*m, self.root.height() - 2*m)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit_live_container()

    def _cleanup_live_widget(self):
        w = self._live_widget
        if w is None:
            return
        try:
            if hasattr(w, "stop_processing"):
                w.stop_processing()
            elif hasattr(w, "stop"):
                w.stop()
        except Exception:
            pass

    def _cleanup_camera(self):
        try:
            if self._camera is not None:
                self._camera.close()
                self._camera = None
        except Exception:
            pass

    def _switch_mode(self, mode):
        if mode == self._mode:
            return

        self._mode = mode
        current_title = self.live_title.text()

        if self._live_widget is not None:
            self._cleanup_live_widget()
            self._live_widget.setParent(None)
            self._live_widget.deleteLater()
            self._live_widget = None

        if current_title:
            self.start_live(current_title)

    def start_live(self, title_text: str):
        if self._mode == "live":
            self._start_live_camera(title_text)
        else:
            self._start_video_upload(title_text)

    def _start_live_camera(self, title_text: str):
        from modules.ui.realtime_widgets import (
            CameraManager,
            SquatWidget,
            ShoulderPressWidget,
            PushUpWidget,
        )

        if self._camera is None:
            self._camera = CameraManager(camera_index=0, w=1280, h=720, fps_ms=15)

        if self._live_widget is not None:
            self._cleanup_live_widget()
            self._live_widget.setParent(None)
            self._live_widget.deleteLater()
            self._live_widget = None

        if "squat" in title_text.lower():
            self._live_widget = SquatWidget(self._camera)
        elif "shoulder" in title_text.lower():
            self._live_widget = ShoulderPressWidget(self._camera)
        else:
            self._live_widget = PushUpWidget(self._camera)

        self.live_title.setText(title_text)
        self.live_body_layout.addWidget(self._live_widget)

        self.live_container.show()
        self.live_container.raise_()

    def _start_video_upload(self, title_text: str):
        from modules.ui.video_upload_widgets import (
            SquatVideoUploadWidget,
            ShoulderPressVideoUploadWidget,
            PushUpVideoUploadWidget,
        )

        if self._live_widget is not None:
            self._cleanup_live_widget()
            self._live_widget.setParent(None)
            self._live_widget.deleteLater()
            self._live_widget = None

        if "squat" in title_text.lower():
            self._live_widget = SquatVideoUploadWidget()
        elif "shoulder" in title_text.lower():
            self._live_widget = ShoulderPressVideoUploadWidget()
        else:
            self._live_widget = PushUpVideoUploadWidget()

        self.live_title.setText(title_text)
        self.live_body_layout.addWidget(self._live_widget)
        self.live_container.show()
        self.live_container.raise_()

    def stop_live(self):
        self.live_container.hide()

        if self._live_widget is not None:
            self._cleanup_live_widget()
            self._live_widget.setParent(None)
            self._live_widget.deleteLater()
            self._live_widget = None
            self._cleanup_camera()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            if self.live_container.isVisible():
                self.stop_live()
                return
            self.close()
            return
        super().keyPressEvent(e)

    def closeEvent(self, e):
        try:
            self.stop_live()
        except Exception:
            pass

        self._cleanup_camera()
        super().closeEvent(e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI"))
    w = MainWindow()
    w.show()
    app.exec()