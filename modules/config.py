from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "squats"
PROCESSED_DIR = PROJECT_ROOT / "data" / "squats_processed"
SHOULDER_PRESS_DATA_DIR = PROJECT_ROOT / "data" / "shoulder_press"
SHOULDER_PRESS_PROCESSED_DIR = PROJECT_ROOT / "data" / "shoulder_press_processed"
PUSH_UP_DATA_DIR = PROJECT_ROOT / "data" / "push_up"
PUSH_UP_PROCESSED_DIR = PROJECT_ROOT / "data" / "push_up_processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
EVAL_DIR = OUTPUTS_DIR / "eval"
YOLO_DIR = PROJECT_ROOT / "yolo_training"
YOLO_WEIGHTS = YOLO_DIR / "runs" / "train" / "weights" / "best.pt"   
YOLO_DUMBELL_CLASS_ID = 0  
YOLO_BARBELL_CLASS_ID = 1  
CLASS_NAMES = {
    YOLO_DUMBELL_CLASS_ID: "dumbbell",
    YOLO_BARBELL_CLASS_ID: "barbell",
}
APP_LOGO = PROJECT_ROOT / "assest" / "app_logo.png"
SQUATS_LOGO = PROJECT_ROOT / "assest" / "squat_logo.png"
SHOULDER_PRESS_LOGO = PROJECT_ROOT / "assest" / "shoulder_press_logo.png"
PUSHUP_LOGO = PROJECT_ROOT / "assest" / "pushup_logo.png"
SEED = 42
TRAIN_SPLIT = 0.8

@dataclass
class SquatHyperparameters:
    class_names: tuple = ("correct", "fast", "incomplete", "wrong_position")
    num_frames: int = 40
    img_size: int = 224
    batch_size: int = 8
    num_epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 2  
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    gru_hidden: int = 256
    gru_layers: int = 1
    dropout: float = 0.3
    freeze_backbone: bool = True 
    webcam_id: int = 0
    mp_model_complexity: int = 1
    calib_seconds: float = 2.0
    up_offset: float = 0.10
    down_offset: float = 0.10
    smoothing: int = 5

@dataclass
class ShoulderPressHyperparameters:
    class_names: tuple = ("correct", "fast", "incomplete", "wrong_position")
    num_frames: int = 40
    img_size: int = 224
    batch_size: int = 4
    num_epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    gru_hidden: int = 256
    gru_layers: int = 1
    dropout: float = 0.35
    freeze_backbone: bool = True
    webcam_id: int = 0
    mp_model_complexity: int = 1
    calib_seconds: float = 2.0
    up_offset: float = 0.15  
    down_offset: float = 0.15
    smoothing: int = 5


@dataclass
class PushUpHyperparameters:
    class_names: tuple = ("correct", "bad_body_alignment", "incomplete")
    num_frames: int = 24
    img_size: int = 224
    batch_size: int = 6
    num_epochs: int = 60
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    gru_hidden: int = 256
    gru_layers: int = 1
    dropout: float = 0.3
    freeze_backbone: bool = True
    webcam_id: int = 0
    mp_model_complexity: int = 1
    calib_seconds: float = 2.0
    up_offset: float = 0.12 
    down_offset: float = 0.12
    smoothing: int = 5


HP = SquatHyperparameters()
SHOULDER_PRESS_HP = ShoulderPressHyperparameters()
PUSH_UP_HP = PushUpHyperparameters()

def ensure_dirs():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SHOULDER_PRESS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PUSH_UP_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "squats").mkdir(exist_ok=True)
    (MODELS_DIR / "shoulder_press").mkdir(exist_ok=True)
    (MODELS_DIR / "push_up").mkdir(exist_ok=True)
    (EVAL_DIR / "squats").mkdir(exist_ok=True)
    (EVAL_DIR / "shoulder_press").mkdir(exist_ok=True)
    (EVAL_DIR / "push_up").mkdir(exist_ok=True)