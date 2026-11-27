
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset directory
DATA_DIR = BASE_DIR / "data"

# Trained model files
MODEL_PATH = BASE_DIR / "mobilenet_custom.pth"
LABELS_PATH = BASE_DIR / "class_names.txt"

# Training settings
EPOCHS = 5
BATCH_SIZE = 16
LR = 0.0001
VAL_RATIO = 0.2
IMG_SIZE = 224
