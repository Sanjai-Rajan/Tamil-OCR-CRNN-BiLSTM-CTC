from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA = BASE_DIR / "data/raw"
PROCESSED_DATA = BASE_DIR / "data/processed"
SEGMENTED_DATA = BASE_DIR / "data/segmented"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "outputs"

DEVICE = "cuda"

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 512

BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
