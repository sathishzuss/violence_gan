import os

class Config:
    # Paths
    DATA_DIR = "data"
    TRAIN_VID_DIR = os.path.join(DATA_DIR, "videos", "train")
    TEST_VID_DIR = os.path.join(DATA_DIR, "videos", "test")
    TRAIN_FRAME_DIR = os.path.join(DATA_DIR, "train")
    TEST_FRAME_DIR = os.path.join(DATA_DIR, "test")
    OUTPUT_DIR = "output"
    LOG_DIR = "logs"

    # Frame extraction
    FRAME_RATE = 1  # 1 frame per second

    # Model training
    IMAGE_SIZE = 64
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 0.0002
    LATENT_DIM = 100

    # Device
    DEVICE = "cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu"

    # Detection threshold
    ANOMALY_THRESHOLD = 0.25
