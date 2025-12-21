import os

class Config:
    # -- PATHS --
    # Root of the dataset
    BASE_DIR = "Brats_Scan"
    
    # Train-Val paths
    TRAIN_VAL_DIR = os.path.join(BASE_DIR, "Train-Val")
    TRAIN_LOG_FILE = os.path.join(TRAIN_VAL_DIR, "train_dataset_log") # Assuming csv content
    
    # Test paths
    TEST_DIR = os.path.join(BASE_DIR, "Test")
    TEST_LOG_FILE = os.path.join(TEST_DIR, "test_dataset_log")
    
    # Output paths
    CHECKPOINT_DIR = "checkpoints"
    LOGS_DIR = "logs"
    RESULTS_DIR = "results"
    EXEC_LOG_FILE = os.path.join(LOGS_DIR, "execution.log")
    
    # -- IMAGE SPECS --
    # Kept at 256 to solve the U-Net shape mismatch error
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    NUM_CHANNELS = 4  # t1c, t1n, t2f, t2w
    NUM_CLASSES = 4   # 0: Background, 1: NCR, 2: ED, 3: ET
    
    # -- HYPERPARAMETERS --
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    VAL_SPLIT = 0.2  # 20% of Train-Val folder used for validation
    SEED = 42
    
    # Training Strategy
    EARLY_STOPPING_PATIENCE = 15
    
    # Model Architecture
    FILTERS = 16
    DROPOUT_RATE = 0.1
    
    def __init__(self):
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.LOGS_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)

config = Config()