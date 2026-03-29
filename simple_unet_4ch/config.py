import os
import matplotlib.pyplot as plt

class Config:
    # -- PATHS --
    BASE_DIR = "Brats_Scan"
    TRAIN_VAL_DIR = os.path.join(BASE_DIR, "Train-Val")
    TRAIN_LOG_FILE = os.path.join(TRAIN_VAL_DIR, "train_dataset_log") 
    TEST_DIR = os.path.join(BASE_DIR, "Test")
    TEST_LOG_FILE = os.path.join(TEST_DIR, "test_dataset_log")
    
    # -- OUTPUT STRUCTURE --
    OUTPUT_ROOT = "u-net_outputs"
    CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, "models")
    LOGS_DIR = os.path.join(OUTPUT_ROOT, "logs")
    RESULTS_DIR = os.path.join(OUTPUT_ROOT, "results")
    PLOTS_DIR = os.path.join(OUTPUT_ROOT, "plots")
    EXEC_LOG_FILE = os.path.join(LOGS_DIR, "execution.log")
    
    # -- IMAGE SPECS --
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    NUM_CHANNELS = 4  # 4-Channel Input (Stacked T1c, T1n, T2f, T2w)
    NUM_CLASSES = 4   # 0: Background, 1: NCR, 2: ED, 3: ET
    CLASS_NAMES = ["Background", "NCR (Necrotic)", "ED (Edema)", "ET (Enhancing)"]
    
    # -- HYPERPARAMETERS --
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    VAL_SPLIT = 0.2
    SEED = 42
    EARLY_STOPPING_PATIENCE = 15
    
    # Model Architecture
    FILTERS = 32      # UPGRADED: 32 base filters -> 1024 at the Bridge
    DROPOUT_RATE = 0.1
    
    def __init__(self):
        # Create output directories
        for d in [self.CHECKPOINT_DIR, self.LOGS_DIR, self.RESULTS_DIR, self.PLOTS_DIR]:
            os.makedirs(d, exist_ok=True)
            
        # -- GLOBAL PLOT SETTINGS (Times New Roman) --
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif", "serif"],
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 300
        })

config = Config()