import tensorflow as tf
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from config import config
from data_preprocess import data_generator

def get_scan_ids(log_file, root_dir):
    """
    Helper to get scan IDs from a CSV log or by listing directories.
    """
    if os.path.exists(log_file):
        try:
            df = pd.read_csv(log_file)
            pass 
        except:
            pass
            
    # ROBUST FALLBACK: List directories
    scan_ids = [d for d in os.listdir(root_dir) 
                if os.path.isdir(os.path.join(root_dir, d)) and "BraTS" in d]
    return sorted(scan_ids)

def get_train_val_datasets():
    """
    Loads data from Brats_Scan/Train-Val and splits it 80/20.
    """
    print(f"Looking for Training data in: {config.TRAIN_VAL_DIR}")
    all_ids = get_scan_ids(config.TRAIN_LOG_FILE, config.TRAIN_VAL_DIR)
    
    if not all_ids:
        raise ValueError(f"No data found in {config.TRAIN_VAL_DIR}")

    print(f"Found {len(all_ids)} scans. Generating {len(all_ids)*4} modality examples.")
    
    # Split
    train_ids, val_ids = train_test_split(
        all_ids, 
        test_size=config.VAL_SPLIT, 
        random_state=config.SEED
    )
    
    # Define Signature
    # CRITICAL: First element is (H, W, 1) because config.NUM_CHANNELS is 1
    # Second element is (H, W, 4) because config.NUM_CLASSES is 4
    output_signature = (
        tf.TensorSpec(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.NUM_CHANNELS), dtype=tf.float32),
        tf.TensorSpec(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.NUM_CLASSES), dtype=tf.float32)
    )

    # Generators
    train_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(train_ids, config.TRAIN_VAL_DIR, is_train=True),
        output_signature=output_signature
    )

    val_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(val_ids, config.TRAIN_VAL_DIR, is_train=False),
        output_signature=output_signature
    )
    
    # Optimization
    train_ds = train_ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, len(train_ids)*4, len(val_ids)*4

def get_test_dataset():
    """
    Loads data from Brats_Scan/Test. No splitting.
    """
    print(f"Looking for Test data in: {config.TEST_DIR}")
    test_ids = get_scan_ids(config.TEST_LOG_FILE, config.TEST_DIR)
    
    if not test_ids:
        raise ValueError(f"No data found in {config.TEST_DIR}")

    print(f"Found {len(test_ids)} scans for Testing ({len(test_ids)*4} examples).")

    output_signature = (
        tf.TensorSpec(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.NUM_CHANNELS), dtype=tf.float32),
        tf.TensorSpec(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.NUM_CLASSES), dtype=tf.float32)
    )

    test_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(test_ids, config.TEST_DIR, is_train=False),
        output_signature=output_signature
    )
    
    test_ds = test_ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return test_ds, len(test_ids)*4