import tensorflow as tf
import cv2
import numpy as np
import os
from config import config

def load_image(path):
    """Loads a grayscale image, normalizes it to [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    
    img = cv2.resize(img, (config.IMG_WIDTH, config.IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    return img

def load_mask(path):
    """
    Loads the segmentation mask from PNG.
    Handles user specific scaling: 0, 85, 170, 255 -> 0, 1, 2, 3
    """
    # Read image (Lossless PNG)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    
    # Resize Nearest Neighbor to avoid interpolation artifacts
    img = cv2.resize(img, (config.IMG_WIDTH, config.IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
    
    # Map 0, 85, 170, 255 -> 0, 1, 2, 3
    # Division by 85 gives: 0, 1, 2, 3 exactly
    img = np.round(img / 85.0).astype(np.int32)
    
    # Safety clip
    img = np.clip(img, 0, config.NUM_CLASSES - 1)
    
    return img

def process_path(scan_id, root_dir):
    """
    Constructs file paths. Priority is PNG.
    """
    scan_dir = os.path.join(root_dir, scan_id)
    
    # Check for PNG (default new format)
    ext = ".png"
    if not os.path.exists(os.path.join(scan_dir, f"{scan_id}-t1c{ext}")):
        ext = ".jpg" # Fallback

    p_t1c = os.path.join(scan_dir, f"{scan_id}-t1c{ext}")
    p_t1n = os.path.join(scan_dir, f"{scan_id}-t1n{ext}")
    p_t2f = os.path.join(scan_dir, f"{scan_id}-t2f{ext}")
    p_t2w = os.path.join(scan_dir, f"{scan_id}-t2w{ext}")
    p_seg = os.path.join(scan_dir, f"{scan_id}-seg{ext}")
    
    return p_t1c, p_t1n, p_t2f, p_t2w, p_seg

def data_generator(scan_ids, root_dir, is_train=True):
    """
    Generator function for tf.data.Dataset.
    YIELDS: (X, Y) where X is (256, 256, 1) and Y is (256, 256, 4)
    It splits one patient into 4 independent training samples.
    """
    for scan_id in scan_ids:
        try:
            t1c_p, t1n_p, t2f_p, t2w_p, seg_p = process_path(scan_id, root_dir)
            
            # Load Mask (Common target for all modalities)
            mask = load_mask(seg_p)
            Y = tf.one_hot(mask, depth=config.NUM_CLASSES) # Shape: (256, 256, 4)
            
            # Paths to iterate
            modality_paths = [t1c_p, t1n_p, t2f_p, t2w_p]
            
            for mod_path in modality_paths:
                # Load Image
                img = load_image(mod_path) # Shape: (256, 256)
                
                # Expand dims to make it (H, W, 1)
                X = np.expand_dims(img, axis=-1) # Shape: (256, 256, 1)
                
                yield X, Y
            
        except Exception as e:
            # print(f"Error processing {scan_id}: {e}")
            continue