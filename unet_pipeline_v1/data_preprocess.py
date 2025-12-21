import tensorflow as tf
import cv2
import numpy as np
import os
from config import config

def load_image(path):
    """Loads a grayscale image, normalizes it to [0, 1]."""
    # Read as grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    
    img = cv2.resize(img, (config.IMG_WIDTH, config.IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    return img

def load_mask(path):
    """
    Loads the segmentation mask. 
    Since the source is JPG (lossy), we assume pixel values correspond to 
    classes [0, 1, 2, 3] scaled or raw. 
    We snap to the nearest integer to handle JPG artifacts.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    
    img = cv2.resize(img, (config.IMG_WIDTH, config.IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
    
    # Unique values in BraTS are typically 0, 1, 2, 3 (or 4).
    # If saved as standard image, they might be barely visible (values 1, 2, 3).
    # If they were scaled (e.g. 0, 85, 170, 255), we would need to map them back.
    # ASSUMPTION: The JPGs contain raw integers 0-3 but might have noise like 0.1 or 2.9 due to compression.
    
    # Thresholding logic to clean compression noise
    # We round to nearest integer to recover class ID
    img = np.round(img).astype(np.int32)
    
    # Safety: Clip to max classes (assuming 4 classes: 0, 1, 2, 3)
    img = np.clip(img, 0, config.NUM_CLASSES - 1)
    
    return img

def process_path(scan_id, root_dir):
    """
    Constructs file paths based on the directory structure provided.
    Structure: Root/ScanID/ScanID-modality.jpg
    """
    # Scan folder name matches Scan ID
    scan_dir = os.path.join(root_dir, scan_id)
    
    # Construct filenames (assuming .jpg based on prompt)
    # Note: Prompt said "BraTS-GLI-00000-000-t1c", checking for extension usually .jpg or .png
    # We will try to detect, or default to .jpg as requested.
    ext = ".jpg" 
    
    p_t1c = os.path.join(scan_dir, f"{scan_id}-t1c{ext}")
    p_t1n = os.path.join(scan_dir, f"{scan_id}-t1n{ext}")
    p_t2f = os.path.join(scan_dir, f"{scan_id}-t2f{ext}")
    p_t2w = os.path.join(scan_dir, f"{scan_id}-t2w{ext}")
    p_seg = os.path.join(scan_dir, f"{scan_id}-seg{ext}")
    
    return p_t1c, p_t1n, p_t2f, p_t2w, p_seg

def data_generator(scan_ids, root_dir, is_train=True):
    """
    Generator function for tf.data.Dataset
    """
    for scan_id in scan_ids:
        try:
            t1c_p, t1n_p, t2f_p, t2w_p, seg_p = process_path(scan_id, root_dir)
            
            # Load Modalities
            t1c = load_image(t1c_p)
            t1n = load_image(t1n_p)
            t2f = load_image(t2f_p)
            t2w = load_image(t2w_p)
            
            # Stack modalities: (H, W, 4)
            X = np.stack([t1c, t1n, t2f, t2w], axis=-1)
            
            # Load Mask
            mask = load_mask(seg_p)
            
            # One-hot encode mask: (H, W, Num_Classes)
            # 0 -> [1,0,0,0], 1 -> [0,1,0,0], etc.
            Y = tf.one_hot(mask, depth=config.NUM_CLASSES)
            
            yield X, Y
            
        except Exception as e:
            print(f"Error processing {scan_id}: {e}")
            continue