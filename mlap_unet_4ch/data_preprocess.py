import tensorflow as tf
import cv2
import numpy as np
import os
import random
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
    """Loads the segmentation mask from PNG."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    
    img = cv2.resize(img, (config.IMG_WIDTH, config.IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
    img = np.round(img / 85.0).astype(np.int32)
    img = np.clip(img, 0, config.NUM_CLASSES - 1)
    
    return img

def process_path(scan_id, root_dir):
    """Constructs file paths. Priority is PNG."""
    scan_dir = os.path.join(root_dir, scan_id)
    ext = ".png"
    if not os.path.exists(os.path.join(scan_dir, f"{scan_id}-t1c{ext}")):
        ext = ".jpg"

    p_t1c = os.path.join(scan_dir, f"{scan_id}-t1c{ext}")
    p_t1n = os.path.join(scan_dir, f"{scan_id}-t1n{ext}")
    p_t2f = os.path.join(scan_dir, f"{scan_id}-t2f{ext}")
    p_t2w = os.path.join(scan_dir, f"{scan_id}-t2w{ext}")
    p_seg = os.path.join(scan_dir, f"{scan_id}-seg{ext}")
    
    return p_t1c, p_t1n, p_t2f, p_t2w, p_seg

def augment_data(X, Y):
    """
    Applies random 90-degree rotations and flips.
    Crucial: Applies same transform to Image (X) and Mask (Y).
    """
    # 1. Random Flip Left-Right
    if random.random() < 0.5:
        X = np.fliplr(X)
        Y = np.fliplr(Y)
        
    # 2. Random Flip Up-Down
    if random.random() < 0.5:
        X = np.flipud(X)
        Y = np.flipud(Y)
        
    # 3. Random Rotation (0, 90, 180, 270)
    k = random.randint(0, 3)
    if k > 0:
        X = np.rot90(X, k=k)
        Y = np.rot90(Y, k=k)
        
    return X, Y

def data_generator(scan_ids, root_dir, is_train=True):
    """
    Generator with On-the-Fly Augmentation.
    """
    for scan_id in scan_ids:
        try:
            t1c_p, t1n_p, t2f_p, t2w_p, seg_p = process_path(scan_id, root_dir)
            
            mask = load_mask(seg_p)
            Y_raw = tf.one_hot(mask, depth=config.NUM_CLASSES).numpy() # Shape: (256, 256, 4)
            
            img_t1c = load_image(t1c_p)
            img_t1n = load_image(t1n_p)
            img_t2f = load_image(t2f_p)
            img_t2w = load_image(t2w_p)
            
            # Stack: (256, 256, 4)
            X_raw = np.stack([img_t1c, img_t1n, img_t2f, img_t2w], axis=-1)
            
            # Apply Augmentation ONLY if training
            if is_train:
                X, Y = augment_data(X_raw, Y_raw)
            else:
                X, Y = X_raw, Y_raw
            
            yield X, Y
            
        except Exception as e:
            continue