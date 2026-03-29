import tensorflow as tf
import numpy as np
import cv2
import os
import argparse
from config import config
from data_preprocess import load_image
from losses import combined_loss, dice_coef
import utils

def predict_single_scan(scan_folder):
    # UPDATED MODEL PATH
    model_path = os.path.join(config.CHECKPOINT_DIR, "u-net_brats_best.keras")
    
    custom_objects = {"combined_loss": combined_loss, "dice_coef": dice_coef}
    
    if not os.path.exists(model_path):
        print("Model not found. Please train the Deep U-Net first.")
        return

    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    scan_id = os.path.basename(scan_folder)
    print(f"Processing Patient: {scan_id} (4-Channel Deep U-Net)...")
    
    ext = ".jpg" 
    f_t1c = os.path.join(scan_folder, f"{scan_id}-t1c.png")
    if not os.path.exists(f_t1c): f_t1c = os.path.join(scan_folder, f"{scan_id}-t1c.jpg")
    
    f_t1n = os.path.join(scan_folder, f"{scan_id}-t1n.png")
    if not os.path.exists(f_t1n): f_t1n = os.path.join(scan_folder, f"{scan_id}-t1n.jpg")
    
    f_t2f = os.path.join(scan_folder, f"{scan_id}-t2f.png")
    if not os.path.exists(f_t2f): f_t2f = os.path.join(scan_folder, f"{scan_id}-t2f.jpg")
    
    f_t2w = os.path.join(scan_folder, f"{scan_id}-t2w.png")
    if not os.path.exists(f_t2w): f_t2w = os.path.join(scan_folder, f"{scan_id}-t2w.jpg")

    paths = [f_t1c, f_t1n, f_t2f, f_t2w]
    
    if not all(os.path.exists(p) for p in paths):
        print(f"Error: One or more modality files missing in {scan_folder}")
        return

    try:
        images = [load_image(p) for p in paths]
        input_stack = np.stack(images, axis=-1) 
        input_batch = np.expand_dims(input_stack, axis=0) 
        
        print("  > Running Inference...")
        pred = model.predict(input_batch, verbose=0)
        pred_mask_raw = np.argmax(pred[0], axis=-1)
        
        # Apply Post-Processing to single prediction
        pred_mask_cleaned = utils.clean_segmentation_mask(pred_mask_raw)

        out_path = os.path.join(config.RESULTS_DIR, f"{scan_id}_segmentation_pred.png")
        vis_mask = (pred_mask_cleaned * (255 // (config.NUM_CLASSES - 1))).astype(np.uint8)
        
        cv2.imwrite(out_path, vis_mask)
        print(f"    Saved Segmentation: {out_path}")
        
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_dir", type=str, required=True, help="Path to the folder of the single scan")
    args = parser.parse_args()
    
    predict_single_scan(args.scan_dir)