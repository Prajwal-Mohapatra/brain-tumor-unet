import tensorflow as tf
import numpy as np
import cv2
import os
import argparse
from config import config
from data_preprocess import load_image
from losses import combined_loss, dice_coef

def predict_single_scan(scan_folder):
    """
    Predicts mask for a specific scan folder containing the 4 modality images.
    """
    model_path = os.path.join(config.CHECKPOINT_DIR, "unet_brats_best.keras")
    custom_objects = {"combined_loss": combined_loss, "dice_coef": dice_coef}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    scan_id = os.path.basename(scan_folder)
    print(f"Processing {scan_id}...")

    # Load 4 modalities
    # Assumes specific naming convention in folder: ID-modality.jpg
    ext = ".jpg"
    try:
        t1c = load_image(os.path.join(scan_folder, f"{scan_id}-t1c{ext}"))
        t1n = load_image(os.path.join(scan_folder, f"{scan_id}-t1n{ext}"))
        t2f = load_image(os.path.join(scan_folder, f"{scan_id}-t2f{ext}"))
        t2w = load_image(os.path.join(scan_folder, f"{scan_id}-t2w{ext}"))
    except FileNotFoundError as e:
        print(e)
        return

    # Stack
    img_stack = np.stack([t1c, t1n, t2f, t2w], axis=-1)
    img_stack = np.expand_dims(img_stack, axis=0) # Add batch dim -> (1, 240, 240, 4)

    # Predict
    pred = model.predict(img_stack)
    pred_mask = np.argmax(pred[0], axis=-1) # (240, 240)

    # Save Output
    out_path = os.path.join(config.RESULTS_DIR, f"{scan_id}_pred.png")
    
    # Normalize for saving (0,1,2,3) -> visible range (0, 85, 170, 255) for visualization
    vis_mask = (pred_mask * (255 // (config.NUM_CLASSES - 1))).astype(np.uint8)
    
    cv2.imwrite(out_path, vis_mask)
    print(f"Saved prediction to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_dir", type=str, required=True, help="Path to the folder of the single scan")
    args = parser.parse_args()
    
    predict_single_scan(args.scan_dir)