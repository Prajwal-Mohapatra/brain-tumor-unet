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
    Predicts masks for a specific scan folder.
    Process: Iterates t1c, t1n, t2f, t2w individually.
    """
    model_path = os.path.join(config.CHECKPOINT_DIR, "unet_brats_best.keras")
    custom_objects = {"combined_loss": combined_loss, "dice_coef": dice_coef}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    scan_id = os.path.basename(scan_folder)
    print(f"Processing Patient: {scan_id}...")
    
    ext = ".jpg"
    modality_files = {
        "t1c": f"{scan_id}-t1c{ext}",
        "t1n": f"{scan_id}-t1n{ext}",
        "t2f": f"{scan_id}-t2f{ext}",
        "t2w": f"{scan_id}-t2w{ext}"
    }

    # Iterate over modalities
    for mod_name, file_name in modality_files.items():
        full_path = os.path.join(scan_folder, file_name)
        
        if not os.path.exists(full_path):
            print(f"Skipping {mod_name}: File not found ({full_path})")
            continue

        print(f"  > Predicting {mod_name}...")
        
        # Load & Preprocess
        img = load_image(full_path) # (256, 256)
        img_input = np.expand_dims(img, axis=-1) # (256, 256, 1)
        img_batch = np.expand_dims(img_input, axis=0) # (1, 256, 256, 1)

        # Predict
        pred = model.predict(img_batch, verbose=0)
        pred_mask = np.argmax(pred[0], axis=-1) # (256, 256)

        # Save Output
        out_path = os.path.join(config.RESULTS_DIR, f"{scan_id}_{mod_name}_pred.png")
        
        # Normalize for visualization (0,1,2,3 -> 0,85,170,255)
        vis_mask = (pred_mask * (255 // (config.NUM_CLASSES - 1))).astype(np.uint8)
        
        cv2.imwrite(out_path, vis_mask)
        print(f"    Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_dir", type=str, required=True, help="Path to the folder of the single scan")
    args = parser.parse_args()
    
    predict_single_scan(args.scan_dir)