import tensorflow as tf
import numpy as np
import cv2
import os
import argparse
from config import config
from data_preprocess import load_image
from losses import combined_loss, dice_coef
# Import Custom Layer for Loading
from model import LaplacianLayer, MeanEnabledBlock 
# Note: MeanEnabledBlock is a Layer now if we saved it as such, but in my implementation 
# I used functional API calls for pooling, so only LaplacianLayer needs explicit registration 
# if it was saved as a layer. The MeanEnabled logic was functional. 
# Wait, I defined LaplacianLayer as a class. I defined MeanEnabledBlock logic inside functions 
# mostly, but let's check model.py carefully.
# Ah, I defined class MeanEnabledBlock(Layer) but didn't use it in build_unet, 
# I used functional calls. 
# Correction: I will register LaplacianLayer.

def predict_single_scan(scan_folder):
    # Updated model path for Phase 3
    model_path = os.path.join(config.CHECKPOINT_DIR, "laplacian_unet_best.keras")
    
    custom_objects = {
        "combined_loss": combined_loss, 
        "dice_coef": dice_coef,
        "LaplacianLayer": LaplacianLayer
    }
    
    if not os.path.exists(model_path):
        print("Model not found. Please train Phase 3 first.")
        return

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

    for mod_name, file_name in modality_files.items():
        full_path = os.path.join(scan_folder, file_name)
        if not os.path.exists(full_path):
            print(f"Skipping {mod_name}: File not found")
            continue

        print(f"  > Predicting {mod_name}...")
        img = load_image(full_path) 
        img_input = np.expand_dims(img, axis=-1) 
        img_batch = np.expand_dims(img_input, axis=0) 

        pred = model.predict(img_batch, verbose=0)
        pred_mask = np.argmax(pred[0], axis=-1)

        out_path = os.path.join(config.RESULTS_DIR, f"{scan_id}_{mod_name}_lap_pred.png")
        vis_mask = (pred_mask * (255 // (config.NUM_CLASSES - 1))).astype(np.uint8)
        
        cv2.imwrite(out_path, vis_mask)
        print(f"    Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_dir", type=str, required=True, help="Path to the folder of the single scan")
    args = parser.parse_args()
    
    predict_single_scan(args.scan_dir)