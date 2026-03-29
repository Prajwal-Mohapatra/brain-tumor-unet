import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from config import config
from data import get_scan_ids
from data_preprocess import process_path, load_image, load_mask
from losses import combined_loss, dice_coef
from logger import get_logger
import utils

log = get_logger("EVAL")

# -- TARGET PATIENTS FOR VISUALIZATION --
TARGET_PATIENTS = [
    "BraTS-GLI-00006-000", 
    "BraTS-GLI-00009-001", 
    "BraTS-GLI-00011-000", 
    "BraTS-GLI-00018-000", 
    "BraTS-GLI-00026-000"
]

def evaluate():
    log.info("--------------------------------------------------")
    log.info("           EVALUATION DEEP U-NET                  ")
    log.info("--------------------------------------------------")

    # UPDATED MODEL PATH
    model_path = os.path.join(config.CHECKPOINT_DIR, "u-net_brats_best.keras")
    if not os.path.exists(model_path):
        log.error("Model not found. Please train the Deep U-Net first.")
        return

    custom_objects = {"combined_loss": combined_loss, "dice_coef": dice_coef}
    log.info(f"Loading model from {model_path}...")
    
    log.info(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    test_ids = get_scan_ids(config.TEST_LOG_FILE, config.TEST_DIR)
    
    if not test_ids:
        log.error(f"No data found in {config.TEST_DIR}")
        return

    log.info(f"Found {len(test_ids)} patients. Processing...")
    
    batch_metrics_list = []
    
    # Initialize a master confusion matrix for all pixels across all test images
    total_cm = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES))
    
    for scan_id in tqdm(test_ids, desc="Evaluating"):
        try:
            t1c_p, t1n_p, t2f_p, t2w_p, seg_path = process_path(scan_id, config.TEST_DIR)
            gt_mask = load_mask(seg_path)
            
            img_t1c = load_image(t1c_p)
            img_t1n = load_image(t1n_p)
            img_t2f = load_image(t2f_p)
            img_t2w = load_image(t2w_p)
            
            input_stack = np.stack([img_t1c, img_t1n, img_t2f, img_t2w], axis=-1)
            input_batch = np.expand_dims(input_stack, axis=0)
            
            # Predict
            preds = model.predict(input_batch, verbose=0)
            y_pred_int = np.argmax(preds[0], axis=-1)
            
            # Post-Processing Cleanup
            y_pred_cleaned = utils.clean_segmentation_mask(y_pred_int)
            
            # Accumulate Confusion Matrix (flattening 2D masks into 1D arrays)
            y_true_flat = gt_mask.flatten()
            y_pred_flat = y_pred_cleaned.flatten()
            
            cm = confusion_matrix(y_true_flat, y_pred_flat, labels=range(config.NUM_CLASSES))
            total_cm += cm
            
            # Calculate general metrics
            m = utils.calculate_metrics_per_class(gt_mask, y_pred_cleaned, config.NUM_CLASSES)
            batch_metrics_list.append(m)
            
            if scan_id in TARGET_PATIENTS:
                save_name = f"{scan_id}_u-net_cleaned.png"
                utils.visualize_inference(
                    input_stack, 
                    gt_mask,
                    y_pred_cleaned,
                    title_suffix=f"({scan_id} - U-Net)",
                    save_name=save_name
                )
                    
        except Exception as e:
            log.error(f"Error evaluating {scan_id}: {e}")
            continue

    log.info("Aggregating metrics...")
    df_metrics = utils.save_metrics_to_csv(batch_metrics_list)
    
    if df_metrics.empty:
        log.error("Evaluation produced no metrics.")
        return

    log.info("Generating Metrics and Confusion Matrix Plots...")
    utils.plot_metrics_summary(df_metrics)
    
    # Plot the accumulated confusion matrix
    utils.plot_confusion_matrix(total_cm)
    
    print("\n" + "="*50)
    print("U-Net)")
    print("="*50)
    print(df_metrics.to_string(index=False))
    print("="*50 + "\n")
    
    log.info(f"Evaluation Complete. Check {config.OUTPUT_ROOT} for artifacts.")

if __name__ == "__main__":
    evaluate()