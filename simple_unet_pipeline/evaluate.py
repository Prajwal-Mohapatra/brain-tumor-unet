import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from config import config
from data import get_scan_ids
from data_preprocess import process_path, load_image, load_mask
from losses import combined_loss, dice_coef
from logger import get_logger
import utils

log = get_logger("EVAL")

def evaluate():
    log.info("--------------------------------------------------")
    log.info("             STARTING EVALUATION                  ")
    log.info("--------------------------------------------------")

    # 1. Load Model
    model_path = os.path.join(config.CHECKPOINT_DIR, "unet_brats_best.keras")
    if not os.path.exists(model_path):
        log.error("Model not found. Please train first.")
        return

    custom_objects = {"combined_loss": combined_loss, "dice_coef": dice_coef}
    log.info(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    # 2. Get Data IDs Manually to track Patient ID
    print(f"Looking for Test data in: {config.TEST_DIR}")
    test_ids = get_scan_ids(config.TEST_LOG_FILE, config.TEST_DIR)
    
    if not test_ids:
        log.error(f"No data found in {config.TEST_DIR}")
        return

    log.info(f"Found {len(test_ids)} patients. Will process 4 modalities per patient.")
    
    # 3. Iterate Patients -> Modalities
    batch_metrics_list = []
    
    modality_names = ['t1c', 't1n', 't2f', 't2w']
    
    for scan_id in tqdm(test_ids, desc="Evaluating Patients"):
        try:
            # Get paths
            paths = process_path(scan_id, config.TEST_DIR)
            # Paths tuple: t1c, t1n, t2f, t2w, seg
            mod_paths = paths[:4]
            seg_path = paths[4]
            
            # Load GT Mask (Once per patient)
            gt_mask = load_mask(seg_path) # (H, W)
            
            # Iterate 4 Modalities
            for i, p_mod in enumerate(mod_paths):
                mod_name = modality_names[i]
                
                # Load Image -> (H, W) -> (1, H, W, 1)
                img = load_image(p_mod)
                input_tensor = np.expand_dims(img, axis=-1)
                input_batch = np.expand_dims(input_tensor, axis=0)
                
                # Predict
                preds = model.predict(input_batch, verbose=0) # (1, H, W, 4)
                
                # Convert to Class Map
                y_pred_int = np.argmax(preds[0], axis=-1) # (H, W)
                
                # Calculate Metrics
                m = utils.calculate_metrics_per_class(
                    gt_mask, 
                    y_pred_int, 
                    config.NUM_CLASSES
                )
                batch_metrics_list.append(m)
                
                # Save Visualization (For the first 5 patients only to save space)
                if len(batch_metrics_list) <= (5 * 4): 
                    save_name = f"{scan_id}_{mod_name}_pred.png"
                    utils.visualize_inference(
                        input_tensor,
                        gt_mask,
                        y_pred_int,
                        title_suffix=f"({scan_id} - {mod_name})",
                        save_name=save_name
                    )
                    
        except Exception as e:
            log.error(f"Error evaluating {scan_id}: {e}")
            continue

    # 4. Aggregation and Saving
    log.info("Aggregating metrics...")
    df_metrics = utils.save_metrics_to_csv(batch_metrics_list)
    
    log.info("Generating Metrics Plot...")
    utils.plot_metrics_summary(df_metrics)
    
    # Print Summary to Console
    print("\n" + "="*50)
    print("FINAL TEST METRICS (Average over all modalities)")
    print("="*50)
    print(df_metrics.to_string(index=False))
    print("="*50 + "\n")
    
    log.info(f"Evaluation Complete. Check {config.OUTPUT_ROOT} for artifacts.")

if __name__ == "__main__":
    evaluate()