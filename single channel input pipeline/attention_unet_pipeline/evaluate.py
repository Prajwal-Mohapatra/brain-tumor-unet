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
# Import Custom Layers
from model import ChannelAttention, SpatialAttention

log = get_logger("EVAL")

def evaluate():
    log.info("--------------------------------------------------")
    log.info("        PHASE 2: DUAL ATTENTION EVALUATION        ")
    log.info("--------------------------------------------------")

    model_path = os.path.join(config.CHECKPOINT_DIR, "att_unet_brats_best.keras")
    if not os.path.exists(model_path):
        log.error("Model not found. Please train Phase 2 first.")
        return

    # Register Custom Layers here so Keras knows what they are
    custom_objects = {
        "combined_loss": combined_loss, 
        "dice_coef": dice_coef,
        "ChannelAttention": ChannelAttention,
        "SpatialAttention": SpatialAttention
    }
    
    log.info(f"Loading model from {model_path}...")
    # Safe loading now possible
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    print(f"Looking for Test data in: {config.TEST_DIR}")
    test_ids = get_scan_ids(config.TEST_LOG_FILE, config.TEST_DIR)
    
    if not test_ids:
        log.error(f"No data found in {config.TEST_DIR}")
        return

    log.info(f"Found {len(test_ids)} patients. Will process 4 modalities per patient.")
    
    batch_metrics_list = []
    modality_names = ['t1c', 't1n', 't2f', 't2w']
    
    for scan_id in tqdm(test_ids, desc="Evaluating Patients"):
        try:
            paths = process_path(scan_id, config.TEST_DIR)
            mod_paths = paths[:4]
            seg_path = paths[4]
            gt_mask = load_mask(seg_path)
            
            for i, p_mod in enumerate(mod_paths):
                mod_name = modality_names[i]
                img = load_image(p_mod)
                input_tensor = np.expand_dims(img, axis=-1)
                input_batch = np.expand_dims(input_tensor, axis=0)
                
                preds = model.predict(input_batch, verbose=0)
                y_pred_int = np.argmax(preds[0], axis=-1)
                
                m = utils.calculate_metrics_per_class(gt_mask, y_pred_int, config.NUM_CLASSES)
                batch_metrics_list.append(m)
                
                if len(batch_metrics_list) <= (5 * 4): 
                    save_name = f"{scan_id}_{mod_name}_att_pred.png"
                    utils.visualize_inference(
                        input_tensor,
                        gt_mask,
                        y_pred_int,
                        title_suffix=f"({scan_id} - {mod_name} - Attention)",
                        save_name=save_name
                    )
                    
        except Exception as e:
            log.error(f"Error evaluating {scan_id}: {e}")
            continue

    log.info("Aggregating metrics...")
    df_metrics = utils.save_metrics_to_csv(batch_metrics_list)
    
    log.info("Generating Metrics Plot...")
    utils.plot_metrics_summary(df_metrics)
    
    print("\n" + "="*50)
    print("FINAL PHASE 2 TEST METRICS (Dual Attention)")
    print("="*50)
    print(df_metrics.to_string(index=False))
    print("="*50 + "\n")
    
    log.info(f"Evaluation Complete. Check {config.OUTPUT_ROOT} for artifacts.")

if __name__ == "__main__":
    evaluate()