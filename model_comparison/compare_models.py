import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm
from matplotlib.colors import ListedColormap, BoundaryNorm

# Import configurations and utils
from config import config
from data import get_scan_ids
from data_preprocess import process_path, load_image, load_mask
from losses import combined_loss, dice_coef
import utils
from logger import get_logger

# Import Custom Layers from the uploaded model files
from att_model import ChannelAttention, SpatialAttention
from laplacian_model import LaplacianLayer

log = get_logger("COMPARE")

# -- CONFIGURATION FOR PLOTS --
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300
})

def load_models():
    """
    Loads all three trained models with their specific custom objects.
    """
    models = {}
    
    # 1. Simple U-Net
    path_unet = os.path.join(config.CHECKPOINT_DIR, "unet_brats_best.keras")
    if os.path.exists(path_unet):
        log.info(f"Loading Simple U-Net from {path_unet}...")
        models["Simple U-Net"] = tf.keras.models.load_model(
            path_unet, 
            custom_objects={"combined_loss": combined_loss, "dice_coef": dice_coef}
        )
    else:
        log.warning("Simple U-Net checkpoint not found.")

    # 2. Dual Attention U-Net
    path_att = os.path.join(config.CHECKPOINT_DIR, "att_unet_brats_best.keras")
    if os.path.exists(path_att):
        log.info(f"Loading Dual Attention U-Net from {path_att}...")
        models["Dual Attention U-Net"] = tf.keras.models.load_model(
            path_att, 
            custom_objects={
                "combined_loss": combined_loss, 
                "dice_coef": dice_coef,
                "ChannelAttention": ChannelAttention,
                "SpatialAttention": SpatialAttention
            }
        )
    else:
        log.warning("Dual Attention U-Net checkpoint not found.")

    # 3. Laplacian U-Net
    path_lap = os.path.join(config.CHECKPOINT_DIR, "laplacian_unet_best.keras")
    if os.path.exists(path_lap):
        log.info(f"Loading Laplacian U-Net from {path_lap}...")
        models["Laplacian U-Net"] = tf.keras.models.load_model(
            path_lap, 
            custom_objects={
                "combined_loss": combined_loss, 
                "dice_coef": dice_coef,
                "LaplacianLayer": LaplacianLayer
            }
        )
    else:
        log.warning("Laplacian U-Net checkpoint not found.")
        
    return models

def evaluate_models(models, test_ids, num_patients=5):
    """
    Runs evaluation on all models.
    1. Calculates aggregate metrics.
    2. Measures inference time.
    3. Generates visual comparisons for specific patients.
    """
    
    overall_metrics = []
    inference_times = {name: [] for name in models.keys()}
    
    # We will pick the first 'num_patients' for visual comparison
    visual_ids = test_ids[:num_patients]
    
    log.info(f"Starting evaluation on {len(test_ids)} patients...")
    
    for scan_id in tqdm(test_ids, desc="Evaluating"):
        try:
            paths = process_path(scan_id, config.TEST_DIR)
            mod_paths = paths[:4] # t1c, t1n, t2f, t2w
            seg_path = paths[4]
            gt_mask = load_mask(seg_path) # (256, 256)
            
            # Prepare inputs for this patient (4 modalities)
            # Shape: (4, 256, 256, 1) to process as a batch
            patient_imgs = []
            for p in mod_paths:
                img = load_image(p)
                patient_imgs.append(np.expand_dims(img, axis=-1))
            
            input_batch = np.array(patient_imgs) # (4, 256, 256, 1)
            
            # -- Run Inference for each model --
            predictions = {}
            
            for model_name, model in models.items():
                # Measure Time
                start_time = time.time()
                preds = model.predict(input_batch, verbose=0) # (4, 256, 256, 4)
                end_time = time.time()
                
                # Record time per sample (batch size 4) -> avg time per slice
                inference_times[model_name].append((end_time - start_time) / 4.0)
                
                # Convert to integer mask
                preds_int = np.argmax(preds, axis=-1) # (4, 256, 256)
                predictions[model_name] = preds_int
                
                # Calculate metrics (Averaged over the 4 modalities for this patient)
                for i in range(4): # Loop modalities
                    m = utils.calculate_metrics_per_class(gt_mask, preds_int[i], config.NUM_CLASSES)
                    
                    # Flatten metrics for DataFrame
                    # We store one row per modality per patient per model
                    # But for summary, we can average immediately per class
                    for c_id, c_metrics in m.items():
                        row = {
                            "Model": model_name,
                            "Patient": scan_id,
                            "Class": config.CLASS_NAMES[c_id],
                            **c_metrics
                        }
                        overall_metrics.append(row)
            
            # -- Visual Comparison for selected patients --
            if scan_id in visual_ids:
                save_visual_comparison(scan_id, input_batch, gt_mask, predictions, models.keys())
                
        except Exception as e:
            log.error(f"Error processing {scan_id}: {e}")
            continue

    return pd.DataFrame(overall_metrics), inference_times

def save_visual_comparison(scan_id, input_batch, gt_mask, predictions, model_names):
    """
    Generates a Multi-Row plot:
    Rows: Modalities (T1c, T1n, T2f, T2w)
    Cols: Input | GT | Model 1 | Model 2 | Model 3
    """
    modality_labels = ["T1c", "T1n", "T2f", "T2w"]
    model_names_list = list(model_names)
    
    cols = 2 + len(model_names_list) # Input + GT + N models
    rows = 4 # 4 Modalities
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    fig.suptitle(f"Model Comparison - Patient: {scan_id}", fontsize=16, fontweight='bold', fontfamily='serif')
    
    # Colors
    mask_colors = ['black', '#377eb8', '#4daf4a', '#e41a1c']
    cmap_mask = ListedColormap(mask_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap_mask.N)
    
    for i in range(rows): # Modalities
        # 1. Input
        img = input_batch[i, :, :, 0]
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_ylabel(modality_labels[i], fontsize=12, fontweight='bold')
        if i == 0: axes[i, 0].set_title("Input", fontweight='bold')
        
        # 2. GT
        axes[i, 1].imshow(gt_mask, cmap=cmap_mask, norm=norm)
        if i == 0: axes[i, 1].set_title("Ground Truth", fontweight='bold')
        
        # 3. Models
        for j, m_name in enumerate(model_names_list):
            pred_mask = predictions[m_name][i] # Get prediction for this modality
            axes[i, 2+j].imshow(pred_mask, cmap=cmap_mask, norm=norm)
            if i == 0: axes[i, 2+j].set_title(m_name, fontweight='bold')

    # Cleanup
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Legend
    patches = [plt.Rectangle((0,0),1,1, color=mask_colors[k]) for k in range(4)]
    fig.legend(patches, config.CLASS_NAMES, loc='lower center', ncol=4, fontsize=12, frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08)
    
    save_path = os.path.join(config.PLOTS_DIR, f"comparison_{scan_id}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    log.info(f"Saved comparison plot: {save_path}")

def plot_performance_metrics(df):
    """
    Plots grouped bar charts comparing the models on key metrics (Dice, IoU, Hausdorff-proxy etc).
    """
    metrics_to_plot = ["Dice", "Jaccard", "Sensitivity", "Specificity", "F1-Score"]
    
    # Average across patients, keep Model and Class
    df_avg = df.groupby(["Model", "Class"])[metrics_to_plot].mean().reset_index()
    
    # Plot 1: Dice comparison per class
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_avg, x="Class", y="Dice", hue="Model", palette="viridis")
    plt.title("Dice Similarity Coefficient by Class", fontweight='bold')
    plt.ylim(0, 1.0)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(config.PLOTS_DIR, "comparison_dice.png"), dpi=300)
    plt.close()

    # Plot 2: Jaccard comparison per class
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_avg, x="Class", y="Jaccard", hue="Model", palette="magma")
    plt.title("Jaccard Index (IoU) by Class", fontweight='bold')
    plt.ylim(0, 1.0)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(config.PLOTS_DIR, "comparison_jaccard.png"), dpi=300)
    plt.close()
    
    return df_avg

def save_summary_report(df_metrics, inference_times):
    """
    Saves a text and CSV report.
    """
    # 1. Metrics Summary
    summary_csv = os.path.join(config.RESULTS_DIR, "final_model_comparison.csv")
    df_metrics.to_csv(summary_csv, index=False)
    
    # 2. Time Summary
    gpu_name = tf.test.gpu_device_name()
    
    txt_path = os.path.join(config.RESULTS_DIR, "final_report.txt")
    with open(txt_path, "w") as f:
        f.write("==================================================\n")
        f.write("       FINAL MODEL COMPARISON REPORT              \n")
        f.write("==================================================\n\n")
        f.write(f"GPU Device: {gpu_name if gpu_name else 'CPU'}\n\n")
        
        f.write("--- INFERENCE SPEED (Avg Time per Slice) ---\n")
        for model, times in inference_times.items():
            avg_time = np.mean(times) * 1000 # to ms
            f.write(f"{model}: {avg_time:.2f} ms\n")
        f.write("\n")
        
        f.write("--- METRICS (Averaged over Test Set) ---\n")
        f.write(df_metrics.to_string())
        
    log.info(f"Report saved to {txt_path}")

def main():
    log.info("Phase 4: Multi-Model Comparison Started")
    
    # 1. Load Data IDs
    test_ids = get_scan_ids(config.TEST_LOG_FILE, config.TEST_DIR)
    if not test_ids:
        log.error("No test data found.")
        return
        
    # 2. Load Models
    models = load_models()
    if not models:
        log.error("No models loaded.")
        return
        
    # 3. Evaluate
    # We evaluate all test_ids for metrics, but visualize only first 5
    df_results, inf_times = evaluate_models(models, test_ids, num_patients=5)
    
    # 4. Generate Reports & Plots
    if not df_results.empty:
        df_avg = plot_performance_metrics(df_results)
        save_summary_report(df_avg, inf_times)
        
    log.info("Comparison Complete. Check outputs/ folder.")

if __name__ == "__main__":
    main()