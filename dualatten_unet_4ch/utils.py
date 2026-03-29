import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from scipy import ndimage
from matplotlib.colors import ListedColormap, BoundaryNorm
from config import config

def clean_segmentation_mask(pred_mask):
    """Post-processing to remove salt-and-pepper noise."""
    cleaned_mask = pred_mask.copy()
    for c in [1, 2, 3]:
        binary_class_mask = (pred_mask == c)
        opened_mask = ndimage.binary_opening(binary_class_mask, structure=np.ones((3,3))).astype(np.int32)
        diff = (binary_class_mask ^ opened_mask)
        cleaned_mask[diff] = 0
    return cleaned_mask

def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metrics = [
        ('loss', 'val_loss', 'Loss (GDL+Focal)'),
        ('dice_coef', 'val_dice_coef', 'Dice Coefficient'),
        ('iou', 'val_iou', 'Jaccard Index (IoU)'),
        ('accuracy', 'val_accuracy', 'Pixel Accuracy')
    ]
    for i, (train_m, val_m, title) in enumerate(metrics):
        ax = axes[i//2, i%2]
        if train_m in history.history:
            ax.plot(history.history[train_m], label='Train', color='#1f77b4', linewidth=2)
            ax.plot(history.history[val_m], label='Validation', color='#ff7f0e', linewidth=2, linestyle='--')
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Score')
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    save_path = os.path.join(config.PLOTS_DIR, "training_history.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix(cm):
    """
    Plots a row-normalized confusion matrix to handle massive class imbalance.
    Shows the percentage of true class pixels assigned to each predicted class.
    """
    # Normalize by row (True labels) to get percentages
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', annot_kws={"size": 20},
                xticklabels=config.CLASS_NAMES, 
                yticklabels=config.CLASS_NAMES)
    
    plt.title('Pixel-Level Confusion Matrix (Normalized)', fontweight='bold', fontsize=20)
    plt.ylabel('Ground Truth Class', fontweight='bold', fontsize=18)
    plt.xlabel('Predicted Class', fontweight='bold', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16, rotation=90)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18) 
    plt.tight_layout()
    
    save_path = os.path.join(config.PLOTS_DIR, "confusion_matrix_new.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to {save_path}")

def calculate_hd95(y_true, y_pred, spacing=1.0):
    if np.sum(y_true) == 0 and np.sum(y_pred) == 0: return 0.0
    if np.sum(y_true) == 0 or np.sum(y_pred) == 0: return 100.0 
    d_to_true = ndimage.distance_transform_edt(1 - y_true)
    d_to_pred = ndimage.distance_transform_edt(1 - y_pred)
    dist_p_to_t = d_to_true[y_pred.astype(bool)]
    dist_t_to_p = d_to_pred[y_true.astype(bool)]
    all_dists = np.concatenate([dist_p_to_t, dist_t_to_p])
    if len(all_dists) > 0: return np.percentile(all_dists, 95)
    else: return 0.0

def calculate_metrics_per_class(y_true, y_pred, num_classes):
    metrics = {}
    for c in range(num_classes):
        p = (y_pred == c)
        t = (y_true == c)
        TP = np.sum(p & t)
        FP = np.sum(p & ~t)
        FN = np.sum(~p & t)
        TN = np.sum(~p & ~t)
        smooth = 1e-6
        sensitivity = TP / (TP + FN + smooth)
        specificity = TN / (TN + FP + smooth)
        precision = TP / (TP + FP + smooth)
        accuracy = (TP + TN) / (TP + TN + FP + FN + smooth)
        dice = (2 * TP) / (2 * TP + FP + FN + smooth)
        jaccard = TP / (TP + FP + FN + smooth)
        hd95 = calculate_hd95(t, p)
        metrics[c] = {
            "Dice": dice, "Jaccard": jaccard, "Sensitivity": sensitivity,
            "Specificity": specificity, "Accuracy": accuracy, "Precision": precision,
            "Recall": sensitivity, "F1-Score": dice, "HD95": hd95
        }
    return metrics

def save_metrics_to_csv(metrics_list):
    if not metrics_list: return pd.DataFrame()
    final_data = []
    for c_idx, c_name in enumerate(config.CLASS_NAMES):
        class_metrics = [m[c_idx] for m in metrics_list]
        avg_metrics = {}
        for key in class_metrics[0].keys():
            avg_metrics[key] = np.mean([item[key] for item in class_metrics])
        avg_metrics["Class"] = c_name
        final_data.append(avg_metrics)
    df = pd.DataFrame(final_data)
    cols = ["Class", "Dice", "Jaccard", "HD95", "Sensitivity", "Specificity", "Accuracy", "Precision", "Recall", "F1-Score"]
    df = df[cols]
    csv_path = os.path.join(config.RESULTS_DIR, "test_metrics.csv")
    df.to_csv(csv_path, index=False)
    return df

def plot_metrics_summary(df):
    if df.empty: return
    plot_cols = [c for c in df.columns if c not in ['Class', 'HD95']]
    df_melt = df.melt(id_vars="Class", value_vars=plot_cols, var_name="Metric", value_name="Score")
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_melt, x="Metric", y="Score", hue="Class", palette="viridis")
    plt.title("Evaluation Metrics by Class (Excl. HD95)", fontweight='bold')
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, "metrics_summary.png"), dpi=300)
    plt.close()
    if "HD95" in df.columns:
        plt.figure(figsize=(8, 6))
        sns.barplot(data=df, x="Class", y="HD95", palette="magma")
        plt.title("Hausdorff Distance (HD95) by Class - Lower is Better", fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIR, "hd95_summary.png"), dpi=300)
        plt.close()

def visualize_inference(x_img, y_true, y_pred, title_suffix="", save_name="inference.png"):
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(f'Single Modality Analysis {title_suffix}', fontsize=16, fontweight='bold', fontfamily='serif')
    mask_colors = ['black', '#377eb8', '#4daf4a', '#e41a1c']
    cmap_mask = ListedColormap(mask_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap_mask.N)
    if x_img.ndim == 3: x_show = x_img[:, :, 0]
    else: x_show = x_img
    axes[0].imshow(x_show, cmap='gray')
    axes[0].set_title("Input Modality", fontweight='bold')
    axes[1].imshow(y_true, cmap=cmap_mask, norm=norm)
    axes[1].set_title("Ground Truth", fontweight='bold')
    axes[2].imshow(y_pred, cmap=cmap_mask, norm=norm)
    axes[2].set_title("Prediction (Cleaned)", fontweight='bold')
    axes[3].imshow(x_show, cmap='gray')
    axes[3].imshow(y_pred, cmap=cmap_mask, norm=norm, alpha=0.4)
    axes[3].set_title("Overlay", fontweight='bold')
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    patches = [plt.Rectangle((0,0),1,1, color=mask_colors[i]) for i in range(4)]
    fig.legend(patches, config.CLASS_NAMES, loc='lower center', ncol=4, fontsize=12, frameon=True)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    plt.savefig(os.path.join(config.PLOTS_DIR, save_name), dpi=300)
    plt.close()