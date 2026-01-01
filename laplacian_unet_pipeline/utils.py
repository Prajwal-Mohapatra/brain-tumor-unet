import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from matplotlib.colors import ListedColormap, BoundaryNorm
from config import config

def plot_training_history(history):
    """
    Plots Loss, Dice, Jaccard (IoU), and Accuracy over epochs.
    Saves to outputs/plots/training_history.png
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = [
        ('loss', 'val_loss', 'Combined Loss'),
        ('dice_coef', 'val_dice_coef', 'Dice Coefficient'),
        ('iou', 'val_iou', 'Jaccard Index (IoU)'),
        ('accuracy', 'val_accuracy', 'Pixel Accuracy')
    ]
    
    for i, (train_m, val_m, title) in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        # Check if metric exists in history (IoU might be named differently depending on compile)
        if train_m in history.history:
            ax.plot(history.history[train_m], label='Train', color='#1f77b4', linewidth=2)
            ax.plot(history.history[val_m], label='Validation', color='#ff7f0e', linewidth=2, linestyle='--')
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Score')
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.6)
        else:
            ax.text(0.5, 0.5, f"{title} not found in history", ha='center')

    plt.tight_layout()
    save_path = os.path.join(config.PLOTS_DIR, "training_history.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training history plot saved to {save_path}")

def calculate_metrics_per_class(y_true, y_pred, num_classes):
    """
    Calculates detailed metrics for each class.
    """
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
        
        metrics[c] = {
            "Dice": dice,
            "Jaccard": jaccard,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": sensitivity, 
            "F1-Score": dice       
        }
    return metrics

def save_metrics_to_csv(metrics_list):
    """Aggregates metrics and saves to CSV."""
    final_data = []
    for c_idx, c_name in enumerate(config.CLASS_NAMES):
        class_metrics = [m[c_idx] for m in metrics_list]
        avg_metrics = {}
        for key in class_metrics[0].keys():
            avg_metrics[key] = np.mean([item[key] for item in class_metrics])
        avg_metrics["Class"] = c_name
        final_data.append(avg_metrics)
    
    df = pd.DataFrame(final_data)
    cols = ["Class", "Dice", "Jaccard", "Sensitivity", "Specificity", "Accuracy", "Precision", "Recall", "F1-Score"]
    df = df[cols]
    
    csv_path = os.path.join(config.RESULTS_DIR, "test_metrics.csv")
    df.to_csv(csv_path, index=False)
    return df

def plot_metrics_summary(df):
    """Plots bar chart of metrics."""
    df_melt = df.melt(id_vars="Class", var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_melt, x="Metric", y="Score", hue="Class", palette="viridis")
    plt.title("Evaluation Metrics by Class", fontweight='bold')
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(config.PLOTS_DIR, "metrics_summary.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def visualize_inference(x_img, y_true, y_pred, title_suffix="", save_name="inference.png"):
    """
    Visualizes Single Modality Input | Ground Truth | Prediction | Overlay
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(f'Single Modality Analysis {title_suffix}', fontsize=16, fontweight='bold', fontfamily='serif')
    
    # Colors: BG (Black), NCR (Blue), ED (Green), ET (Red)
    mask_colors = ['black', '#377eb8', '#4daf4a', '#e41a1c']
    cmap_mask = ListedColormap(mask_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap_mask.N)
    
    # 1. Input Image (Handle (H,W,1) or (H,W))
    if x_img.ndim == 3:
        x_show = x_img[:, :, 0]
    else:
        x_show = x_img
        
    axes[0].imshow(x_show, cmap='gray')
    axes[0].set_title("Input Modality", fontweight='bold')
    
    # 2. Ground Truth
    axes[1].imshow(y_true, cmap=cmap_mask, norm=norm)
    axes[1].set_title("Ground Truth", fontweight='bold')
    
    # 3. Prediction
    axes[2].imshow(y_pred, cmap=cmap_mask, norm=norm)
    axes[2].set_title("Prediction", fontweight='bold')
    
    # 4. Overlay
    axes[3].imshow(x_show, cmap='gray')
    axes[3].imshow(y_pred, cmap=cmap_mask, norm=norm, alpha=0.4)
    axes[3].set_title("Overlay", fontweight='bold')
    
    # Remove ticks
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    # Legend
    patches = [plt.Rectangle((0,0),1,1, color=mask_colors[i]) for i in range(4)]
    fig.legend(patches, config.CLASS_NAMES, loc='lower center', ncol=4, fontsize=12, frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    
    save_path = os.path.join(config.PLOTS_DIR, save_name)
    plt.savefig(save_path, dpi=300)
    plt.close()