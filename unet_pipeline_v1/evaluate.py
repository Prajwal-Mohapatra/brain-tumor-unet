import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from config import config
from data import get_test_dataset
from losses import combined_loss, dice_coef
from logger import get_logger

log = get_logger("EVAL")

def evaluate():
    # Load model
    model_path = os.path.join(config.CHECKPOINT_DIR, "unet_brats_best.keras")
    if not os.path.exists(model_path):
        log.error("Model not found. Please train first.")
        return

    # Helper for loading custom objects
    custom_objects = {"combined_loss": combined_loss, "dice_coef": dice_coef}
    log.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    # Get TEST Dataset
    try:
        test_ds, n_test = get_test_dataset()
    except Exception as e:
        log.error(e)
        return
    
    log.info(f"Evaluating on TEST set ({n_test} samples)...")
    results = model.evaluate(test_ds)
    
    log.info("Cannot reliably map metrics to names without dict, printing raw:")
    log.info(results)
    
    # If the model.evaluate returns a list, usually [loss, accuracy, dice_coef] based on compilation
    if len(results) >= 3:
        log.info(f"Test Loss: {results[0]:.4f}")
        log.info(f"Test Accuracy: {results[1]:.4f}")
        log.info(f"Test Dice Coef: {results[2]:.4f}")
    
    # Visual check on one batch from Test Set
    for x_batch, y_batch in test_ds.take(1):
        preds = model.predict(x_batch)
        
        # Plot first 3 images in batch
        fig, axes = plt.subplots(3, 4, figsize=(15, 10))
        for i in range(min(3, len(x_batch))):
            # Input (T1c channel)
            axes[i, 0].imshow(x_batch[i, :, :, 0], cmap='gray')
            axes[i, 0].set_title(f"Test Input T1c")
            axes[i, 0].axis('off')
            
            # Ground Truth
            axes[i, 1].imshow(np.argmax(y_batch[i], axis=-1), cmap='jet')
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')
            
            # Prediction
            axes[i, 2].imshow(np.argmax(preds[i], axis=-1), cmap='jet')
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis('off')

            # Prediction Confidence
            axes[i, 3].imshow(np.max(preds[i], axis=-1), cmap='hot')
            axes[i, 3].set_title("Confidence")
            axes[i, 3].axis('off')
            
        plt.tight_layout()
        save_path = os.path.join(config.RESULTS_DIR, "test_set_preview.png")
        plt.savefig(save_path)
        log.info(f"Saved test preview to {save_path}")

if __name__ == "__main__":
    evaluate()