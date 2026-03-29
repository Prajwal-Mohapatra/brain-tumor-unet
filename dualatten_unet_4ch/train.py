import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from config import config
from data import get_train_val_datasets
from model import build_unet
from losses import combined_loss, dice_coef
from logger import get_logger
from utils import plot_training_history

log = get_logger("TRAIN")

def main():
    log.info("--------------------------------------------------")
    log.info("              DA-UNET (1024 FILTER)               ")
    log.info("--------------------------------------------------")
    
    log.info("Loading Data...")
    try:
        train_ds, val_ds, n_train, n_val = get_train_val_datasets()
        log.info(f"Training samples: {n_train} | Validation samples: {n_val}")
    except Exception as e:
        log.error(f"Failed to load data: {e}")
        return
    
    log.info("Building 4-Channel DA-UNet Model...")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_unet()
        
        # Save Model Summary
        summary_path = os.path.join(config.RESULTS_DIR, "model_summary_da_unet.txt")
        with open(summary_path, "w") as f:
            model.summary(print_fn=lambda x: f.write(x + "\n"))
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
        
        iou_metric = tf.keras.metrics.OneHotIoU(
            num_classes=config.NUM_CLASSES, 
            target_class_ids=[0, 1, 2, 3], 
            name='iou'
        )
        
        # COMPILE WITH NEW ROBUST LOSS
        model.compile(
            optimizer=optimizer,
            loss=combined_loss, 
            metrics=['accuracy', dice_coef, iou_metric]
        )

    # -- CALLBACKS --
    # UPDATED: New Save Names
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(config.CHECKPOINT_DIR, "att_unet_brats_best.keras"),
        save_best_only=True,
        monitor='val_dice_coef',
        mode='max',
        verbose=1
    )
    
    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_dice_coef',
        patience=config.EARLY_STOPPING_PATIENCE,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )
    
    lr_scheduler_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_dice_coef',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        mode='max',
        verbose=1
    )
    
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=config.LOGS_DIR)
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(config.LOGS_DIR, "training_log_da_unet.csv"))

    log.info("Starting Training...")
    
    history = model.fit(
        train_ds,
        epochs=config.EPOCHS,
        validation_data=val_ds,
        callbacks=[checkpoint_cb, early_stop_cb, lr_scheduler_cb, tensorboard_cb, csv_logger],
        verbose=1
    )
    
    log.info("Training Complete.")
    
    # UPDATED: New Save Name
    save_path = os.path.join(config.CHECKPOINT_DIR, "att_unet_brats_final.keras")
    model.save(save_path)
    
    log.info("Generating Training Plots...")
    plot_training_history(history)
    log.info("Pipeline Finished Successfully.")

if __name__ == "__main__":
    main()