import tensorflow as tf
import os
from config import config
from data import get_train_val_datasets
from model import build_unet
from losses import combined_loss, dice_coef
from logger import get_logger

# Initialize Logger
log = get_logger("TRAIN")

def main():
    # 1. Setup Data
    log.info("Loading Data...")
    try:
        train_ds, val_ds, n_train, n_val = get_train_val_datasets()
    except Exception as e:
        log.error(f"Failed to load data: {e}")
        return
    
    # 2. Setup Model
    log.info("Building Model...")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_unet()
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
        
        model.compile(
            optimizer=optimizer,
            loss=combined_loss,
            metrics=['accuracy', dice_coef]
        )

    # 3. Callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(config.CHECKPOINT_DIR, "unet_brats_best.keras"),
        save_best_only=True,
        monitor='val_dice_coef',
        mode='max',
        verbose=1
    )
    
    # UPDATED: Patience set to 15 via config
    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_dice_coef',
        patience=config.EARLY_STOPPING_PATIENCE,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )
    
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=config.LOGS_DIR)
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(config.LOGS_DIR, "training_log.csv"))

    # 4. Train
    log.info(f"Starting training on {n_train} samples, validating on {n_val} samples...")
    
    history = model.fit(
        train_ds,
        epochs=config.EPOCHS,
        validation_data=val_ds,
        callbacks=[checkpoint_cb, early_stop_cb, tensorboard_cb, csv_logger]
    )
    
    log.info("Training Complete.")
    
    # Save final model
    save_path = os.path.join(config.CHECKPOINT_DIR, "unet_brats_final.keras")
    model.save(save_path)
    log.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()