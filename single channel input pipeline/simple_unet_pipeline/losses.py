import tensorflow as tf
import tensorflow.keras.backend as K

def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Dice Coefficient for multi-class.
    Calculates per class and averages.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal loss to handle class imbalance.
    """
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    cross_entropy = -y_true * K.log(y_pred)
    weight = alpha * y_true * K.pow((1 - y_pred), gamma)
    return K.mean(K.sum(weight * cross_entropy, axis=-1))

def combined_loss(y_true, y_pred):
    """
    Weighted combination of Dice and Focal Loss.
    """
    # 0.5 * Dice + 0.5 * Focal is a good starting point
    return 0.5 * dice_loss(y_true, y_pred) + 0.5 * focal_loss(y_true, y_pred)