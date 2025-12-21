import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model
from config import config

def conv_block(input_tensor, num_filters):
    """
    A block containing: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    """
    x = Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def encoder_block(input_tensor, num_filters, dropout_rate=0.0):
    """
    Encoder: ConvBlock -> MaxPool
    """
    x = conv_block(input_tensor, num_filters)
    p = MaxPooling2D((2, 2))(x)
    if dropout_rate > 0:
        p = Dropout(dropout_rate)(p)
    return x, p

def decoder_block(input_tensor, skip_tensor, num_filters, dropout_rate=0.0):
    """
    Decoder: UpSample -> Concat -> ConvBlock
    """
    x = UpSampling2D((2, 2))(input_tensor)
    x = Concatenate()([x, skip_tensor])
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    x = conv_block(x, num_filters)
    return x

def build_unet():
    """
    U-Net with 5 Encoders and 5 Decoders (Deep U-Net)
    Input shape: (240, 240, 4)
    Output shape: (240, 240, 4)
    """
    inputs = Input((config.IMG_HEIGHT, config.IMG_WIDTH, config.NUM_CHANNELS))
    
    # -- ENCODER --
    # Level 1
    c1, p1 = encoder_block(inputs, config.FILTERS, dropout_rate=0.0)
    # Level 2
    c2, p2 = encoder_block(p1, config.FILTERS * 2, dropout_rate=0.1)
    # Level 3
    c3, p3 = encoder_block(p2, config.FILTERS * 4, dropout_rate=0.2)
    # Level 4
    c4, p4 = encoder_block(p3, config.FILTERS * 8, dropout_rate=0.2)
    # Level 5
    c5, p5 = encoder_block(p4, config.FILTERS * 16, dropout_rate=0.3)
    
    # -- BRIDGE --
    b = conv_block(p5, config.FILTERS * 32)
    
    # -- DECODER --
    # Level 5 (Up from Bridge)
    d5 = decoder_block(b, c5, config.FILTERS * 16, dropout_rate=0.3)
    # Level 4
    d4 = decoder_block(d5, c4, config.FILTERS * 8, dropout_rate=0.2)
    # Level 3
    d3 = decoder_block(d4, c3, config.FILTERS * 4, dropout_rate=0.2)
    # Level 2
    d2 = decoder_block(d3, c2, config.FILTERS * 2, dropout_rate=0.1)
    # Level 1
    d1 = decoder_block(d2, c1, config.FILTERS, dropout_rate=0.0)
    
    # -- OUTPUT --
    # Softmax for multi-class segmentation (Background, NCR, ED, ET)
    outputs = Conv2D(config.NUM_CLASSES, (1, 1), activation='softmax')(d1)
    
    model = Model(inputs=[inputs], outputs=[outputs], name="UNet_5Layer_BraTS")
    return model

if __name__ == "__main__":
    model = build_unet()
    model.summary()