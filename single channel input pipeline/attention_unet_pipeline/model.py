import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D, 
                                     Concatenate, Dropout, BatchNormalization, 
                                     Activation, GlobalAveragePooling2D, Reshape, 
                                     Dense, Multiply, Add, Layer)
from tensorflow.keras.models import Model
from config import config

# --- Custom Layers for Serialization Safety ---

class ChannelAttention(Layer):
    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        filters = input_shape[-1]
        self.se_sq = GlobalAveragePooling2D()
        self.se_reshape = Reshape((1, 1, filters))
        self.se_ex1 = Dense(filters // self.ratio, activation='relu', use_bias=False)
        self.se_ex2 = Dense(filters, activation='sigmoid', use_bias=False)
        self.multiply = Multiply()
        super(ChannelAttention, self).build(input_shape)

    def call(self, input_tensor):
        x = self.se_sq(input_tensor)
        x = self.se_reshape(x)
        x = self.se_ex1(x)
        x = self.se_ex2(x)
        return self.multiply([input_tensor, x])

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({"ratio": self.ratio})
        return config

class SpatialAttention(Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
    
    def build(self, input_shape):
        self.conv = Conv2D(1, (self.kernel_size, self.kernel_size), padding='same', activation='sigmoid')
        self.multiply = Multiply()
        self.concat = Concatenate(axis=-1)
        super(SpatialAttention, self).build(input_shape)

    def call(self, input_tensor):
        # TensorFlow operations inside call are safe in Custom Layers
        avg_pool = tf.reduce_mean(input_tensor, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(input_tensor, axis=-1, keepdims=True)
        x = self.concat([avg_pool, max_pool])
        x = self.conv(x)
        return self.multiply([input_tensor, x])

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({"kernel_size": self.kernel_size})
        return config

# --- Standard U-Net Blocks ---

def conv_block(input_tensor, num_filters):
    x = Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def encoder_block(input_tensor, num_filters, dropout_rate=0.0):
    x = conv_block(input_tensor, num_filters)
    p = MaxPooling2D((2, 2))(x)
    if dropout_rate > 0:
        p = Dropout(dropout_rate)(p)
    return x, p

def dual_attention_block(input_tensor):
    """
    Combines Custom Channel and Spatial Attention Layers.
    """
    ca = ChannelAttention()(input_tensor)
    sa = SpatialAttention()(input_tensor)
    x = Add()([input_tensor, ca, sa])
    return x

def decoder_block(input_tensor, skip_tensor, num_filters, dropout_rate=0.0, use_attention=True):
    x = UpSampling2D((2, 2))(input_tensor)
    
    if use_attention:
        skip_tensor = dual_attention_block(skip_tensor)
        
    x = Concatenate()([x, skip_tensor])
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    x = conv_block(x, num_filters)
    return x

def build_unet():
    inputs = Input((config.IMG_HEIGHT, config.IMG_WIDTH, config.NUM_CHANNELS))
    
    # -- ENCODER --
    c1, p1 = encoder_block(inputs, config.FILTERS, dropout_rate=0.0)
    c2, p2 = encoder_block(p1, config.FILTERS * 2, dropout_rate=0.1)
    c3, p3 = encoder_block(p2, config.FILTERS * 4, dropout_rate=0.2)
    c4, p4 = encoder_block(p3, config.FILTERS * 8, dropout_rate=0.2)
    c5, p5 = encoder_block(p4, config.FILTERS * 16, dropout_rate=0.3)
    
    # -- BRIDGE --
    b = conv_block(p5, config.FILTERS * 32)
    
    # -- DECODER --
    d5 = decoder_block(b, c5, config.FILTERS * 16, dropout_rate=0.3, use_attention=True)
    d4 = decoder_block(d5, c4, config.FILTERS * 8, dropout_rate=0.2, use_attention=True)
    d3 = decoder_block(d4, c3, config.FILTERS * 4, dropout_rate=0.2, use_attention=True)
    d2 = decoder_block(d3, c2, config.FILTERS * 2, dropout_rate=0.1, use_attention=True)
    d1 = decoder_block(d2, c1, config.FILTERS, dropout_rate=0.0, use_attention=True)
    
    # -- OUTPUT --
    outputs = Conv2D(config.NUM_CLASSES, (1, 1), activation='softmax')(d1)
    
    model = Model(inputs=[inputs], outputs=[outputs], name="DualAttention_UNet")
    return model

if __name__ == "__main__":
    model = build_unet()
    model.summary()