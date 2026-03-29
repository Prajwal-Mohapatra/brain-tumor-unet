import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D, 
                                     Concatenate, Dropout, BatchNormalization, 
                                     Activation, AveragePooling2D, Add, Layer)
from tensorflow.keras.models import Model
from config import config

# --- Custom Layers ---

class LaplacianLayer(Layer):
    """
    Computes the Laplacian of the input tensor to highlight edges.
    Uses a fixed 3x3 Laplacian kernel applied independently to EACH channel.
    """
    def __init__(self, **kwargs):
        super(LaplacianLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Dynamically get the number of input channels (e.g., 4)
        in_channels = input_shape[-1]
        
        # Fixed Laplacian Kernel
        # [[0,  1, 0],
        #  [1, -4, 1],
        #  [0,  1, 0]]
        k = tf.constant([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=tf.float32)
        
        # Reshape for depthwise conv2d: [filter_height, filter_width, in_channels, channel_multiplier]
        k = tf.reshape(k, (3, 3, 1, 1))
        
        # Tile the kernel to match the number of input channels so each gets its own edge detection
        self.kernel = tf.tile(k, [1, 1, in_channels, 1])
        super(LaplacianLayer, self).build(input_shape)

    def call(self, inputs):
        # Apply depthwise conv to treat each channel (T1c, T1n, T2f, T2w) independently
        edges = tf.nn.depthwise_conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding='SAME')
        return edges

    def get_config(self):
        return super(LaplacianLayer, self).get_config()

class MeanEnabledBlock(Layer):
    """
    A block that combines Max Pooling (Features) and Average Pooling (Mean/Context).
    """
    def __init__(self, **kwargs):
        super(MeanEnabledBlock, self).__init__(**kwargs)
        self.avg_pool = AveragePooling2D((2, 2))
        self.max_pool = MaxPooling2D((2, 2))
        self.concat = Concatenate(axis=-1)

    def call(self, inputs):
        avg_feat = self.avg_pool(inputs)
        max_feat = self.max_pool(inputs)
        return self.concat([avg_feat, max_feat])

    def get_config(self):
        return super(MeanEnabledBlock, self).get_config()

# --- U-Net Components ---

def conv_block(input_tensor, num_filters):
    x = Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def laplacian_encoder_block(input_tensor, num_filters, dropout_rate=0.0):
    """
    Encoder that uses Mean-Enabled pooling (Avg+Max).
    """
    x = conv_block(input_tensor, num_filters)
    
    # Mean-Enabled Pooling: Combine Max and Avg to capture texture + peaks
    p_avg = AveragePooling2D((2, 2))(x)
    p_max = MaxPooling2D((2, 2))(x)
    p = Concatenate(axis=-1)([p_avg, p_max])
    
    # Since we doubled channels via concat, we optionally reduce them or just let the next layer handle it.
    # Let's keep the richness.
    
    if dropout_rate > 0:
        p = Dropout(dropout_rate)(p)
    return x, p

def decoder_block(input_tensor, skip_tensor, num_filters, dropout_rate=0.0):
    x = UpSampling2D((2, 2))(input_tensor)
    x = Concatenate()([x, skip_tensor])
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    x = conv_block(x, num_filters)
    return x

def build_unet():
    """
    Phase 3: Mean-Enabled Laplacian U-Net
    1. Input Branch: Original Image (4 Channels)
    2. Laplacian Branch: Edge Map of Image (4 Channels)
    3. Fusion: Concatenate Input + Edges before Encoder (8 Channels total)
    4. Pooling: Use Mean+Max pooling
    """
    inputs = Input((config.IMG_HEIGHT, config.IMG_WIDTH, config.NUM_CHANNELS))
    
    # 1. Laplacian Edge Extraction
    # Extract edges from raw input to guide the network (4 Independent Edge Maps)
    edges = LaplacianLayer()(inputs)
    
    # 2. Early Fusion
    # Merge original intensity with edge information (Shape becomes H, W, 8)
    fused_input = Concatenate()([inputs, edges])
    
    # Initial Conv to scale up to filter size
    x = Conv2D(config.FILTERS, (3, 3), padding='same')(fused_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # -- ENCODER (Using Mean-Enabled Pooling) --
    
    # Level 1
    # Note: We use the fused 'x' as start, but standard UNet skip connections
    # typically come from the conv block output.
    
    c1 = conv_block(x, config.FILTERS)
    # Custom Pooling (Mean + Max)
    p1_avg = AveragePooling2D((2, 2))(c1)
    p1_max = MaxPooling2D((2, 2))(c1)
    p1 = Concatenate(axis=-1)([p1_avg, p1_max])
    
    # Level 2
    c2 = conv_block(p1, config.FILTERS * 2)
    p2_avg = AveragePooling2D((2, 2))(c2)
    p2_max = MaxPooling2D((2, 2))(c2)
    p2 = Concatenate(axis=-1)([p2_avg, p2_max])
    if config.DROPOUT_RATE > 0: p2 = Dropout(0.1)(p2)

    # Level 3
    c3 = conv_block(p2, config.FILTERS * 4)
    p3_avg = AveragePooling2D((2, 2))(c3)
    p3_max = MaxPooling2D((2, 2))(c3)
    p3 = Concatenate(axis=-1)([p3_avg, p3_max])
    if config.DROPOUT_RATE > 0: p3 = Dropout(0.2)(p3)

    # Level 4
    c4 = conv_block(p3, config.FILTERS * 8)
    p4_avg = AveragePooling2D((2, 2))(c4)
    p4_max = MaxPooling2D((2, 2))(c4)
    p4 = Concatenate(axis=-1)([p4_avg, p4_max])
    if config.DROPOUT_RATE > 0: p4 = Dropout(0.2)(p4)

    # Level 5 (Bottom)
    c5 = conv_block(p4, config.FILTERS * 16)
    if config.DROPOUT_RATE > 0: c5 = Dropout(0.3)(c5)
    
    # -- BRIDGE --
    # No pooling here, just convolution
    
    # -- DECODER --
    d5 = decoder_block(c5, c4, config.FILTERS * 8, dropout_rate=0.2)
    d4 = decoder_block(d5, c3, config.FILTERS * 4, dropout_rate=0.2)
    d3 = decoder_block(d4, c2, config.FILTERS * 2, dropout_rate=0.1)
    d2 = decoder_block(d3, c1, config.FILTERS, dropout_rate=0.0)
    
    # -- OUTPUT --
    outputs = Conv2D(config.NUM_CLASSES, (1, 1), activation='softmax')(d2)
    
    model = Model(inputs=[inputs], outputs=[outputs], name="Laplacian_Mean_UNet")
    return model

if __name__ == "__main__":
    model = build_unet()
    model.summary()