import matplotlib.pyplot as plt
import os
import sys
import tensorflow as tf
from src.config import config
from src.utils.base_attention import BaseAttention
from src.utils.feed_forward import FeedForward
from src.utils.positional_encoding import PositionalEncoding

# Add the 'Deep_Learning_App' directory to the Python path
sys.path.insert(1, os.path.join(os.getcwd(), 'Deep_Learning_App'))

class GlobalSelfAttention(BaseAttention):
    """
    Global Self-Attention Layer
    
    Args:
        x (tf.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        
    Returns:
        tf.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
    """
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class EncoderLayer(tf.keras.layers.Layer):
    """
    Encoder Layer
    
    Args:
        x (tf.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        
    Returns:
        tf.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
    """
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x

class Encoder(tf.keras.layers.Layer):
    """
    Encoder Layer
    
    Args:
        x (tf.Tensor): Input tensor of shape (batch_size, image_height, image_width, number_channels).
        training (bool): Boolean indicating if the model is in training mode.
        
    Returns:
        tf.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
    """
    def __init__(self, *, dropout_rate=0.1):
        super().__init__()

        # Ensure that image dimensions are divisible by the patch size
        assert config.image_height % config.patch_height == 0 and config.image_width % config.patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # Calculate the total number of patches and the dimension of each patch
        self.total_number_of_patches = (config.image_height // config.patch_height) * (config.image_width // config.patch_width)
        self.patch_dim = config.number_channels * config.patch_height * config.patch_width

        # Define a convolutional layer for image patching
        self.patched_image = tf.keras.layers.Conv2D(
            filters=config.d_model,
            kernel_size=(config.patch_height, config.patch_width),
            strides=(config.patch_height, config.patch_width),
            input_shape=(config.image_height, config.image_width, 3)  # Assumes input shape (height, width, channels)
        )

        self.pos_embedding = PositionalEncoding(
            length=self.total_number_of_patches,
            d_model=config.d_model)

        self.enc_layers = [
            EncoderLayer(d_model=config.d_model,
                         num_heads=config.num_heads,
                         dff=config.dff,
                         dropout_rate=dropout_rate)
            for _ in range(config.num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training):
        x = self.patched_image(x)  # Shape `(batch_size, total_number_of_batches, d_model)`
        x = tf.reshape(x, (-1, int((config.image_height // config.patch_height) * (config.image_width // config.patch_width)), config.d_model))
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
        x = self.dropout(x)
        for i in range(config.num_layers):
            x = self.enc_layers[i](x)
        return x  # Shape `(batch_size, seq_len, d_model)`.
