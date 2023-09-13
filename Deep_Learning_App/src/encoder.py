import matplotlib.pyplot as plt

import os, sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, os.path.join(os.getcwd(), 'Deep_Learning_App' ))


from src.config import config  # Import the configuration from the 'utils' module
from src.utils.base_attention import BaseAttention
from src.utils.feed_forward import FeedForward
from src.utils.positional_encoding import PositionalEncoding
import tensorflow as tf


class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
  
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
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
    # print("1. Encoder Tracing:", x.shape)
    x = self.patched_image(x) # Shape `(batch_size, total_number_of_batches, )`
    # print("2. Encoder Tracing:", x.shape)
    # x = tf.reshape(x, (-1, int(x.shape[1]*x.shape[2]), config.d_model))  # Equivalent to view(B, C, -1) in PyTorch
    x = tf.reshape(x, (-1, int((config.image_height // config.patch_height)*(config.image_width // config.patch_width)), config.d_model))  # Equivalent to view(B, C, -1) in PyTorch
    # print("3. Encoder Tracing:", x.shape)
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
    # print("4. Encoder Tracing:", x.shape)
    # Add dropout.
    x = self.dropout(x)
    for i in range(config.num_layers):
      x = self.enc_layers[i](x)
    return x  # Shape `(batch_size, seq_len, d_model)`.