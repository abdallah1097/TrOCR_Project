import tensorflow as tf
import numpy as np

class PositionalEmbedding(tf.keras.layers.Layer):
    """
    Positional embedding layer for use in transformer models.
    """

    def __init__(self, vocab_size, length, d_model):
        """
        Initialize the PositionalEmbedding layer.

        Args:
            vocab_size (int): Size of the vocabulary.
            length (int): Length of the input sequence.
            d_model (int): Dimensionality of the model's output.
        """
        super().__init__()
        self.d_model = d_model
        self.length = length
        self.vocab_size = vocab_size

        # Embedding layer with mask_zero=True to handle padding masks
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.d_model, mask_zero=True)
        self.pos_encoding = self.pos_enc(length=self.length, depth=self.d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        """
        Call function for the PositionalEmbedding layer.

        Args:
            x: Input tensor.

        Returns:
            tf.Tensor: Output tensor after positional embedding.
        """
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # Scaling factor to balance the embedding and positional encoding
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

    def pos_enc(self, length, depth):
        """
        Calculate positional encodings.

        Args:
            length (int): Length of the sequence.
            depth (int): Depth of the positional encoding.

        Returns:
            tf.Tensor: Positional encoding tensor.
        """
        depth = depth / 2

        positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
        depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

        angle_rates = 1 / (10000**depths)  # (1, depth)
        angle_rads = positions * angle_rates  # (pos, depth)

        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1)

        return tf.cast(pos_encoding, dtype=tf.float32)
