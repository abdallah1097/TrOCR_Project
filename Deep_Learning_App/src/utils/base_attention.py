import tensorflow as tf

class BaseAttention(tf.keras.layers.Layer):
    """
    Base attention layer for a neural network model.
    """

    def __init__(self, **kwargs):
        """
        Initialize the BaseAttention layer.

        Args:
            **kwargs: Keyword arguments to pass to MultiHeadAttention.
        """
        super().__init__()

        # Multi-head attention layer
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)

        # Layer normalization
        self.layernorm = tf.keras.layers.LayerNormalization()

        # Element-wise addition layer
        self.add = tf.keras.layers.Add()
