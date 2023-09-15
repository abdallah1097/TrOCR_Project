import tensorflow as tf

class FeedForward(tf.keras.layers.Layer):
    """
    FeedForward layer for a neural network model.
    """

    def __init__(self, d_model, dff, dropout_rate=0.1):
        """
        Initialize the FeedForward layer.

        Args:
            d_model (int): Dimensionality of the model's output.
            dff (int): Dimensionality of the feedforward layer.
            dropout_rate (float): Dropout rate for regularization (default is 0.1).
        """
        super().__init__()

        # Sequential layers for feedforward processing
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),  # Dense layer with ReLU activation
            tf.keras.layers.Dense(d_model),  # Dense layer without activation
            tf.keras.layers.Dropout(dropout_rate)  # Dropout layer for regularization
        ])

        # Element-wise addition layer
        self.add = tf.keras.layers.Add()

        # Layer normalization
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        """
        Call function for the FeedForward layer.

        Args:
            x: Input tensor.

        Returns:
            tf.Tensor: Output tensor after feedforward processing.
        """
        x = self.add([x, self.seq(x)])  # Element-wise addition
        x = self.layer_norm(x)  # Layer normalization
        return x
