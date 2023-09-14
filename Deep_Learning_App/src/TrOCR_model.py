import os
import sys
import tensorflow as tf
from src.encoder import Encoder
from src.decoder import Decoder
from src.config import config

# Add the 'Deep_Learning_App' directory to the Python path
sys.path.insert(1, os.path.join(os.getcwd(), 'Deep_Learning_App'))

class TrOCR(tf.keras.Model):
    """
    TrOCR Model
    
    Args:
        dropout_rate (float): Dropout rate for the model.
        
    Attributes:
        encoder (Encoder): Instance of the Encoder class.
        decoder (Decoder): Instance of the Decoder class.
        final_layer (tf.keras.layers.Dense): Final linear layer for output.
    """
    def __init__(self, *, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.final_layer = tf.keras.layers.Dense(config.vocab_size)

    def call(self, inputs, training):
        """
        Forward pass of the TrOCR model.

        Args:
            inputs (list): List containing two tensors - src_image and target_text.
            training (bool): Boolean indicating if the model is in training mode.

        Returns:
            logits (tf.Tensor): The output logits from the model. It has shape (batch_size, seq_length, vocab_size).
        """
        src_image, target_text = inputs[0], inputs[1]

        # Encoding phase
        encoder_output = self.encoder(src_image, training=training)

        # Decoding phase
        x = self.decoder([target_text, encoder_output], training=training)

        # Final linear layer output.
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the final output.
        return logits
