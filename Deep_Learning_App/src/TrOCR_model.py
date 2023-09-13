import os, sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, os.path.join(os.getcwd(), 'Deep_Learning_App' ))

from src.encoder import Encoder
from src.decoder import Decoder
import tensorflow as tf
from src.config import config

class TrOCR(tf.keras.Model):
  def __init__(self, *, dropout_rate=0.1):
    super().__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()

    self.final_layer = tf.keras.layers.Dense(config.vocab_size)

  def call(self, inputs, training):
        """
        Forward pass of the TrOCR model.

        Args:
            src_image (Tensor): Input source image data. It should have shape (batch_size, channels, height, width).
            target_text (list of str): List of target text sequences for decoding.

        Returns:
            decoder_output_probabilities (Tensor): The output probabilities from the decoder. It has shape (seq_length, batch_size, vocab_size).
            target_ids (Tensor): The target text converted to token IDs. It has shape (seq_length * batch_size).
        """
        # print("0.   In model Tracing:", inputs[0].shape, inputs[1].shape)
        src_image, target_text = inputs[0], inputs[1]
        # print("1.   In model Tracing:", src_image.shape, target_text.shape, training)
        # Encoding phase
        encoder_output = self.encoder(src_image, training=training)
        # print("2.   In model Tracing:", target_text.shape, encoder_output.shape)
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

        # Return the final output and the attention weights.
        return logits
  

# class TrOCR(nn.Module):
#     def __init__(self):
#         super(TrOCR, self).__init__()
#         # Initialize the encoder and decoder modules
#         self.encoder = encoder()
#         self.decoder = decoder()
        
#     def forward(self, src_image, target_text):
#         """
#         Forward pass of the TrOCR model.

#         Args:
#             src_image (Tensor): Input source image data. It should have shape (batch_size, channels, height, width).
#             target_text (list of str): List of target text sequences for decoding.

#         Returns:
#             decoder_output_probabilities (Tensor): The output probabilities from the decoder. It has shape (seq_length, batch_size, vocab_size).
#             target_ids (Tensor): The target text converted to token IDs. It has shape (seq_length * batch_size).
#         """
#         # Encoding phase
#         encoder_output = self.encoder(src_image)
#         # Decoding phase
#         decoder_output_probabilities, target_ids = self.decoder(target_text, encoder_output)

#         return decoder_output_probabilities, target_ids
