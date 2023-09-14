import os
import sys
import tensorflow as tf
from src.config import config
from src.utils.base_attention import BaseAttention
from src.utils.feed_forward import FeedForward
from src.utils.positional_embeddings import PositionalEmbedding
from transformers import AutoTokenizer  # Import the GPT2Tokenizer from the Hugging Face Transformers library

# Add the 'Deep_Learning_App' directory to the Python path
sys.path.insert(1, os.path.join(os.getcwd(), 'Deep_Learning_App'))

class CrossAttention(BaseAttention):
    """
    Cross-Attention Layer
    
    Args:
        x (tf.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        context (tf.Tensor): Context tensor of shape (batch_size, seq_len, d_model).
        
    Returns:
        tf.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
    """
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x

class CausalSelfAttention(BaseAttention):
    """
    Causal Self-Attention Layer
    
    Args:
        x (tf.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        
    Returns:
        tf.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
    """
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class DecoderLayer(tf.keras.layers.Layer):
    """
    Decoder Layer
    
    Args:
        x (tf.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        context (tf.Tensor): Context tensor of shape (batch_size, seq_len, d_model).
        
    Returns:
        tf.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
    """
    def __init__(self,
                 *,
                 d_model,
                 num_heads,
                 dff,
                 dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x

class Decoder(tf.keras.layers.Layer):
    """
    Decoder Layer
    
    Args:
        inputs (list): List containing two tensors - x and context.
        training (bool): Boolean indicating if the model is in training mode.
        
    Returns:
        tf.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
    """
    def __init__(self, *, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.total_number_of_patches = (config.image_height // config.patch_height) * (
            config.image_width // config.patch_width)

        self.tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
        self.vocab_size = self.tokenizer.vocab_size

        self.pos_embedding = PositionalEmbedding(
            vocab_size=self.vocab_size,
            length=config.max_length,
            d_model=config.d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=config.d_model, num_heads=config.num_heads,
                         dff=config.dff, dropout_rate=dropout_rate)
            for _ in range(config.num_layers)]

        self.last_attn_scores = None

    def call(self, inputs, training):
        x, context = inputs[0], inputs[1]

        # Token embeddings and positional encoding
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

        x = self.dropout(x)

        for i in range(config.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x
