import os, sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, os.path.join(os.getcwd(), 'Deep_Learning_App' ))

from src.config import config  # Import the configuration from the 'utils' module
from src.utils.base_attention import BaseAttention
from src.utils.feed_forward import FeedForward
from src.utils.positional_embeddings import PositionalEmbedding
import tensorflow as tf



class CrossAttention(BaseAttention):
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
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask = True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class DecoderLayer(tf.keras.layers.Layer):
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
  def __init__(self, *, dropout_rate=0.1):
    super(Decoder, self).__init__()

    self.total_number_of_patches = (config.image_height // config.patch_height) * (config.image_width // config.patch_width)

    self.pos_embedding = PositionalEmbedding(vocab_size = config.vocab_size,
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
    # print("1. Decoder Tracing:", x.shape, context.shape)
    # `x` is token-IDs shape (batch, target_seq_len)
    x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)
    # print("2. Decoder Tracing:", x.shape)

    x = self.dropout(x)

    for i in range(config.num_layers):
      x  = self.dec_layers[i](x, context)

    self.last_attn_scores = self.dec_layers[-1].last_attn_scores

    # The shape of x is (batch_size, target_seq_len, d_model).
    return x
  

# import torch  # Import the PyTorch library
# from torch import nn  # Import the neural network module from PyTorch
# from transformers import GPT2Tokenizer  # Import the GPT2Tokenizer from the Hugging Face Transformers library
# from deep_learning.utils.config import config  # Import the configuration from the 'utils' module
# import math  # Import the math module
# import numpy as np  # Import the NumPy library for numerical operations
# import torch.nn.functional as F  # Import functional operations from PyTorch


# # Check if a CUDA-enabled GPU is available, otherwise use CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Define a custom decoder class as a subclass of nn.Module
# class decoder(nn.Module):
#     def __init__(self):
#         super(decoder, self).__init__()
#         # Ensure that image dimensions are divisible by the patch size
#         assert config.image_height % config.patch_height == 0 and config.image_width % config.patch_width == 0, 'Image dimensions must be divisible by the patch size.'

#         # Calculate the total number of patches and the dimension of each patch
#         self.total_num_patches = (config.image_height // config.patch_height) * (config.image_width // config.patch_width)
#         self.patch_dim = config.number_channels * config.patch_height * config.patch_width

#         # Create a target mask for the transformer
#         self.tgt_mask = torch.triu(torch.ones(config.max_length, config.max_length)).T
#         self.tgt_mask = self.tgt_mask.masked_fill(self.tgt_mask == 0, float('-inf')).to(device)

#         # Initialize a GPT2 tokenizer with an Arabic language model
#         self.tokenizer = GPT2Tokenizer.from_pretrained("akhooli/gpt2-small-arabic")
#         self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#         # Define vocabulary size and embeddings layer
#         self.vocab_size = self.tokenizer.vocab_size + 1
#         self.embeddings = nn.Embedding(self.vocab_size, config.hidden_size)

#         # Create positional encoding
#         self.positional_encoding = PositionalEncoding(config.hidden_size, self.total_num_patches)

#         # Define the transformer decoder
#         self.transformer_decoder = nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(d_model=config.hidden_size,
#                                        nhead=config.num_heads,
#                                        activation="gelu",
#                                        norm_first=True),
#             num_layers=config.num_layers
#         )

#         # Define the fully connected layer and activation function
#         self.fc = nn.Linear(config.hidden_size, self.vocab_size)
#         self.activation = nn.Softmax(dim=-1)

#     def forward(self, target_text, encoder_output):
#         target_text = list(target_text)
#         # Tokenize and prepare the input text
#         tokenizer_output = self.tokenizer.batch_encode_plus(target_text,
#                                                             padding="max_length",
#                                                             return_tensors="pt",
#                                                             max_length=config.max_length,
#                                                             add_special_tokens=True
#                                                             ).to(device)

#         target_ids = tokenizer_output['input_ids']
#         target_embedded_text = self.embeddings(tokenizer_output['input_ids']).permute(1, 0, 2)

#         # Pass the input through the transformer decoder
#         x = self.transformer_decoder(
#             tgt=target_embedded_text,
#             memory=encoder_output,
#             tgt_mask=self.tgt_mask
#         )

#         # Apply softmax activation and reshape the output
#         output_probabilities = self.activation(self.fc(x)).permute(1, 0, 2)
#         output_probabilities = output_probabilities.reshape(-1, output_probabilities.shape[-1])
#         target_ids = target_ids.reshape(-1)
#         return output_probabilities, target_ids

# # Define a positional encoding class
# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model, max_len, dropout: float = 0):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         # Calculate positional encoding values
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         """
#         Arguments:
#             x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
#         """
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)