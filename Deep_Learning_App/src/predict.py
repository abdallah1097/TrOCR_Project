import tensorflow as tf
import os, sys
import argparse
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, os.path.join(os.getcwd(), 'Deep_Learning_App' ))

from transformers import GPT2Tokenizer  # Import the GPT2Tokenizer from the Hugging Face Transformers library
from src.TrOCR_model import TrOCR
from data_handler.data_splitter import DataSplitter
from data_handler.data_loader import CustomDataset
from src.utils.training_helpers import CustomSchedule, masked_loss, masked_accuracy
from src.config import config
from PIL import Image
from data_handler.data_preprocessing import DataPreprocessor
import numpy as np

def main(image_path):
    TrOCR_model = TrOCR()
    learning_rate = CustomSchedule(config.d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                        epsilon=1e-9)

    TrOCR_model.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy])

    image_input = (None, config.image_height, config.image_width, config.number_channels)
    target_text_input = (None, config.max_length)
    TrOCR_model.build(input_shape=[image_input, target_text_input])
    TrOCR_model.load_weights(os.path.join(config.deep_learning_model_path, 'best_model_weights.h5'))

    tokenizer = GPT2Tokenizer.from_pretrained("akhooli/gpt2-small-arabic")
    start_token = "<START>"
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    preprocessor = DataPreprocessor(config.image_size)
    image = Image.open(image_path).convert("RGB") # Shape (80, 500, 3)
    preprocessed_image = preprocessor.preprocess(image)

    encoder_input = preprocessed_image[tf.newaxis, :, :, :]
    
        
    start = "<START>"
    start = tokenizer.encode([start],
                                padding=False,
                                return_tensors="tf",
                                add_special_tokens=True
                                        )
    start = start[0][tf.newaxis]
    print("start", start.shape, start)
    end = tokenizer.encode([''],
                                padding=False,
                                return_tensors="tf",
                                add_special_tokens=True
                                        )
    end = end[0][tf.newaxis]
    print("end", end.shape, end)

    # `tf.TensorArray` is required here (instead of a Python list), so that the
    # dynamic-loop can be traced by `tf.function`.
    output_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    output_array = output_array.write(0, start)

    for i in tf.range(config.max_length):
      output = tf.transpose(output_array.stack())
      print("Encoder Input Shapes:", np.array(encoder_input).shape, np.array(output).shape, output)
      predictions = TrOCR_model([encoder_input, output], training=False)

      # Select the last token from the `seq_len` dimension.
      predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

      predicted_id = tf.argmax(predictions, axis=-1)
      predicted_id = tf.cast(predicted_id, dtype=tf.int32)

      # print("\npredicted_id:", predicted_id[0].shape, predicted_id[0])
      # print("Predicted is:", tokenizer.decode(predicted_id[0], skip_special_tokens=True))

      # Concatenate the `predicted_id` to the output which is given to the
      # decoder as its input.
      output_array = output_array.write(i+1, predicted_id[0])

      if predicted_id == end:
        break

    output = tf.transpose(output_array.stack())
    print("output:", output.shape, output)
    # The output shape is `(1, tokens)`.
    text = tokenizer.decode(output[0], skip_special_tokens=True)  # Shape: `()`.

    # tokens = tokenizers.en.lookup(output)[0]

    # # `tf.function` prevents us from using the attention_weights that were
    # # calculated on the last iteration of the loop.
    # # So, recalculate them outside the loop.
    # self.transformer([encoder_input, output[:,:-1]], training=False)
    # attention_weights = self.transformer.decoder.last_attn_scores
    print("Predicted Text:", text)

    return text


  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OCR_Detector Inference")
    # Define the command-line arguments
    parser.add_argument("image_path", type=str, help="Absolute path of the image")

    # Parse the command-line arguments
    args = parser.parse_args()
    print("Image Path:", args.image_path)
    main(args.image_path)