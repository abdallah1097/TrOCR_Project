import tensorflow as tf
import os
import sys
import argparse
from transformers import AutoTokenizer
from src.TrOCR_model import TrOCR
from data_handler.data_splitter import DataSplitter
from data_handler.data_loader import CustomDataset
from src.utils.training_helpers import CustomSchedule, masked_loss, masked_accuracy
from src.config import config
from PIL import Image
from data_handler.data_preprocessing import DataPreprocessor
import numpy as np

def main(image_path):
    # Initialize the TrOCR model
    TrOCR_model = TrOCR()

    # Define the learning rate schedule
    learning_rate = CustomSchedule(config.d_model)

    # Initialize the Adam optimizer with custom learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # Compile the TrOCR model with custom loss and accuracy metrics
    TrOCR_model.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy])

    # Define input shapes for building the model
    image_input = (None, config.image_height, config.image_width, config.number_channels)
    target_text_input = (None, config.max_length)

    # Build the TrOCR model
    TrOCR_model.build(input_shape=[image_input, target_text_input])

    # Load pre-trained model weights
    TrOCR_model.load_weights(os.path.join(config.deep_learning_model_path, 'best_model_weights.h5'))

    # Initialize the tokenizer for text generation
    tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")

    # Define initial decoder input with special tokens
    decoder_inputs = tokenizer('', padding='max_length', max_length=tokenizer.vocab_size, add_special_tokens=True)["input_ids"]
    sos_ID = tokenizer.bos_token_id
    eos_ID = tokenizer.eos_token_id

    # Initialize the data preprocessor for images
    preprocessor = DataPreprocessor(config.image_size)

    # Open and preprocess the input image
    image = Image.open(image_path).convert("RGB")
    preprocessed_image = preprocessor.preprocess(image)

    # Add a new axis to the input tensors
    encoder_input_image = preprocessed_image[tf.newaxis, :, :, :]
    decoder_input_text = tf.convert_to_tensor([decoder_inputs])

    for i in tf.range(config.max_length):
        predictions = TrOCR_model([encoder_input_image, decoder_input_text], training=False)

        # Select the last token from the `seq_len` dimension.
        predictions = predictions[:, -1:, :]

        predicted_id = tf.argmax(predictions, axis=-1)
        predicted_id = tf.cast(predicted_id, dtype=tf.int32)

        predicted_ID_list = predicted_id[0].numpy().flatten().tolist()

        for added_id in predicted_ID_list:
            numpy_decoder_input_text = decoder_input_text[0].numpy()
            zero_index = np.where(numpy_decoder_input_text == 0)[0]
            if zero_index.size > 0:
                zero_index = zero_index[0]
                numpy_decoder_input_text[zero_index] = added_id
                decoder_input_text = tf.convert_to_tensor([numpy_decoder_input_text])

        if predicted_id == eos_ID:
            break

    # Decode the predicted text
    text = tokenizer.decode(decoder_input_text[0], skip_special_tokens=True)

    # Print the decoded text
    sys.stdout.write(text)

    return text

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OCR_Detector Inference")
    # Define the command-line arguments
    parser.add_argument("--image_path", type=str, help="Absolute path of the image")

    # Parse the command-line arguments
    args = parser.parse_args()
    main(args.image_path)
