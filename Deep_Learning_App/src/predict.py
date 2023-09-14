import tensorflow as tf
import os, sys
import argparse
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, os.path.join(os.getcwd(), 'Deep_Learning_App' ))

from transformers import GPT2Tokenizer, AutoTokenizer  # Import the GPT2Tokenizer from the Hugging Face Transformers library
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

    # tokenizer = GPT2Tokenizer.from_pretrained("akhooli/gpt2-small-arabic")
    tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")

    decoder_inputs = tokenizer('', padding='max_length', max_length=config.max_length, add_special_tokens=True)["input_ids"]
    sos_ID = tokenizer.bos_token_id
    eos_ID = tokenizer.eos_token_id
    
    # # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    preprocessor = DataPreprocessor(config.image_size)
    image = Image.open(image_path).convert("RGB") # Shape (80, 500, 3)
    preprocessed_image = preprocessor.preprocess(image)

    encoder_input_image = preprocessed_image[tf.newaxis, :, :, :]
    decoder_input_text = tf.convert_to_tensor([decoder_inputs])  # Add a new axis with [ ]

    print("Encoder/ Decoder Inputs", encoder_input_image.shape, decoder_input_text.shape)
    print("SOS ID:", sos_ID, "EOS ID:", eos_ID)

    # `tf.TensorArray` is required here (instead of a Python list), so that the
    # dynamic-loop can be traced by `tf.function`.
    # output_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    # output_array = output_array.write(0, sos_ID)

    for i in tf.range(config.max_length):
        # output = output_array.stack() # tf.transpose(output_array.stack())
        # print("Encoder Input Shapes:", np.array(encoder_input_image).shape, np.array(decoder_input_text).shape)
        predictions = TrOCR_model([encoder_input_image, decoder_input_text], training=False)

        # Select the last token from the `seq_len` dimension.
        predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.


        predicted_id = tf.argmax(predictions, axis=-1)
        predicted_id = tf.cast(predicted_id, dtype=tf.int32)

        # print("\npredicted_id:", predicted_id[0].shape, predicted_id[0])
        # print("Predicted is:", tokenizer.decode(predicted_id[0], skip_special_tokens=True))

        # Concatenate the `predicted_id` to the output which is given to the
        # decoder as its input.
        #   output_array = output_array.write(i+1, tf.transpose(predicted_id[0]))

        # decoder_input_text = decoder_input_text.index(0)
        predicted_ID_list = predicted_id[0].numpy().flatten().tolist()

        # decoder_input_text[0] = [predicted_ID_list.pop(0) if x == 0 else x for x in decoder_input_text[0]]
        for added_id in predicted_ID_list:
            numpy_decoder_input_text= decoder_input_text[0].numpy()
            zero_index = np.where(numpy_decoder_input_text == 0)[0][0] # Get index of the first zero only
            numpy_decoder_input_text[zero_index] = added_id
            decoder_input_text = tf.convert_to_tensor([numpy_decoder_input_text])  # Add a new axis with [ ]


        if predicted_id == eos_ID:
            break
      

        # The output shape is `(1, tokens)`.
        text = tokenizer.decode(decoder_input_text[0], skip_special_tokens=True)  # Shape: `()`.


        print("Predicted Text:", text)
        # sys.stdout.write("Pleaseeeeee" + "\n")

    return text


  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OCR_Detector Inference")
    # Define the command-line arguments
    parser.add_argument("--image_path", type=str, help="Absolute path of the image")

    # Parse the command-line arguments
    args = parser.parse_args()
    # print("Image Path:", args.image_path)
    # sys.stdout.write("Pleaseeeeee" + "\n")
    main(args.image_path)