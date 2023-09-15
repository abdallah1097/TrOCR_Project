import tensorflow as tf
import argparse
import tf2onnx
import os, sys
# Insert the Deep_Learning_App directory into the system path
# This allows importing modules from the specified directory
sys.path.insert(1, os.path.join(os.getcwd(), 'Deep_Learning_App'))
from src.config import config

from transformers import AutoTokenizer
from src.TrOCR_model import TrOCR
from data_handler.data_splitter import DataSplitter
from data_handler.data_loader import CustomDataset
from src.utils.training_helpers import CustomSchedule, masked_loss, masked_accuracy
from src.config import config
from PIL import Image
from data_handler.data_preprocessing import DataPreprocessor
import numpy as np

def main(model_path, output_path):
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
    TrOCR_model.load_weights(model_path)

    # Convert the model to ONNX format
    onnx_model, _ = tf2onnx.convert._from_keras_tf1(TrOCR_model)
    print(onnx_model)

    # Save the ONNX model to a file
    with open(os.path.join(output_path, 'model.onnx'), 'wb') as f:
        f.write(onnx_model.SerializeToString())



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert Tensorflow to Onnx")
    # Define the command-line arguments
    parser.add_argument("--model_path", type=str, help="Absolute path of the .h5 model", 
                        default=os.path.join(config.deep_learning_model_path, 'best_model_weights.h5'))
    parser.add_argument("--output_path", type=str, help="Absolute path of the onnx model", 
                        default=config.deep_learning_model_path)

    # Parse the command-line arguments
    args = parser.parse_args()
    main(args.model_path, args.output_path)