import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
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
    
    # An offline converter for TF-TRT transformation for TF 2.0 SavedModels.

    # Define to convert to FP32/FP16 precision
    params = tf.experimental.tensorrt.ConversionParams(
        precision_mode='FP16')
    converter = tf.experimental.tensorrt.Converter(
        input_saved_model_dir=model_path, conversion_params=params)
    converter.convert()

    # Saving the model
    converter.save(output_path)



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