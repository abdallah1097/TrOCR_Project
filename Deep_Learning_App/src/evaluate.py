import tensorflow as tf
import os
import sys

# Insert the Deep_Learning_App directory into the system path
# This allows importing modules from the specified directory
sys.path.insert(1, os.path.join(os.getcwd(), 'Deep_Learning_App'))

from src.TrOCR_model import TrOCR
from data_handler.data_splitter import DataSplitter
from data_handler.data_loader import CustomDataset
from src.utils.training_helpers import CustomSchedule, masked_loss, masked_accuracy
from src.config import config

def main():
    # Create an instance of the TrOCR model
    TrOCR_model = TrOCR()

    # Initialize the data splitter to split data into train, validation, and test sets
    data_splitter = DataSplitter(config.data_path, config.train_test_validation_ratios[1], config.train_test_validation_ratios[2])

    # Get test data paths
    test_paths = data_splitter.get_test_paths()
    print(f"Number of test samples: {len(test_paths)}")

    # Create a dataset generator for testing
    testing_dataset_generator = CustomDataset(test_paths)

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

    # Evaluate the model on the testing dataset
    evaluation_result = TrOCR_model.evaluate(testing_dataset_generator)

    # The evaluation_result typically contains the loss and selected metrics
    print("Loss:", evaluation_result[0])
    print("Accuracy:", evaluation_result[1])

if __name__ == '__main__':
    main()
