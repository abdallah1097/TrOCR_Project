import tensorflow as tf
import os
import sys

import os, sys
sys.path.insert(1, os.path.join(os.getcwd(), 'Deep_Learning_App' ))

from src.TrOCR_model import TrOCR
from data_handler.data_splitter import DataSplitter
from data_handler.data_loader import CustomDataset
from src.utils.training_helpers import CustomSchedule, masked_loss, masked_accuracy
from src.config import config
from PIL import Image

def main():
    # Initialize the TrOCR model
    TrOCR_model = TrOCR()

    # Initialize the DataSplitter
    data_splitter = DataSplitter(config.data_path, config.train_test_validation_ratios[1], config.train_test_validation_ratios[2])

    # Get data paths for training, validation, and testing
    train_paths = data_splitter.get_train_paths()
    val_paths = data_splitter.get_val_paths()
    test_paths = data_splitter.get_test_paths()

    # Print the number of samples in each dataset
    print(f"Number of training samples: {len(train_paths)}")
    print(f"Number of validation samples: {len(val_paths)}")
    print(f"Number of test samples: {len(test_paths)}")

    # Create dataset generators for training, validation, and testing
    training_dataset_generator = CustomDataset(train_paths)
    validation_dataset_generator = CustomDataset(val_paths)
    testing_dataset_generator = CustomDataset(test_paths)

    # Define the learning rate schedule
    learning_rate = CustomSchedule(config.d_model)
    
    # Initialize the Adam optimizer with custom learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # Compile the TrOCR model
    TrOCR_model.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy])
    
    # Define a TensorBoard callback for logging
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=config.log_dir, histogram_freq=1)
    
    # Define a callback to save the best weights based on validation loss
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(config.deep_learning_model_path, 'best_model_weights.h5'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        verbose=1
    )


    # Start training
    TrOCR_model.fit(
        training_dataset_generator,
        epochs=config.epochs,
        steps_per_epoch=len(training_dataset_generator),
        validation_data=validation_dataset_generator,
        validation_steps=len(validation_dataset_generator),
        verbose=1,
        shuffle=False,
        callbacks=[tensorboard_callback, checkpoint_callback]
    )

if __name__ == '__main__':
    main()
