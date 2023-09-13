import tensorflow as tf

import os, sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, os.path.join(os.getcwd(), 'Deep_Learning_App' ))
import numpy as np
from src.TrOCR_model import TrOCR
from data_handler.data_splitter import DataSplitter
from data_handler.data_loader import CustomDataset
from src.utils.training_helpers import CustomSchedule, masked_loss, masked_accuracy
from src.config import config
from PIL import Image

def main():
    # print("\n\nWorking dir in train.py:", os.getcwd())
    TrOCR_model = TrOCR()

    data_splitter = DataSplitter(config.data_path, config.train_test_validation_ratios[1], config.train_test_validation_ratios[2])

    train_paths = data_splitter.get_train_paths()
    val_paths = data_splitter.get_val_paths()
    test_paths = data_splitter.get_test_paths()

    print(f"Number of training samples: {len(train_paths)}")
    print(f"Number of validation samples: {len(val_paths)}")
    print(f"Number of test samples: {len(test_paths)}")

    training_dataset_generator = CustomDataset(train_paths)
    validation_dataset_generator = CustomDataset(val_paths)
    testing_dataset_generator = CustomDataset(test_paths)

    # for inputs, outputs in training_dataset_generator:
    #     # tensor_image = tf.cast(inputs[0][0]*255, tf.uint8)

    #     # # Convert the TensorFlow tensor to a NumPy array
    #     # numpy_array = tensor_image.numpy()

    #     # # Create a PIL image from the NumPy array
    #     # pil_image = Image.fromarray(numpy_array)
    #     # pil_image.show()
    #     pass

    # writer = SummaryWriter()
    learning_rate = CustomSchedule(config.d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

    TrOCR_model.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy])
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=config.log_dir, histogram_freq=1)
    # Define a callback to save the best weights based on validation loss
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(config.deep_learning_model_path, 'best_model_weights.h5'),  # Specify the file where weights will be saved
        monitor='val_loss',               # Monitor validation loss
        save_best_only=True,              # Save only the best weights
        save_weights_only=True,
        mode='min',                       # Mode can be 'min' or 'max' depending on the monitored metric
        verbose=1                          # Set to 1 to see saving messages
    )

    print("\n\n\n\n\nThis is start of training... Steps:", len(training_dataset_generator))
    TrOCR_model.fit(training_dataset_generator,
                epochs=10,
                steps_per_epoch=len(training_dataset_generator),
                validation_data=validation_dataset_generator,
                validation_steps=len(validation_dataset_generator),
                verbose = 1,
                shuffle=False, 
                callbacks=[tensorboard_callback, checkpoint_callback],
                )
    

if __name__ == '__main__':
    main()