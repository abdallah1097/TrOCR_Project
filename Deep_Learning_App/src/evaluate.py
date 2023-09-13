import tensorflow as tf

import os, sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, os.path.join(os.getcwd(), 'deep_learning' ))

from src.TrOCR_model import TrOCR
from data_handler.data_splitter import DataSplitter
from data_handler.data_splitter import DataSplitter
from data_handler.data_loader import CustomDataset
from src.utils.training_helpers import CustomSchedule, masked_loss, masked_accuracy
from src.config import config


def main():
    TrOCR_model = TrOCR()
    data_splitter = DataSplitter(config.data_path, config.train_test_validation_ratios[1], config.train_test_validation_ratios[2])

    test_paths = data_splitter.get_test_paths()
    print(f"Number of test samples: {len(test_paths)}")
    testing_dataset_generator = CustomDataset(test_paths)

    # writer = SummaryWriter()
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

    # Assuming `model` is your loaded TensorFlow model and `test_generator` is your test data generator
    evaluation_result = TrOCR_model.evaluate(testing_dataset_generator)

    # The evaluation_result typically contains the loss and selected metrics
    print("Loss:", evaluation_result[0])
    print("Accuracy:", evaluation_result[1])

if __name__ == '__main__':
    main()