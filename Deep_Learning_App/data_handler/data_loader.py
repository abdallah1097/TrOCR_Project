from PIL import Image
import numpy as np
import os
import sys
# Caution: path[0] is reserved for the script path (or '' in REPL)
sys.path.insert(1, os.path.join(os.getcwd(), 'Deep_Learning_App'))

from data_handler.data_preprocessing import DataPreprocessor
from transformers import AutoTokenizer
from src.config import config
import tensorflow as tf
import random

class CustomDataset(tf.keras.utils.Sequence):
    """
    Custom dataset class for handling data loading and preprocessing.
    """

    def __init__(self, data_paths, transform=None):
        """
        Initialize the CustomDataset.

        Args:
            data_paths (list): List of data file paths.
            transform (callable, optional): Optional data transformation function.
        """
        self.data_paths = data_paths
        self.preprocessor = DataPreprocessor(config.image_size)
        self.tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
        self.vocab_size = self.tokenizer.vocab_size
        self.batch_size = config.batch_size

    def __len__(self):
        """
        Get the number of batches in the dataset.

        Returns:
            int: Number of batches.
        """
        return int(np.floor(len(self.data_paths) / self.batch_size))

    def on_epoch_end(self):
        """
        Shuffle the dataset at the end of each epoch.
        """
        self.data_paths = random.sample(self.data_paths, len(self.data_paths))

    def __getitem__(self, indx):
        """
        Get a batch of data.

        Args:
            indx (int): Batch index.

        Returns:
            tuple: A tuple containing input and target data.
        """
        start_idx = indx * self.batch_size
        end_idx = (indx + 1) * self.batch_size
        batch_indices = self.data_paths[start_idx:end_idx]

        [image_batch, labels_batch], labels_batch = self.__data_generation(batch_indices)
        return [image_batch, labels_batch], labels_batch

    def __data_generation(self, batch_indices):
        """
        Generate a batch of data including images and labels.

        Args:
            batch_indices (list): List of data file indices for the batch.

        Returns:
            tuple: A tuple containing input and target data.
        """
        image_batch = np.zeros((self.batch_size, config.image_height, config.image_width, config.number_channels),
                               dtype='float32')
        labels_batch = np.zeros((self.batch_size, config.max_length), dtype='float32')

        for i in range(len(batch_indices)):
            image_path = batch_indices[i] + ".jpg"
            image = Image.open(image_path).convert("RGB")
            preprocessed_image = self.preprocessor.preprocess(image)

            label_path = batch_indices[i] + ".txt"
            with open(label_path, "r", encoding='utf-8') as label_file:
                label = label_file.read().strip()
                tokenizer_output = self.tokenizer(label, padding='max_length',
                                                   max_length=config.max_length,
                                                   add_special_tokens=True)["input_ids"]

            image_batch[i, :, :, :] = preprocessed_image
            labels_batch[i, :] = tokenizer_output

        return [image_batch, labels_batch], labels_batch

    def __iter__(self):
        """
        Iterator function to iterate over batches of data.

        Yields:
            tuple: A tuple containing input and target data.
        """
        for i in range(self.__len__()):
            print('\n\nIn __iter__', i)
            yield self.__getitem__(i)
