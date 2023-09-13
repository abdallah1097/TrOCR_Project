from PIL import Image
import numpy as np
import os, sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, os.path.join(os.getcwd(), 'Deep_Learning_App' ))

from data_handler.data_preprocessing import DataPreprocessor
from transformers import GPT2Tokenizer  # Import the GPT2Tokenizer from the Hugging Face Transformers library
from src.config import config
import tensorflow as tf
import random

class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.preprocessor = DataPreprocessor(config.image_size)
        # self.image_paths = [os.path.join(data_dir, img_name) for img_name in os.listdir(data_dir)]
        # self.image_paths = [ data_path + ".jpg" for data_path in data_paths] 

        self.tokenizer = GPT2Tokenizer.from_pretrained("akhooli/gpt2-small-arabic")
        self.start_token = "<START>"
        # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    def __len__(self):
        # return len(self.data_paths)
        return int(np.floor(len(self.data_paths) / config.batch_size))
    
    def on_epoch_end(self):
        self.data_paths = random.sample(self.data_paths, len(self.data_paths))
        
    def __getitem__(self, indx):
        
        start_idx = indx * config.batch_size
        end_idx = (indx + 1) * config.batch_size
        batch_indices = self.data_paths[start_idx:end_idx]
        
        [image_batch, labels_batch], labels_batch = self.__data_generation(batch_indices)

        # print("\n\n__getitem__ index:", indx, " Shapes:", image_batch.shape, labels_batch.shape, labels_batch.shape)
        return [image_batch, labels_batch], labels_batch

    def __data_generation(self, batch_indices):
        image_batch = np.zeros((config.batch_size, config.image_height, config.image_width, config.number_channels), dtype='float32') #, dtype=
        labels_batch = np.zeros((config.batch_size, config.max_length), dtype='float32')
        

        for i in range(len(batch_indices)):
            image_path = batch_indices[i] + ".jpg"
            image = Image.open(image_path).convert("RGB") # Shape (80, 500, 3)
            preprocessed_image = self.preprocessor.preprocess(image)
            
            label_path = batch_indices[i] + ".txt"
            with open(label_path, "r", encoding='utf-8') as label_file:
                label = label_file.read().strip()
                label = self.start_token + " " + label
                tokenizer_output = self.tokenizer.encode(label,
                                                        padding="max_length",
                                                        return_tensors="tf",
                                                        max_length=config.max_length,
                                                        add_special_tokens=True
                                                                )
                tokenizer_output = tokenizer_output[0]
            
            # tokenizer_output = [1, tokenizer_output[:-1]]
            # tokenizer_output.insert(0, 1)
            # tokenizer_output = tokenizer_output[:-1]


            # print(tokenizer_output)
            image_batch[i, :, :, :]  = preprocessed_image
            labels_batch[i, :] = tokenizer_output
            
        
        return [image_batch, labels_batch], labels_batch

    def __iter__(self):
        for i in range(self.__len__()):
            print('\n\nIn __iter__', i)
            yield self.__getitem__(i)

    # def data_generator(self):
    #     # sample_path = self.data_paths[idx]
    #     for path in self.data_paths:
    #         image_path = path + ".jpg"
    #         image = Image.open(image_path).convert("RGB") # Shape (80, 500, 3)
    #         preprocessed_image = self.preprocessor.preprocess(image)
            
    #         label_path = path + ".txt"
    #         with open(label_path, "r", encoding='utf-8') as label_file:
    #             label = label_file.read().strip()
    #             tokenizer_output = self.tokenizer.encode(label,
    #                                                     padding="max_length",
    #                                                     return_tensors="pt",
    #                                                     max_length=config.max_length,
    #                                                     add_special_tokens=True
    #                                                             )
    #             tokenizer_output = tokenizer_output[0]
    #             # print("\n\nTracing: ", len(label), tokenizer_output.shape)
            
    #         # print(label)
    #         # image.show()
    #         # TO BE ADDED TOkenizer:
            
    #         # print("In getitem:", preprocessed_image.shape, tokenizer_output.shape)
    #     yield (preprocessed_image, tokenizer_output)
    
    # def create_dataset(self):
    #     dataset = tf.data.Dataset.from_generator(
    #         self.data_generator,
    #         output_types = ((tf.float32, tf.uint8)),
    #         output_shapes = (([config.image_height, config.image_width, config.number_channels], [config.max_length]))

    #     )
    #     # dataset = dataset.shuffle(buffer_size=len(self.data_paths))
    #     # dat aset = dataset.batch(config.batch_size)
    #     # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    #     return dataset
