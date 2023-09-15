import os

class Config:
    def __init__(self):
        # Data-related settings
        self.data_path = os.path.join(os.getcwd(), 'media', 'dataset') # "D:/OCR Task/dataset/"
        self.log_dir = os.path.join(os.getcwd(), 'Deep_Learning_App', 'logs')
        self.deep_learning_model_path = os.path.join(os.getcwd(), 'Deep_Learning_App', 'models' , 'saved_models')
        self.train_test_validation_ratios = [0.7, 0.15, 0.15]
        self.split_random_seed = 52 # 42 # To fairly compare trials
        self.batch_size = 64
        self.image_size = (80, 500)
        self.number_channels = 3
        self.image_height = self.image_size[0]
        self.image_width = self.image_size[1]
        self.patch_height = self.image_height # to treat the entire letter as w whole
        self.patch_width = 25
        self.d_model = 128
        self.dff = 512
        self.num_heads = 8
        self.num_layers = 2
        self.max_length = 150
        self.vocab_size = 55000

        self.epochs = 10

# Create a single instance of the Config class
config = Config()