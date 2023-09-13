import os
import random
from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, data_dir, validation_split=0.15, test_split=0.15, random_seed=42):
        self.data_dir = data_dir
        self.validation_split = validation_split
        self.test_split = test_split
        self.random_seed = random_seed
        
        # Split the path and get the first (name) removing the second (extension)
        # Collect a list of unique image paths by removing extensions and joining with the data directory
        self.samples_paths = sorted(
            list(
                set(
                    [os.path.join(self.data_dir, os.path.splitext(img_name)[0]) for img_name in os.listdir(self.data_dir.replace("\\", "/"))]
                )
            )
        )
        assert self.check_unlabelled_samples()  # Ensure that the samples are labeled correctly

        self.num_images = len(self.samples_paths)
        
        # Split paths into training, validation, and test sets
        self.train_paths, self.test_paths = train_test_split(self.samples_paths, test_size=self.test_split, random_state=random_seed)
        self.train_paths, self.val_paths = train_test_split(self.train_paths, test_size=self.validation_split, random_state=random_seed)
    
    def get_train_paths(self):
        return self.train_paths
    
    def get_val_paths(self):
        return self.val_paths
    
    def get_test_paths(self):
        return self.test_paths

    def has_corresponding_files(self, path):
        txt_path = path + ".txt"
        jpg_path = path + ".jpg"
        return os.path.exists(txt_path) and os.path.exists(jpg_path)

    def check_unlabelled_samples(self):
        # Check if the number of labeled images matches the number of label files
        if len([image_path for image_path in self.samples_paths if os.path.isfile(image_path+".jpg")]) == len([label_path for label_path in self.samples_paths if os.path.isfile(label_path+".txt")]):
            print("Data is balanced!")
            return True
        else:
            difference = abs(
                    len([image_path for image_path in self.samples_paths if os.path.isfile(image_path+".jpg")]) \
                        - len([label_path for label_path in self.samples_paths if os.path.isfile(label_path+".txt")])
                            )
            print(f"WARNING: There are {difference} unlabelled data")
            print("Removing unlabelled data...")
            # Remove unlabelled data paths from the list
            self.samples_paths = [path for path in self.samples_paths if self.has_corresponding_files(path)]
            return True
