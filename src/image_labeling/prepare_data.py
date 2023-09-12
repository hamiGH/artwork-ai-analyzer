import os
import ast
import operator
import cv2 as cv
import numpy as np
import pandas as pd
from functools import reduce
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from .custom_data_generator import CustomDataGenerator
from ..utils import load_params


class PrepareData:
    def __init__(self, config_dir: str):
        # loading config parameters
        params = load_params(os.path.join(config_dir, "config.yml"))

        # creating a root directory for images
        self.data_dir = os.path.join(Path(params['data_dir']), "train") if params['phase'] == 'train' \
            else os.path.join(Path(params['data_dir']), "test")
        self.batch_size = params['batch_size']
        self.white_labels = params['white_labels']
        self.input_shape = params['input_shape']
        self.validation_split = params['validation_split']
        self.test_split = params['test_split']
        self.classes = None

    def read_data(self):
        # Lists to store images data
        total_images = []
        total_labels = []

        df_labels = pd.read_csv(os.path.join(self.data_dir, "art_dataset.csv"),
                                usecols=['image_name', 'category_labels']).set_index('image_name')

        df_labels['category_labels'] = df_labels['category_labels'].apply(ast.literal_eval)

        # Loop through all classes in subset
        for folder in os.listdir(self.data_dir):
            # Loop through all images in each class
            path_to_folder = os.path.join(self.data_dir, folder)
            if os.path.isdir(path_to_folder):
                for image in os.listdir(path_to_folder):
                    label = df_labels.loc[image, 'category_labels']
                    filtered_labels = list(set(label).intersection(set(self.white_labels)))
                    if filtered_labels:
                        # Defining path to image
                        path_to_image = os.path.join(self.data_dir, folder, image)
                        # Reading image using cv2
                        image = cv.imread(path_to_image)
                        # Resizing image
                        image = cv.resize(image, (self.input_shape[1], self.input_shape[0]))
                        # Appending images and labels
                        total_images.append(image)
                        total_labels.append(filtered_labels)

        # Return converted X to numpy array and scale values between 0 and 1
        total_images = np.array(total_images) / 255.0

        # Binarize labels
        mlb = MultiLabelBinarizer()
        total_labels = mlb.fit_transform(total_labels)
        self.classes = mlb.classes_

        return total_images, total_labels

    @staticmethod
    def __calculate_probability(train_labels):
        category_counts = Counter(category for sample_label in train_labels for category in sample_label)

        # sorted_labels_counts = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        categories, counts = zip(*category_counts.items())

        total_count = sum(counts)
        category_probabilities = {category: count / total_count for category, count in zip(categories, counts)}

        sample_probabilities = []
        for index, sample_label in enumerate(train_labels):
            mult = reduce(operator.mul, [category_probabilities[category] for category in sample_label], 1)
            probability = 1 - mult
            # probability = 1 - sum([category_probabilities[category] for category in sample_label])
            sample_probabilities.append(probability)

        sample_probabilities = np.array(sample_probabilities)
        sample_probabilities /= sample_probabilities.sum()  # Normalize to ensure they sum to 1

        return sample_probabilities

    def create_train_val_data(self, batch_size):
        total_images, total_labels = self.read_data()

        x, test_x, y, test_y = train_test_split(total_images, total_labels, test_size=self.test_split,
                                                shuffle=True, random_state=1)

        train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=self.validation_split,
                                                          shuffle=True, random_state=1)

        sample_probabilities = self.__calculate_probability(train_y)

        validation_test_dataset = {'val_x': val_x, 'val_y': val_y, 'test_x': test_x, 'test_y': test_y}

        print(train_x.shape, train_y.shape)
        print(val_x.shape, val_y.shape)

        # using ImageDataGenerator to apply random transformations to images
        # image_generator = ImageDataGenerator(rotation_range=45,
        #                                      width_shift_range=0.1,
        #                                      height_shift_range=0.1,
        #                                      zoom_range=0.2,
        #                                      horizontal_flip=True,
        #                                      validation_split=0.2)

        image_generator = ImageDataGenerator(rotation_range=0)

        train_generator = CustomDataGenerator(train_x, train_y, batch_size, sample_probabilities, image_generator)

        return train_generator, validation_test_dataset, self.classes
