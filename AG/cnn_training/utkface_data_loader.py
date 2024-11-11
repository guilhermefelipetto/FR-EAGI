import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


class UTKFaceDataLoader:
    """
    A data loader class for the UTKFace dataset.
    Attributes:
        dataset_path (str): Path to the dataset directory.
        img_shape (int): The shape to which images will be resized. Default is 128.
        images (list): List to store loaded images.
        ages (list): List to store ages corresponding to the images.
        genders (list): List to store genders corresponding to the images.
    Methods:
        load_data():
            Loads images and their corresponding age and gender labels from the dataset directory.
            Returns:
                tuple: A tuple containing numpy arrays of images, ages, and genders.
        train_test_split(test_size=0.2, random_state=42):
            Splits the data into training and testing sets.
            Args:
                test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
                random_state (int): Controls the shuffling applied to the data before applying the split. Default is 42.
            Returns:
                tuple: A tuple containing training and testing sets for images, ages, and genders.
    """
    
    def __init__(self, dataset_path, img_shape=128):
        self.dataset_path = dataset_path
        self.img_shape = img_shape
        self.images = []
        self.ages = []
        self.genders = []

    def load_data(self):
        """
        Loads image data from the dataset path, extracting age, gender, and race information from filenames.
        The method iterates through all files in the dataset path, processes only those with a '.jpg' extension,
        and extracts age, gender, and race information from the filename. It reads the image, resizes it to the
        specified shape, normalizes pixel values, and appends the processed image and extracted information to
        respective lists. In case of an error during processing, it prints an error message.
        Returns:
            tuple: A tuple containing three numpy arrays:
            - images (numpy.ndarray): Array of processed images.
            - ages (numpy.ndarray): Array of ages corresponding to the images.
            - genders (numpy.ndarray): Array of genders corresponding to the images.
        """
        
        for filename in os.listdir(self.dataset_path):
            if filename.endswith('.jpg'):
                try:
                    age, gender, race, _ = filename.split('_')
                    img_path = os.path.join(self.dataset_path, filename)
                    image = cv2.imread(img_path)
                    image = cv2.resize(image, (self.img_shape, self.img_shape))
                    image = image / 255.0
                    self.images.append(image)
                    self.ages.append(int(age))
                    self.genders.append(int(gender))
                except Exception as e:
                    print(f'Error loading image {filename}: {e}')
        
        self.images = np.array(self.images)
        self.ages = np.array(self.ages)
        self.genders = np.array(self.genders)
        return self.images, self.ages, self.genders

    def train_test_split(self, test_size=0.2, random_state=42):
        """
        Splits the dataset into training and testing sets.
        Parameters:
        test_size (float): The proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Controls the shuffling applied to the data before applying the split. Default is 42.
        Returns:
        tuple: A tuple containing the following elements:
            - X_train (array-like): Training data.
            - X_test (array-like): Testing data.
            - y_train_age (array-like): Training labels for age.
            - y_test_age (array-like): Testing labels for age.
            - y_train_gender (array-like): Training labels for gender.
            - y_test_gender (array-like): Testing labels for gender.
        """
        
        X_train, X_test, y_train_age, y_test_age, y_train_gender, y_test_gender = train_test_split(
            self.images, self.ages, self.genders, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train_age, y_test_age, y_train_gender, y_test_gender
