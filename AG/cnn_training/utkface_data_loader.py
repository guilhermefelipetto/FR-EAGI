import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


class UTKFaceDataLoader:
    def __init__(self, dataset_path, img_shape=128):
        self.dataset_path = dataset_path
        self.img_shape = img_shape
        self.images = []
        self.ages = []
        self.genders = []

    def load_data(self):
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
        X_train, X_test, y_train_age, y_test_age, y_train_gender, y_test_gender = train_test_split(
            self.images, self.ages, self.genders, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train_age, y_test_age, y_train_gender, y_test_gender
