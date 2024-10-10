import os

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
from keras.api.layers import (Conv2D, Dense, Dropout, Flatten, Input,
                              MaxPooling2D)
from keras.api.models import Model
from sklearn.model_selection import train_test_split

dataset_path = 'datasets\\UTKFace'

images = []
ages = []
genders = []

shape = 128
epochs = 70
batch_size = 32

def load_utkface_data(dataset_path):
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):
            try:
                age, gender, race, _ = filename.split('_')

                img_path = os.path.join(dataset_path, filename)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (shape, shape))
                
                image = image / 255.0

                images.append(image)
                ages.append(int(age))
                genders.append(int(gender))
            except Exception as e:
                print(f"Erro ao carregar a imagem {filename}: {e}")

load_utkface_data(dataset_path)

model_name = 'modelo_utkface_keras.h5'
images = np.array(images)
ages = np.array(ages)
genders = np.array(genders)

X_train, X_test, y_train_age, y_test_age, y_train_gender, y_test_gender = train_test_split(
    images, ages, genders, test_size=0.2, random_state=42
)

def create_cnn_model():
    input_layer = Input(shape=(shape, shape, 3))

    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(shape, activation='relu')(x)  # testar outras funcoes de ativação
    x = Dropout(0.5)(x)  # testar valores (.5 ~ .2)

    age_output = Dense(1, name='age_output')(x)

    #gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)
    gender_output = Dense(1, activation='tanh', name='gender_output')(x)

    model = Model(inputs=input_layer, outputs=[age_output, gender_output])
    return model

model = create_cnn_model()

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        y_true = K.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (1 - y_pred)
        fl = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(fl)
    return focal_loss_fixed

model.compile(optimizer='adam',
              loss={'age_output': 'mse', 'gender_output': focal_loss()},
              metrics={'age_output': 'mae', 'gender_output': 'accuracy'})

history = model.fit(X_train, {'age_output': y_train_age, 'gender_output': y_train_gender},
                    validation_data=(X_test, {'age_output': y_test_age, 'gender_output': y_test_gender}),
                    epochs=epochs, batch_size=batch_size)  # testar validação cruzada

loss, age_loss, gender_loss, age_mae, gender_accuracy = model.evaluate(
    X_test, {'age_output': y_test_age, 'gender_output': y_test_gender})
print(f"Mean Absolute Error for Age Prediction: {age_mae}")
print(f"Accuracy for Gender Prediction: {gender_accuracy}")

model.save(f'AG\\{model_name}')
print(f'Modelo salvo como "{model_name}".')
