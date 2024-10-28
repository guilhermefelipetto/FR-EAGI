import tensorflow as tf
import tensorflow.python.keras.backend as K
from keras.api.layers import (Conv2D, Dense, Dropout, Flatten, Input,
                              MaxPooling2D)
from keras.api.models import Model

import mlflow
import mlflow.keras


class UTKFaceModel:
    def __init__(self, img_shape=128, model_name='utkface_model_keras.h5'):
        self.img_shape = img_shape
        self.model_name = model_name
        self.model = self.create_model()
    
    def create_model(self):
        input_layer = Input(shape=(self.img_shape, self.img_shape, 3))
        x = Conv2D(32, (3, 3), activation='relu')(input_layer)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(self.img_shape, activation='relu')(x)
        x = Dropout(0.5)(x)

        age_output = Dense(1, name='age_output')(x)
        gender_output = Dense(1, activation='tanh', name='gender_output')(x)

        model = Model(inputs=input_layer, outputs=[age_output, gender_output])
        return model
    
    @staticmethod
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
    
    def compile_model(self):
        self.model.compile(optimizer='adam',
                           loss={'age_output': 'mse', 'gender_output': self.focal_loss()},
                           metrics={'age_output': 'mae', 'gender_output': 'accuracy'})
    
    def train(self, X_train, y_train_age, y_train_gender, X_test, y_test_age, y_test_gender, epochs=70, batch_size=32):
        with mlflow.start_run(nested=True):
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)

        history = self.model.fit(
            X_train, {'age_output': y_train_age, 'gender_output': y_train_gender},
            validation_data=(X_test, {'age_output': y_test_age, 'gender_output': y_test_gender}),
            epochs=epochs, batch_size=batch_size
        )

        for epoch, (val_age_mae, val_gender_accuracy) in enumerate(zip(history.history['val_age_output_mae'], history.history['val_gender_output_accucary'])):
            mlflow.log_metric("val_age_mae", val_age_mae, step=epoch)
            mlflow.log_metric("val_gender_accuracy", val_gender_accuracy, step=epoch)

        return history
    
    def evaluate(self, X_test, y_test_age, y_test_gender):
        loss, age_loss, gender_loss, age_mae, gender_accuracy = self.model.evaluate(
            X_test, {'age_output': y_test_age, 'gender_output': y_test_gender})
        
        mlflow.log_metric("final_age_mae", age_mae)
        mlflow.log_metric("final_gender_accuracy", gender_accuracy)

        print(f"Mean Absolute Error for Age Prediction: {age_mae}")
        print(f"Accuracy for Gender Prediction: {gender_accuracy}")
        return age_mae, gender_accuracy
    
    def save_model(self):
        mlflow.keras.log_model(self.model, self.model_name)
        self.model.save(f"AG\\models\\{self.model_name}")
        print(f'Model saved as {self.model_name}.')