import mlflow
import mlflow.keras
import tensorflow as tf
import tensorflow.python.keras.backend as K
from keras.api.layers import (Conv2D, Dense, Dropout, Flatten, Input,
                              MaxPooling2D)
from keras.api.models import Model


class UTKFaceModel:
    """
    A class used to represent a UTKFace Model for age and gender prediction.

    Attributes
    ----------
    img_shape : int
        The shape of the input images (default is 128).
    model_name : str
        The name of the model file (default is 'utkface_model_keras.h5').
    model : keras.Model
        The Keras model instance.

    Methods
    -------
    - create_model() :
        Creates and returns the Keras model.
    
    - focal_loss(gamma=2., alpha=0.25) :
        Returns a focal loss function with the specified gamma and alpha.
    
    - compile_model() :
        Compiles the Keras model with specified loss functions and metrics.
    
    - train(X_train, y_train_age, y_train_gender, X_test, y_test_age, y_test_gender, epochs=70, batch_size=32) :
        Trains the model on the provided training data and logs parameters and metrics to MLflow.
    
    - evaluate(X_test, y_test_age, y_test_gender) :
        Evaluates the model on the provided test data and logs final metrics to MLflow.
    
    - save_model() :
        Saves the model to a specified path and logs the model to MLflow.
    """
    
    def __init__(self, img_shape=128, model_name='utkface_model_keras.h5'):
        self.img_shape = img_shape
        self.model_name = model_name
        self.model = self.create_model()
    
    def create_model(self):
        """
        Creates a convolutional neural network model for predicting age and gender.
        The model consists of several convolutional layers followed by max-pooling layers,
        a flattening layer, and dense layers. The model has two outputs: one for age prediction
        and one for gender prediction.
        Returns:
            model (tf.keras.Model): A Keras Model instance with two outputs: 'age_output' and 'gender_output'.
        """
        
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
        """
        Creates a focal loss function with specified gamma and alpha parameters.
        Focal loss is designed to address class imbalance by down-weighting easy examples
        and focusing training on hard negatives. It is particularly useful for object detection tasks.
        Args:
            gamma (float, optional): Focusing parameter. Default is 2.0.
            alpha (float, optional): Balancing parameter. Default is 0.25.
        Returns:
            function: A loss function that computes the focal loss between `y_true` and `y_pred`.
        """
        
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
        """
        Compiles the model with specified optimizer, loss functions, and metrics.
        Uses 'adam' optimizer, 'mse' loss for age output, focal loss for gender output,
        and evaluates 'mae' for age output and 'accuracy' for gender output.
        """
        
        self.model.compile(optimizer='adam',
                           loss={'age_output': 'mse', 'gender_output': self.focal_loss()},
                           metrics={'age_output': 'mae', 'gender_output': 'accuracy'})
    
    def train(self, X_train, y_train_age, y_train_gender, X_test, y_test_age, y_test_gender, epochs=70, batch_size=32):
        """
        Trains the model using the provided training and testing data, logs parameters and metrics to MLflow.
        Args:
            X_train (numpy.ndarray): Training data features.
            y_train_age (numpy.ndarray): Training data labels for age.
            y_train_gender (numpy.ndarray): Training data labels for gender.
            X_test (numpy.ndarray): Testing data features.
            y_test_age (numpy.ndarray): Testing data labels for age.
            y_test_gender (numpy.ndarray): Testing data labels for gender.
            epochs (int, optional): Number of epochs to train the model. Defaults to 70.
            batch_size (int, optional): Size of the batches of data. Defaults to 32.
        Returns:
            keras.callbacks.History: A record of training loss values and metrics values at successive epochs.
        """
        
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
        """
        Evaluate the model on the test data and log the results.
        Args:
            X_test (numpy.ndarray): Test data.
            y_test_age (numpy.ndarray): True age labels for the test data.
            y_test_gender (numpy.ndarray): True gender labels for the test data.
        Returns:
            tuple: Mean Absolute Error for age prediction and accuracy for gender prediction.
        """
        
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
