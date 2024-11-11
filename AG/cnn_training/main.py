"""
This script performs the following tasks:
1. Starts an MLflow server and UI as subprocesses.
2. Loads the UTKFace dataset using the UTKFaceDataLoader class.
3. Splits the dataset into training and testing sets.
4. Initializes and compiles a UTKFaceModel.
5. Trains the model on the training data.
6. Evaluates the model on the testing data.
7. Saves the trained model to a file.
8. Terminates the MLflow server and UI subprocesses.

Modules:
    subprocess: To create and manage additional processes.
    time: To introduce delays in the script.

Classes:
    UTKFaceDataLoader: A class to load and preprocess the UTKFace dataset.
    UTKFaceModel: A class to define, compile, train, evaluate, and save a Keras model.

Variables:
    server_process (subprocess.Popen): Process running the MLflow server.
    mlflow_process (subprocess.Popen): Process running the MLflow UI.
    dataset_path (str): Path to the UTKFace dataset.
    data_loader (UTKFaceDataLoader): Instance of UTKFaceDataLoader for loading data.
    X_train (numpy.ndarray): Training data features.
    X_test (numpy.ndarray): Testing data features.
    y_train_age (numpy.ndarray): Training data labels for age.
    y_test_age (numpy.ndarray): Testing data labels for age.
    y_train_gender (numpy.ndarray): Training data labels for gender.
    y_test_gender (numpy.ndarray): Testing data labels for gender.
    model (UTKFaceModel): Instance of UTKFaceModel for the neural network.
"""

import subprocess
import time

from utkface_data_loader import UTKFaceDataLoader
from utkface_model import UTKFaceModel

server_process = subprocess.Popen(['python3', 'AG/mlflow/server.py'])
mlflow_process = subprocess.Popen(['mlflow', 'ui', '--backend-store-uri', 'AG/mlflow/mlruns'])
time.sleep(5)

dataset_path = 'datasets/UTKFace'
data_loader = UTKFaceDataLoader(dataset_path)
data_loader.load_data()
X_train, X_test, y_train_age, y_test_age, y_train_gender, y_test_gender = data_loader.train_test_split()

model = UTKFaceModel(img_shape=128, model_name='model_utkface_keras.h5')
model.compile_model()
model.train(X_train, y_train_age, y_train_gender, X_test, y_test_age, y_test_gender, epochs=2, batch_size=32)
model.evaluate(X_test, y_test_age, y_test_gender)
model.save_model()

server_process.terminate()
mlflow_process.terminate()
