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
