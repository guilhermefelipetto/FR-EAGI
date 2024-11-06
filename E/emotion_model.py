import numpy as np
from keras.api.models import load_model
import cv2

class EmotionDetector:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def detect_emotion(self, frame):
        img_array = cv2.resize(frame, (224, 224))
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0

        def values(lista):
            if len(lista) < 2:
                return -1, -1

            index1 = np.argmax(lista)
            lista[index1] = -np.inf
            index2 = np.argmax(lista)

            return index1, index2

        predictions = self.model.predict(img_array)
        idx_predominant, idx_secondary = values(predictions[0])
        predominant = self.emotion_labels[idx_predominant]
        secondary = self.emotion_labels[idx_secondary]

        return predominant, secondary
