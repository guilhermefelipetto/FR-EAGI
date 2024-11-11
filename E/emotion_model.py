import cv2
import numpy as np
from keras.api.models import load_model


class EmotionDetector:
    """
    A class used to detect emotions from an image frame using a pre-trained model.
    
    Attributes
    ----------
    model : keras.Model
        The pre-trained emotion detection model.
    emotion_labels : list of str
        The list of emotion labels corresponding to the model's output.
    Methods
    -------
    detect_emotion(frame)
        Detects the predominant and secondary emotions from the given image frame.
    
        
        Parameters
        ----------
        model_path : str
            The file path to the pre-trained model.
        
        Detects the predominant and secondary emotions from the given image frame.
        
        Returns
        -------
        tuple of str
            The predominant and secondary emotions detected in the image frame.
        
        """

    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def detect_emotion(self, frame):
        """
        Detects the predominant and secondary emotions from a given video frame.
        Args:
            frame (numpy.ndarray): The input video frame in BGR format.
        Returns:
            tuple: A tuple containing the predominant and secondary emotions as strings.
        """
        
        img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_array = cv2.resize(img_array, (48, 48))
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)
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
