"""
This script captures video from the webcam, detects faces, predicts age and gender, and recognizes emotions.

Modules:
    cv2: OpenCV library for video capture and image processing.
    numpy: Library for numerical operations on arrays.
    keras.api.models: Module to load pre-trained Keras models.
    dotenv: Module to load environment variables from a .env file.
    os: Module to interact with the operating system.
    
Functions:
    preprocess_image(face): Preprocesses the input face image for model prediction.
    
Main Execution:
    - Loads environment variables.
    - Loads the age and gender prediction model.
    - Captures video from the webcam.
    - Detects faces in the video frames.
    - For each detected face:
        - Preprocesses the face image.
        - Predicts age and gender using the pre-trained model.
        - Recognizes emotions using the EmotionDetector.
        - Draws rectangles around detected faces.
        - Annotates the video frame with the predicted age, gender, and emotions.
    - Displays the annotated video frames in a window.
    - Exits the video capture loop when the 'q' key is pressed.
"""

from os import getenv

import cv2
import numpy as np
from dotenv import load_dotenv
from keras.api.models import load_model

from E.emotion_model import EmotionDetector
from I.face_detection import recognize_faces

if __name__ == "__main__":
    load_dotenv()
    
    # Age predict and gender recognition model
    model_path = getenv('AGE_GENDER_MODEL_PATH')
    model = load_model(model_path, compile=False)

    def preprocess_image(face):
        """
        Preprocesses the input face image for model prediction.
        This function resizes the input face image to 128x128 pixels, normalizes the pixel values to the range [0, 1],
        and expands the dimensions to add a batch dimension.
        Parameters:
        face (numpy.ndarray): The input face image.
        Returns:
            numpy.ndarray: The preprocessed face image ready for model prediction.
        """
        
        face = cv2.resize(face, (128, 128))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)
        return face

    # Video capture
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    # Emotions Model
    detector = EmotionDetector(getenv('EMOTIONS_MODEL_PATH'))

    while True:
        ret, frame = video_capture.read()

        # Individual model
        nomes_detectados = recognize_faces(frame)

        for name, (left, top, right, bottom) in nomes_detectados:
            face = frame[top:bottom, left:right]
            if face.size == 0:
                continue

            preprocess_face = preprocess_image(face)
            predictions = model.predict(preprocess_face)

            predicted_age = predictions[0][0][0]
            predicted_gender = 'Masculino' if predictions[1][0][0] < 0.5 else 'Feminino'
            
            predominant_emotion, secondary_emotion = detector.detect_emotion(frame)

            # Draw rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw a label with a name below the face
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw labels to the age and gender
            cv2.putText(frame, f"Idade: {predicted_age:.2f}", (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Genero: {predicted_gender}", (left, top - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw labels to the emotions
            cv2.putText(frame, f"Emocao predominante: {predominant_emotion}", (left, top - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            #cv2.putText(frame, f"Emocao secundaria: {secondary_emotion}", (left, top - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
