import cv2
import numpy as np
from keras.api.models import load_model

from E.emotion_model import EmotionDetector
from I.face_detection import recognize_faces

from dotenv import load_dotenv
from os import getenv

if __name__ == "__main__":
    load_dotenv()
    
    # Age predict and gender recognition model
    model_path = getenv('AGE_GENDER_MODEL_PATH')
    model = load_model(model_path, compile=False)

    def preprocess_image(face):
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
