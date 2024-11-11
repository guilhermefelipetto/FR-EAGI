# FR-EAGI

## Description
FR-EAGI Face Recognition - Emotions, Age, Gender and Individual

The project aims to develop a facial recognition system that identifies emotions, age, gender and specific indivuals.

#### Emotions
Emotions Model is currently based on ResNet-50.
We used the FER2013 database to train the CNN.

#### Age and Gender
Model for Age and Gender is a simple Neural Network assembled with 2D convolutional layers followed by MaxPooling2D layers.
As a loss function, a focal loss function was implemented to compile the CNN model.

#### Individual
A simpler algorithm that uses the face_recognition library to detect faces, the registered faces are in the "registrations" folder, where the model "learns" the patterns of the faces and can recognize them.

## Functionalities
- Functionality 1: Emotion Recognition
- Functionality 2: Age prediction and Gender recognition
- Functionality 3: Individual Recognition

## Contribution
Feel free to open pull-requests.
