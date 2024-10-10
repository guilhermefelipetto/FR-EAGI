import os

import cv2
import face_recognition
import numpy as np

cadastros_dir = "/media/gui/EA0CFB590CFB1F6F/repositories/FR-EAGI/I/cadastros"

rostos_conhecidos = []
nomes_conhecidos = []

for arquivo in os.listdir(cadastros_dir):
    if arquivo.endswith((".jpg", ".jpeg", ".png")):
        nome = os.path.splitext(arquivo)[0]
        imagem = face_recognition.load_image_file(os.path.join(cadastros_dir, arquivo))
        codificacao = face_recognition.face_encodings(imagem)[0]

        rostos_conhecidos.append(codificacao)
        nomes_conhecidos.append(nome)


def reconhecer_faces(frame):
    localizacoes_rostos = face_recognition.face_locations(frame, model="hog")
    codificacoes_rostos = face_recognition.face_encodings(frame, localizacoes_rostos)

    nomes_detectados = []

    for(top, right, bottom, left), face_encoding in zip(localizacoes_rostos, codificacoes_rostos):
        matches = face_recognition.compare_faces(rostos_conhecidos, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(rostos_conhecidos, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = nomes_conhecidos[best_match_index]

        nomes_detectados.append((name, (left, top, right, bottom)))
    
    return nomes_detectados
