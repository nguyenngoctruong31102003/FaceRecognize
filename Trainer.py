import cv2
import os
import numpy as np
from PIL import Image

path = 'Data'
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f)
                  for f in os.listdir(path)]
    faceSamples = []
    ids = []
    
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') #Convert it to grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        
        id = int(os.path.split(imagePath)[-1].split('.')[1])
        faces = face_cascade.detectMultiScale(img_numpy)
        
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    return faceSamples, ids

print('\n [INFO] Đang Training Dữ Liệu ...')
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Khởi tạo 1 Folder Trainer để lưu data đã train
recognizer.write('Trainer/trainer.yml')

print('\n [INFO] Đã có {0} khuôn mặt được train.'.format(len(np.unique(ids))))