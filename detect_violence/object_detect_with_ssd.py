import cv2
from django.conf import settings
import numpy as np


class SsdObjectDetection :

    def __init__(self):
        MODEL_PATH = settings.BASE_DIR+"/static/objectDetection/"
        prototxt_path = MODEL_PATH + "MobileNet_SSD/MobileNetSSD_deploy.prototxt.txt"
        model_path = MODEL_PATH + "MobileNet_SSD/MobileNetSSD_deploy.caffemodel"
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # Effectuer la détection
    def detect_person(self,frame):
        # Prétraiter l'image et l'envoyer à travers le réseau de neurones
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        # Traiter les détections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filtrer les détections avec une confiance minimale
            if confidence > 0.2:
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                # Dessiner la boîte englobante et le label sur l'image
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        return frame
    