import cv2
import math
from ultralytics import YOLO


class YoloObjectDetection :
    # Classes d'objets
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]
    def __init__(self):
        self.model = YOLO("static/objectDetection/yolo/yolov8n.pt")

    # Effectuer la détection
    def detect_person(self,img):
        results = self.model(img, stream=True)
        # Coordonnées des objets détectés
        containPerson = False
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Boîte englobante
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Dessiner la boîte englobante sur l'image
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100

                # Nom de la classe
                cls = int(box.cls[0])

                # Détails de l'objet
                org = (x1, y1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 1
                #detect just person
                if self.classNames[cls] =="person":
                    cv2.putText(img, self.classNames[cls], org, font, fontScale, color, thickness)
                    containPerson = True
        return img,containPerson
