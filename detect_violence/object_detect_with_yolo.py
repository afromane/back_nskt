import cv2
import math
from ultralytics import YOLO

class YoloObjectDetection:
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
    def detect_person(self, img):
        results = self.model(img, stream=True)
        containPerson = False
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = math.ceil(box.conf[0] * 100) / 100
                cls = int(box.cls[0])

                if self.classNames[cls] == "person":
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, self.classNames[cls], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
                    containPerson = True

        return img, containPerson
