import os
import cv2
import numpy as np
import concurrent.futures
from datetime import datetime
from collections import deque
import tensorflow as tf
import math
from ultralytics import YOLO

class ViolenceDetectWithYOLO:
    IMAGE_HEIGHT, IMAGE_WIDTH = 96, 96
    SEQUENCE_LENGTH = 24
    CLASSES_LIST = ["NonViolence", "Violence"]
    CLASS_NAME = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]

    def __init__(self, SAVE_TARGET='test_videos', DETECTED_FRAME_DIR="detected_frame", PERSON_FRAME_DIR="person_frame"):
        """
        Initialize the ViolenceDetectAPI with paths to models and directories.
        """
        self.MODEL_VIOLENCE_PATHS = [
            "static/model/seq24/model_0_300_f600.h5",
            "static/model/seq24/model_300_600_f600.h5",
            "static/model/seq24/model_600_900_f600.h5",
        ]
        self.MODEL_VIOLENCE = [tf.keras.models.load_model(model_path) for model_path in self.MODEL_VIOLENCE_PATHS]
        self.DETECTED_FRAME_DIR = DETECTED_FRAME_DIR
        self.PERSON_FRAME_DIR = PERSON_FRAME_DIR

        # Create directories if they do not exist
        os.makedirs(SAVE_TARGET, exist_ok=True)
        os.makedirs(DETECTED_FRAME_DIR, exist_ok=True)
        os.makedirs(PERSON_FRAME_DIR, exist_ok=True)

        # Prepare output file path with current timestamp
        current_time = datetime.now().time()
        formatted_time = current_time.strftime("%H-%M-%S")
        self.output_file_path = f'{SAVE_TARGET}/Output_{formatted_time}.mp4'

        # Initialize YOLO model for person detection
        self.MODEL_YOLO = YOLO("static/objectDetection/yolo/yolov8n.pt")

    def save_frame(self, frame, directory, FRAME_DIR):
        """
        Save a frame to the specified directory.
        """
        directory_path = os.path.join(FRAME_DIR, directory)
        os.makedirs(directory_path, exist_ok=True)
        num_existing_files = len(os.listdir(directory_path))
        filename = f"f{num_existing_files + 1}.jpg"
        filepath = os.path.join(directory_path, filename)
        cv2.imwrite(filepath, frame)
        return filepath

    def predict_frames_parallel(self, video_file_path):
        """
        Predict violence in frames of a video in parallel.
        """
        video_reader = cv2.VideoCapture(video_file_path)
        original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(self.output_file_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                                       video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

        frames_queue = deque(maxlen=self.SEQUENCE_LENGTH)
        temp_frames_queue = deque(maxlen=self.SEQUENCE_LENGTH)
        predicted_class_name = ''
        violence_detection_times = []
        violence_count = 0
        non_violence_count = 0
        DETECTED_FRAME_PATH = ""
        PERSON_FRAME_PATH = ""

        def predict_with_model(model, frames_queue):
            try:
                return model.predict(np.expand_dims(frames_queue, axis=0))[0]
            except Exception as e:
                print(f"Error predicting with model: {model}, {e}")
                return None

        while video_reader.isOpened():
            ok, frame = video_reader.read()
            if not ok:
                break

            temp_frames_queue.append(frame)
            resized_frame = cv2.resize(frame, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH))
            normalized_frame = resized_frame / 255
            frames_queue.append(normalized_frame)

            if len(frames_queue) == self.SEQUENCE_LENGTH:
                all_predicted_probabilities = []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(predict_with_model, model, frames_queue) for model in self.MODEL_VIOLENCE]

                    for future in concurrent.futures.as_completed(futures):
                        predicted_labels_probabilities = future.result()
                        if predicted_labels_probabilities is not None:
                            all_predicted_probabilities.append(predicted_labels_probabilities)

                average_predicted_probabilities = np.mean(all_predicted_probabilities, axis=0)
                predicted_label = np.argmax(average_predicted_probabilities)
                predicted_class_name = self.CLASSES_LIST[predicted_label]

            if predicted_class_name == "Violence":
                violence_count += 1
                cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 12)
                violence_detection_times.append(video_reader.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)  # In seconds
                video_writer.write(frame)
                directory_name = os.path.split(video_file_path)[-1]
                DETECTED_FRAME_PATH = self.save_frame(frame, directory_name,self.DETECTED_FRAME_DIR )
                DETECTED_FRAME_PATH = os.path.dirname(DETECTED_FRAME_PATH)
                PERSON_FRAME_PATH = self.detect_person(frame , directory_name,self.PERSON_FRAME_DIR)
            else:
                non_violence_count += 1
                video_writer.write(frame)

        video_reader.release()
        video_writer.release()
        violence_percentage = (violence_count / (violence_count + non_violence_count)) * 100
        non_violence_percentage = 100 - violence_percentage
        return violence_percentage, non_violence_percentage, violence_detection_times, DETECTED_FRAME_PATH, self.output_file_path,PERSON_FRAME_PATH

    def predict_images(self, images_list):
        """
        Predict violence in a list of images.
        """
        frames_queue = deque(maxlen=self.SEQUENCE_LENGTH)
        predicted_class_name = ''

        def predict_with_model(model, frames_queue):
            try:
                return model.predict(np.expand_dims(frames_queue, axis=0))[0]
            except Exception as e:
                print(f"Error predicting with model: {model}, {e}")
                return None

        all_predicted_probabilities = []
        for frame in images_list:
            resized_frame = cv2.resize(frame, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH))
            normalized_frame = resized_frame / 255
            frames_queue.append(normalized_frame)

            if len(frames_queue) == self.SEQUENCE_LENGTH:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(predict_with_model, model, frames_queue) for model in self.MODEL_VIOLENCE]

                    for future in concurrent.futures.as_completed(futures):
                        predicted_labels_probabilities = future.result()
                        if predicted_labels_probabilities is not None:
                            all_predicted_probabilities.append(predicted_labels_probabilities)

                average_predicted_probabilities = np.mean(all_predicted_probabilities, axis=0)
                predicted_label = np.argmax(average_predicted_probabilities)
                predicted_class_name = self.CLASSES_LIST[predicted_label]

        return predicted_class_name, average_predicted_probabilities

    def detect_person(self, img,directory_name,FRAME_DIR):
        """
        Detect persons in an image using YOLO model and save frames with detected persons.
        """
        results = self.MODEL_YOLO(img, stream=True)
        PERSON_PATH = ""
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if self.CLASS_NAME[int(box.cls[0])] == "person":
                    # Crop the person's bounding box from the original image
                    person_frame = img[y1:y2, x1:x2]
                    # Save the cropped frame
                    path = self.save_frame(person_frame, directory_name,FRAME_DIR )
                    PERSON_PATH = os.path.dirname(path)
        return PERSON_PATH


