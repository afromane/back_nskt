import os
import cv2
import numpy as np
import concurrent.futures
from datetime import datetime
from collections import deque
import tensorflow as tf

class DectectViolenceAPI:
    IMAGE_HEIGHT, IMAGE_WIDTH = 96, 96
    SEQUENCE_LENGTH = 24
    CLASSES_LIST = ["NonViolence", "Violence"]

    def __init__(self, save_target='test_videos',detected_frame_dir="detected_frame"):
        self.model_list = [
            "static/model/seq24/model_0_300_f600.h5",
            "static/model/seq24/model_300_600_f600.h5",
            "static/model/seq24/model_600_900_f600.h5",
            #"static/model/seq24/model_900_1153_f506.h5",
        ]

        self.model_list = [tf.keras.models.load_model(model_path) for model_path in self.model_list]
        self.detected_frame_dir = detected_frame_dir
        os.makedirs(save_target, exist_ok=True)
        os.makedirs(detected_frame_dir, exist_ok=True)

        current_time = datetime.now().time()
        formatted_time = current_time.strftime("%H-%M-%S")
        self.output_file_path = f'{save_target}/Output_{formatted_time}.mp4'

    def save_frame(self,frame, directory):
        directory = self.detected_frame_dir+ '/' + directory
        os.makedirs(directory, exist_ok=True)
        num_existing_files = len(os.listdir(directory))
        filename = f"frame{num_existing_files + 1}.jpg"

        filepath = os.path.join(directory, filename)
        cv2.imwrite(filepath, frame)
        return filepath  


    def predict_frames_parallel(self, video_file_path):
        video_reader = cv2.VideoCapture(video_file_path)
        original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(self.output_file_path, cv2.VideoWriter_fourcc(*'mp4v'), video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))
        frames_queue = deque(maxlen=self.SEQUENCE_LENGTH)
        temp_frames_queue = deque(maxlen=self.SEQUENCE_LENGTH)
        predicted_class_name = ''
        violence_detection_times = []
        violence_count = 0
        non_violence_count = 0
        DETECTED_FRAME_PATH=""
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
            all_predicted_probabilities = []

            if len(frames_queue) == self.SEQUENCE_LENGTH:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(predict_with_model, model, frames_queue) for model in self.model_list]

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
                violence_detection_times.append(video_reader.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)  # En secondes
                video_writer.write(frame)
                directory_name =  os.path.split(video_file_path)[-1]
                DETECTED_FRAME_PATH = self.save_frame(frame, directory_name )


            else:
                non_violence_count += 1
                video_writer.write(frame)

        video_reader.release()
        video_writer.release()
        violence_percentage = (violence_count / (violence_count + non_violence_count)) * 100
        non_violence_percentage = 100 - violence_percentage
        return violence_percentage,non_violence_percentage,violence_detection_times,DETECTED_FRAME_PATH,self.output_file_path

    def predict_images(self, images_list):

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
                  futures = [executor.submit(predict_with_model, model, frames_queue) for model in self.model_list]

                  for future in concurrent.futures.as_completed(futures):
                      predicted_labels_probabilities = future.result()
                      if predicted_labels_probabilities is not None:
                          all_predicted_probabilities.append(predicted_labels_probabilities)

              average_predicted_probabilities = np.mean(all_predicted_probabilities, axis=0)
              predicted_label = np.argmax(average_predicted_probabilities)
              predicted_class_name = self.CLASSES_LIST[predicted_label]

      return predicted_class_name,average_predicted_probabilities
	
