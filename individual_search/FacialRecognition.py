import cv2
from mtcnn.mtcnn import MTCNN
import dlib
from datetime import datetime
import numpy as np
import os


class FacialRecognition:
    def __init__(self,threshold = 50):
        #shape_predictor_path="/static/faceRecognition/shape_predictor_68_face_landmarks.dat"
        #shape_predictor_path="/home/modafa-pc/Bureau/violence-detection/program/api-violence-detection/static/faceRecognition/shape_predictor_68_face_landmarks.dat"
        shape_predictor_path="static/faceRecognition/shape_predictor_68_face_landmarks.dat"
        #shape_predictor_path = os.path.join(settings.STATIC_ROOT, 'faceRecognition', "shape_predictor_68_face_landmarks.dat")
        # shape_predictor_path = os.path.join(settings.STATIC_ROOT, 'faceRecognition', "shape_predictor_68_face_landmarks.dat")
        self.detector = MTCNN()
        self.predictor = dlib.shape_predictor(shape_predictor_path)
        self.output_video_path = "static/video_analysis"
        self.recognition_frame_dir ="static/recognition_frame"
        # Seuil de similarité
        self.threshold = threshold
        self.metrics={}


    def calculate_face_metrics(self,facial_landmarks):
        # Initialiser un dictionnaire pour stocker les mesures
        metrics = {}
        # Calculer la largeur et la hauteur du visage
        face_width = facial_landmarks[16][0] - facial_landmarks[0][0]
        face_height = facial_landmarks[8][1] - facial_landmarks[27][1]

        # Calculer la distance entre les yeux
        eye_distance = facial_landmarks[45][0] - facial_landmarks[36][0]

        # Calculer la largeur de l'œil gauche
        left_eye_width = facial_landmarks[39][0] - facial_landmarks[36][0]

        # Calculer la largeur de l'œil droit
        right_eye_width = facial_landmarks[45][0] - facial_landmarks[42][0]

        # Calculer la distance verticale entre le nez et les yeux
        nose_to_eyes_distance = np.abs(facial_landmarks[27][1] - (facial_landmarks[39][1] + facial_landmarks[42][1]) / 2)

        # Calculer le sinus en utilisant la distance verticale et la distance horizontale entre les yeux
        sinus = nose_to_eyes_distance / eye_distance

        # Calculer la largeur de la bouche
        mouth_width = facial_landmarks[54][0] - facial_landmarks[48][0]

        # Stocker les mesures dans le dictionnaire
        metrics['face_width'] = face_width
        metrics['face_height'] = face_height
        metrics['eye_distance'] = eye_distance
        metrics['left_eye_width'] = left_eye_width
        metrics['right_eye_width'] = right_eye_width
        metrics['nose_to_eyes_distance'] = nose_to_eyes_distance
        metrics['sinus'] = sinus
        metrics['mouth_width'] = mouth_width

        return metrics

    def calculate_similarity(self,metrics1, metrics2):
        #Methode dictance ecuclidienne
        # Convertir les dictionnaires de métriques en vecteurs numpy
        vec1 = np.array(list(metrics1.values()))
        vec2 = np.array(list(metrics2.values()))

        # Calculer la distance euclidienne entre les deux vecteurs
        distance = np.linalg.norm(vec1 - vec2)

        return 100 -distance

    def get_orginal_image_metrics(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detect_faces(image)
        face_metrics_list = []
        for face in faces:
            x, y, width, height = face['box']
            if width > 0 and height > 0:
                margin = 20
                landmarks = self.predictor(gray, dlib.rectangle(left=max(0, x-margin), top=max(0, y-margin), right=min(x+width+margin, image.shape[1]), bottom=min(y+height+margin, image.shape[0])))
                facial_landmarks = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]
                metrics = self.calculate_face_metrics(facial_landmarks)

                face_metrics_list.append(metrics)

        return face_metrics_list

    def detect_and_extract_facial_landmarks(self, metrics, image, current_time):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detect_faces(image)
        facial_landmarks_list = []
        face_metrics_list = []
        time_detected=None
        for face in faces:
            x, y, width, height = face['box']

            if width > 0 and height > 0:
                margin = 20
                landmarks = self.predictor(gray, dlib.rectangle(left=max(0, x-margin), top=max(0, y-margin), right=min(x+width+margin, image.shape[1]), bottom=min(y+height+margin, image.shape[0])))
                facial_landmarks = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]
                metrics2 = self.calculate_face_metrics(facial_landmarks)

                similarity = self.calculate_similarity(metrics,metrics2)
                print(similarity)

                if similarity >= float(self.threshold):
                    cv2.rectangle(image, (x, y), (x+width, y+height), ( 0, 255, 0), 2)
                else:
                    similarity = None
                    current_time = None
                    cv2.rectangle(image, (x, y), (x+width, y+height), (0, 0, 255), 2)
                return image,current_time,similarity



    def find_image_in_video(self, image, video_path):
        source_metrics = self.get_orginal_image_metrics(image)

        current_time = datetime.now().time()
        formatted_time = current_time.strftime("%H-%M-%S")
        # Générer le nom de fichier de sortie avec l'horodatage actuel
        output_file_path = f'{self.output_video_path}/Output_{formatted_time}.mp4'
        detected_time = []
        similarity_list = []
        frames_list = []
        RECOGNITION_FRAME_PATH=""

        # Ouvrir la vidéo originale
        cap = cv2.VideoCapture(video_path)

        # Vérifier si la vidéo est ouverte correctement
        if not cap.isOpened():
            print("Erreur: Impossible d'ouvrir la vidéo.")
            return
        # Récupérer les propriétés de la vidéo originale
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Convertir la durée maximale en nombre de trames
        """  if max_duration == -1:
            max_frames = float('inf')  # Rechercher sur toute la vidéo
        else:
            max_frames = int(max_duration * fps) """
        # Créer un objet VideoWriter pour écrire la nouvelle vidéo
        out = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        # Parcourir les trames de la vidéo originale
        idx = 0
        #while cap.isOpened() and idx < max_frames:
        while cap.isOpened() :
            ret, frame = cap.read()

            if ret:
                # Récupérer les caractéristiques de la trame
                result = self.detect_and_extract_facial_landmarks(source_metrics[0], frame, datetime.now())
                # image,_time,similarity = self.detect_and_extract_facial_landmarks(source_metrics[0], frame, datetime.now())
                if result is not None:
                    image,_time,similarity = result
                    if similarity is not None :
                        directory_name =  os.path.split(video_path)[-1]
                        RECOGNITION_FRAME_PATH = self.save_frame(image,directory_name)
                        detected_time.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
                        similarity_list.append(similarity)
                out.write(image)
                frames_list.append(image)

            else:
                break

            idx += 1
      
        # Libérer les ressources
        cap.release()
        out.release()

        return detected_time, similarity_list,output_file_path,RECOGNITION_FRAME_PATH

    def find_person_in_image(self, image, target_image):
        source_metrics = self.get_orginal_image_metrics(image)

        result = self.detect_and_extract_facial_landmarks(source_metrics[0], target_image, datetime.now().time())

        return  result
    def save_frame(self,frame, directory):
        directory = self.recognition_frame_dir+ '/' + directory
        os.makedirs(directory, exist_ok=True)
        num_existing_files = len(os.listdir(directory))
        filename = f"frame{num_existing_files + 1}.jpg"

        filepath = os.path.join(directory, filename)
        cv2.imwrite(filepath, frame)
        return filepath  



