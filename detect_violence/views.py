from django.http import HttpResponse, JsonResponse
import os
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
# Create your views here.
from django.core.files.storage import FileSystemStorage
import cv2
from django.http import StreamingHttpResponse,HttpResponseBadRequest
from .detection_api import DectectViolenceAPI # to remove
from .violence_detect_with_yolo import ViolenceDetectWithYOLO
#from .human_detector import DetectorAPI
import numpy as np
import time
from datetime import datetime
from .object_detect_with_yolo import YoloObjectDetection
from .object_detect_with_ssd import SsdObjectDetection
from .models import RecordedVideo,ViolenceEventFromRecordedVideo,ViolenceEventCameraStream
#from camera.models import CameraStream  
from collections import defaultdict
from bson import ObjectId  # Import ObjectId from bson
from setting.models import Camera

""""
        USING YOLO TO INDENTIFY PERSON
"""
@csrf_exempt
def upload_with_yolo(request):
    DETECTED_FRAME_DIR  ='static/detected_frame'
    PERSON_FRAME_DIR  ='static/person_frame'
    SAVE_TARGET = 'static/video_analysis'
    #detector = DectectViolenceAPI(SAVE_TARGET=SAVE_TARGET,DETECTED_FRAME_DIR=DETECTED_FRAME_DIR)
    detector = ViolenceDetectWithYOLO(SAVE_TARGET=SAVE_TARGET,DETECTED_FRAME_DIR=DETECTED_FRAME_DIR,PERSON_FRAME_DIR=PERSON_FRAME_DIR)
    if request.method == 'POST':
        if request.FILES or request.POST:
            video = request.FILES['file']
            upload_dir = 'static/videos/'
            os.makedirs(upload_dir, exist_ok=True)

            fs = FileSystemStorage(location = upload_dir)
            filename = fs.save(video.name,video)
            
            VIDEO_PATH = upload_dir+filename
            analysis_result = detector.predict_frames_parallel(VIDEO_PATH)
            #Save video to database
            recorded_video = RecordedVideo.objects.create(
                name= request.POST.get('name'),
                description=request.POST.get('description'),
                path=upload_dir+filename
            )

            _violence = analysis_result[0]
            _non_violence = analysis_result[1]
            _detection_times  = analysis_result[2]
            _detected_directory  = analysis_result[3]
            _save_directory  = analysis_result[4]
            _person_directory  = analysis_result[5]

            
            violence_event  = ViolenceEventFromRecordedVideo.objects.create(
                violence = _violence,
                non_violence = _non_violence,
                video_stream = recorded_video,
                path_detected = _detected_directory,
                path_analysis_video = _save_directory,
                times_detect = _detection_times,
                path_person =  _person_directory

            )
            latest_event = ViolenceEventFromRecordedVideo.objects.order_by('-createdAt').first()
            print(latest_event._id) 
            return JsonResponse({
                'message': 'Form data received successfully',
                'id' : str(latest_event._id),
                "person" : _person_directory
                }, status=200)
        else:
            return JsonResponse({'error': 'No form data received'}, status=400)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)




def serve_video(request):
    # Chemin vers le fichier vidéo
    #video_path = os.path.join('chemin/vers/votre/dossier/videos', video_name)
    video_name = request.GET.get('video')

    video_path = settings.BASE_DIR+"/"+video_name
    # Vérifie si le fichier vidéo existe
    if os.path.exists(video_path):
        with open(video_path, 'rb') as video_file:
            response = HttpResponse(video_file.read(), content_type='video/mp4')
            response['Content-Disposition'] = f'inline; filename="{video_name}"'
            return response
    else:
        return HttpResponse('La vidéo demandée n\'existe pas', status=404)
def serve_image(request):
    image_path = request.GET.get('image')

    # Vérifie si le fichier image existe
    if os.path.exists(image_path):
        with open(image_path, 'rb') as image_file:
            response = HttpResponse(image_file.read(), content_type='image/jpg')  # Modifier le content_type selon le type d'image
            response['Content-Disposition'] = f'inline; filename="{image_path}"'
            return response
    else:
        return HttpResponse('L\'image demandée n\'existe pas', status=404)

def get_video(request, video_name):
    # Chemin vers le fichier vidéo
    #video_path = os.path.join('chemin/vers/votre/dossier/videos', video_name)
    video_path = settings.BASE_DIR+"/static/videos/"+video_name
    # Vérifie si le fichier vidéo existe
    if os.path.exists(video_path):
        with open(video_path, 'rb') as video_file:
            response = HttpResponse(video_file.read(), content_type='video/mp4')
            response['Content-Disposition'] = f'inline; filename="{video_name}"'
            return response
    else:
        return HttpResponse('La vidéo demandée n\'existe pas', status=404)


def getEventViolenceFromRecordedVideo(request, event_id):
    try:
        event_id = ObjectId(event_id)
        event = ViolenceEventFromRecordedVideo.objects.get(_id=event_id)
        response_data = {
            '_id': str(event._id),
            'violence': event.violence,
            'non_violence': event.non_violence,
            'createdAt': event.createdAt,
            'path_detected' : event.path_detected,
            'path_analysis_video' : event.path_analysis_video,
            'description' : event.video_stream.description,
            'original_video' : event.video_stream.path,
            'original_video_without_path' : os.path.split(event.video_stream.path)[-1],
            'person_path' : event.path_person,
            'times_detect' : event.times_detect 


        }

        return JsonResponse(response_data)
    except ViolenceEventFromRecordedVideo.DoesNotExist:
        return JsonResponse({'error': 'ViolenceEvent not found'}, status=404)

def get_all_violence_events_from_recorded_video(request):
    try:
        events = ViolenceEventFromRecordedVideo.objects.all()
        events_list = []

        for event in events:
            event_details = {
                'id': str(event._id),
                'violence': event.violence,
            'non_violence': event.non_violence,
            'createdAt': event.createdAt,
            'path_detected' : "static/detected_frame/"+event.path_detected,
            'path_analysis_video' : event.path_analysis_video,
            'description' : event.video_stream.description,
            'original_video' : event.video_stream.path,
            'original_video_without_path' : os.path.split(event.video_stream.path)[-1],
            #'times_detect' : event.times_detect 
            }
            events_list.append(event_details)

        return JsonResponse({'events': events_list}, status=200)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def get_all_violence_events_from_camera(request):
    try:
        events = ViolenceEventCameraStream.objects.all()
        events_list = []

        for event in events:
            event_details = {
                'id': str(event._id),
                'violence': event.violence,
            'createdAt': event.createdAt,
            #'path_frame' : "static/detected_frame/"+event.path_frame,
            'times_detect' : event.times_detect,
            'name' : event.camera.name,
            'ip_camera' : event.camera.url,
             'secteur': event.camera.secteur.name if event.camera.secteur else None,
            }
            events_list.append(event_details)

        return JsonResponse({'events': events_list}, status=200)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def getEventViolenceFromCameraById(request, event_id):
    try:
        event_id = ObjectId(event_id)
        event = ViolenceEventCameraStream.objects.get(_id=event_id)
        response_data = {
            '_id': str(event._id),
            'violence': event.violence,
            'createdAt': event.createdAt,
            'path_frame': event.path_frame,
            'camera_name' : event.camera.name ,
            'camera_ip' : event.camera.url ,
            'secteur' : event.camera.secteur.name ,


        }

        return JsonResponse(response_data)
    except ViolenceEventFromRecordedVideo.DoesNotExist:
        return JsonResponse({'error': 'ViolenceEvent not found'}, status=404)


def get_folder_content(request, folder_path):
    directory = "static/detected_frame/"+folder_path
    image_paths = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_paths.append(file_path)

    return JsonResponse({
                'images': image_paths
                }, status=200)


def get_folder_content_with_path(request):
    
    directory = request.GET.get("path")
    image_paths = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_paths.append(file_path)
    return JsonResponse({
                'images': image_paths
                }, status=200)
def get_folders_with_current_date(request):
    base_directory = "static/detected_frame"
    current_date = datetime.now().strftime("%Y-%m-%d")

    current_date_folders = []
    for folder in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder)
        if os.path.isdir(folder_path) and folder.startswith(current_date):
            sub_folders = [os.path.join(folder_path, sub_folder) for sub_folder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, sub_folder))]
            current_date_folders.extend(sub_folders)
        
    return JsonResponse({'folders': current_date_folders}, status=200)

def get_folder_camera_content(request, folder_path):
    current_date = datetime.now().strftime("%Y-%m-%d")

    directory = "static/detected_frame/"+current_date+"/"+folder_path
    image_paths = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_paths.append(file_path)

    return JsonResponse({
                'images': image_paths
                }, status=200)

def getStatistiquePerMonth(request):
   
    # Créer un dictionnaire par défaut pour stocker les statistiques par mois
    stats = defaultdict(int)
    
    results_of_the_day = ViolenceEventFromRecordedVideo.objects.all()
    for result in results_of_the_day:
        mois = result.createdAt.month
        stats[mois] += 1
    """ results_of_the_day = RecordedVideoResult.objects.all()
    for result in results_of_the_day:
        mois = result.createdAt.month
        stats[mois] += 1
     """
    stats_list = [stats[mois] for mois in range(1, 13)]  
    
    # Renvoyer les statistiques
    return JsonResponse(stats_list, safe=False)

def save_frame(frame, directory):
    current_date = datetime.now().strftime("%Y-%m-%d")
    DETECTED_FRAME_DIR="static/detected_frame"
    directory = DETECTED_FRAME_DIR+ '/' +current_date+ "/"+directory
    os.makedirs(directory, exist_ok=True)
    num_existing_files = len(os.listdir(directory))
    current_time = datetime.now().strftime("%H-%M-%S")
    filename = f"{current_time}_{num_existing_files + 1}.jpg"

    filepath = os.path.join(directory, filename)
    cv2.imwrite(filepath, frame)
    return directory
def video_from_camera(request):
    def generate_video(video_url,frame_delay):
        detector = DectectViolenceAPI()
        frames_to_predict = []

        start_time = time.time()
        cap = cv2.VideoCapture(video_url) 
        prediction =0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_to_predict.append(frame)
            if len(frames_to_predict) == 24 :
                # prediction de cadences-frames
                #prediction = 1
                prediction = detector.predict_images(frames_to_predict)
                print(prediction)
                frames_to_predict = []

            
            if prediction is not None :
               #frame = cv2.put(frame, "Predoction : {}".format(prediction),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
               frame = cv2.putText(frame, "Prediction : {}".format(prediction), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        elapsed_time = time.time() - start_time
        wait_time = max(0,frame_delay - elapsed_time)
        time.sleep(wait_time)
                   
    video_url_param =  ""
    video_url = 'http://192.168.100.7:4747/video'
    frame_delay = 1.0/60
    return StreamingHttpResponse(generate_video(video_url,frame_delay), content_type='multipart/x-mixed-replace; boundary=frame') 

def videoFromCameraWithYolo(request):
    detected_frames = []  # Liste pour stocker les frames
    detector = DectectViolenceAPI()
    VIOLENCE_THRESHOLD = 10  # Nombre minimum de détections de violence pour signaler
    DETECTION_INTERVAL = 60  
    def object_detection(frame):
        detect = YoloObjectDetection()
        return detect.detect_person(frame)

    def process_frames_mobilenet(video_url,video_url_param):
        start_time = time.time()  # Temps de départ pour suivre la période de détection
        vs = cv2.VideoCapture(video_url)
        frames_to_predict = []
        violence_count = 0

        while True:
            ret, frame = vs.read()
            if not ret:
                break

            frame = object_detection(frame)
            
            frames_to_predict.append(frame)

            if len(frames_to_predict) == 24:
                prediction = detector.predict_images(frames_to_predict)
                if prediction[0] == "Violence":
                    violence_count += 1

                # Vérifier si la période de détection est écoulée
                elapsed_time = time.time() - start_time
                if elapsed_time >= DETECTION_INTERVAL:
                    # Si plus de cas de violence que le seuil ont été détectés, signaler
                    if violence_count > VIOLENCE_THRESHOLD:
                        #print(f"Plus de {violence_threshold} cas de violence détectés en {detection_interval} secondes.")
                        frame = cv2.putText(frame, "Alert !!!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        #enregistremement des frames
                        detect_path = save_frame(frame,video_url_param)
                        camera = Camera.objects.get(url=video_url_param)
                        current_time = datetime.now().strftime("%H-%M-%S")

                        violence_event  = ViolenceEventCameraStream.objects.create(
                            
                            violence = violence_count,
                            path_frame = detect_path,
                            times_detect = current_time,
                            camera = camera

                        )
                    violence_count = 0

                    start_time = time.time()


                frames_to_predict = []

            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            (flag, encodedImage) = cv2.imencode(".jpg", frame)

            if flag:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')



    video_url_param = request.GET.get('video_url')
    if video_url_param:
        video_url = 'http://' + video_url_param + '/video'
        return StreamingHttpResponse(process_frames_mobilenet(video_url,video_url_param), content_type='multipart/x-mixed-replace; boundary=frame')
    else:
        return HttpResponseBadRequest("Missing or invalid 'video_url' parameter ")

def get_camera_by_id(request):
    event_id = "192.168.100.19:4747"

    try:
        current_date = datetime.now().strftime("%Y-%m-%d")
        DETECTED_FRAME_DIR="static/detected_frame"
        directory = DETECTED_FRAME_DIR+ '/' +current_date+ "/"+event_id
        os.makedirs(directory, exist_ok=True)
        response_data = {
            'path': directory,


        }

        return JsonResponse(response_data)
    except ViolenceEventFromRecordedVideo.DoesNotExist:
        return JsonResponse({'error': 'ViolenceEvent not found'}, status=404)



def videoFromCameraWithSsd(request):
    # Détection d'objets avec MobileNetSSD
    VIOLENCE_THRESHOLD = 10  # Nombre minimum de détections de violence pour signaler
    DETECTION_INTERVAL = 60  # Intervalle de détection en secondes
    violence_predictions = []  # Liste pour stocker les prédictions de violence
    detected_frames = []  # Liste pour stocker les prédictions de violence

    def object_detection(frame):
        detect = SsdObjectDetection()
        return detect.detect_person(frame)

    def process_frames_mobilenet(video_url, frame_delay):
        # Démarrer le flux vidéo à partir de l'URL
        vs = cv2.VideoCapture(video_url)
        detector = DectectViolenceAPI()
        frames_to_predict = []
        violence_count = 0
        start_time = time.time()
        detection_start_time = time.time()

        while True:
            # Lire le cadre actuel du flux vidéo
            ret, frame = vs.read()

            # Vérifier si la lecture du cadre a réussi
            if not ret:
                break

            # Détecter les objets dans le cadre
            frames_to_predict.append(frame)
            frame_with_prediction = frame.copy()  # Copie du cadre pour dessiner les prédictions
            frame = detect_objects_mobilenet(frame)


            if len(frames_to_predict) == 24:
                # Prédiction de cadences-frames
                prediction = detector.predict_images(frames_to_predict)

                # Vérifier si la prédiction est une violence
                if prediction[0] == "Violence":
                    violence_count += 1
                    # Ajouter les frames détectés à la liste
                    detected_frames.extend(frames_to_predict)

                    # Vérifier si le seuil de violence a été atteint dans l'intervalle de détection
                    if violence_count >= VIOLENCE_THRESHOLD:
                        # Calculer le temps écoulé depuis le début de la période
                        elapsed_time = time.time() - detection_start_time

                        # Si le temps écoulé est inférieur à l'intervalle de détection, lancer l'alerte
                        if elapsed_time < DETECTION_INTERVAL:
                            print("Alerte ! Plus de 10 cas de violence détectés en 1 minute.")
                            # Réinitialiser le compteur de violence
                            violence_count = 0
                            # Réinitialiser le temps de début de la période
                            detection_start_time = time.time()
                            # Enregistrer les frames détectées après l'alerte dans un dossier spécifique
                            # save_detected_frames(detected_frames, detection_start_time)

                frames_to_predict = []

                # Dessiner la prédiction sur le cadre
                frame = cv2.putText(frame, "Prediction: {}".format(prediction), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            # Convertir le cadre en format JPEG
            (flag, encodedImage) = cv2.imencode(".jpg", frame)

            # Assurer que l'encodage a réussi
            if flag:
                # Renvoyer le flux d'images encodées en format byte
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

            elapsed_time = time.time() - start_time
            wait_time = max(0, frame_delay - elapsed_time)
            time.sleep(wait_time)

    video_url_param = request.GET.get('video_url')
    if video_url_param:
        frame_delay = 1.0 / 30
        video_url = 'http://' + video_url_param + '/video'
        return StreamingHttpResponse(process_frames_mobilenet(video_url, frame_delay), content_type='multipart/x-mixed-replace; boundary=frame')
    else:
        return HttpResponseBadRequest("Missing or invalid 'video_url' parameter ")



def video_from_camera_fast_detection1(request):
    # Détection d'objets avec MobileNetSSD
    VIOLENCE_THRESHOLD = 10  # Nombre minimum de détections de violence pour signaler
    DETECTION_INTERVAL = 60  # Intervalle de détection en secondes
    violence_predictions = []  # Liste pour stocker les prédictions de violence
    detected_frames = []  # Liste pour stocker les prédictions de violence

    def detect_objects_mobilenet(frame):
        # Charger le modèle pré-entraîné MobileNetSSD
        prototxt_path = settings.BASE_DIR + "/static/MobileNet_SSD/MobileNetSSD_deploy.prototxt.txt"
        model_path = settings.BASE_DIR + "/static/MobileNet_SSD/MobileNetSSD_deploy.caffemodel"
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

        # Prétraiter l'image et l'envoyer à travers le réseau de neurones
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

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

    def process_frames_mobilenet(video_url, frame_delay):
        # Démarrer le flux vidéo à partir de l'URL
        vs = cv2.VideoCapture(video_url)
        detector = DectectViolenceAPI()
        frames_to_predict = []
        violence_count = 0
        start_time = time.time()

        while True:
            # Lire le cadre actuel du flux vidéo
            ret, frame = vs.read()

            # Vérifier si la lecture du cadre a réussi
            if not ret:
                break

            # Détecter les objets dans le cadre
            frames_to_predict.append(frame)
            frame_with_prediction = frame.copy()  # Copie du cadre pour dessiner les prédictions
            frame = detect_objects_mobilenet(frame)

            if len(frames_to_predict) == 24:
                # Prédiction de cadences-frames
                prediction = detector.predict_images(frames_to_predict)

                            
                # Vérifier si la prédiction est une violence
                if prediction[0] == "Violence":
                    violence_count += 1

                    #Ajouter frames detected comme 
                    detected_frames.append(frames_to_predict)

                    # Vérifier si le seuil de violence a été atteint dans l'intervalle de détection
                    if violence_count >= VIOLENCE_THRESHOLD:
                        # Calculer le temps écoulé depuis le début de la période
                        elapsed_time = time.time() - start_time

                        # Si le temps écoulé est inférieur à l'intervalle de détection, lancer l'alerte
                        if elapsed_time < DETECTION_INTERVAL:
                            print("Alerte ! Plus de 10 cas de violence détectés en 1 minute.")
                            # Réinitialiser le compteur de violence
                            violence_count = 0
                            # Réinitialiser le temps de début de la période
                            start_time = time.time()
                
                frames_to_predict = []

                # Dessiner la prédiction sur le cadre
                frame = cv2.putText(frame, "Prediction: {}".format(prediction), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            # Convertir le cadre en format JPEG
            (flag, encodedImage) = cv2.imencode(".jpg", frame)

            # Assurer que l'encodage a réussi
            if flag:
                # Renvoyer le flux d'images encodées en format byte
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

            elapsed_time = time.time() - start_time
            wait_time = max(0, frame_delay - elapsed_time)
            time.sleep(wait_time)

    video_url_param = request.GET.get('video_url')

    if video_url_param:
        frame_delay = 1.0 / 30
        video_url = 'http://' + video_url_param + '/video'
        return StreamingHttpResponse(process_frames_mobilenet(video_url, frame_delay), content_type='multipart/x-mixed-replace; boundary=frame')
    else:
        return HttpResponseBadRequest("Missing or invalid 'video_url' parameter ")


def video_from_camera_precision_detection(request):
    def detect_objects_fast_rnn(image):
        model_path = "/home/modafa-pc/Bureau/violence-detection/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"
        odapi = DetectorAPI(path_to_ckpt=model_path)
        threshold = 0.7
        boxes, scores, classes, num = odapi.processFrame(image)
        person_count = 0
        max_accuracy = 0
        max_avg_accuracy = 0
        acc_sum = 0

        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > threshold:
                person_count += 1
                acc_sum += scores[i]
                if scores[i] > max_accuracy:
                    max_accuracy = scores[i]

                box = boxes[i]
                cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)

        if person_count > 0:
            max_avg_accuracy = acc_sum / person_count

        return image, person_count, max_accuracy, max_avg_accuracy

    def generate_video_fast_rnn(video_url):
        cap = cv2.VideoCapture(video_url)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_with_boxes, person_count, max_accuracy, max_avg_accuracy = detect_objects_fast_rnn(frame)
            
            text = f"P: {person_count}"
             # text = f"Person count: {person_count}, Max accuracy: {max_accuracy}, Max average accuracy: {max_avg_accuracy}"
            cv2.putText(frame_with_boxes, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            _, jpeg = cv2.imencode('.jpg', frame_with_boxes)

            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

    # Obtenir l'URL de la vidéo de la requête GET
    video_url_param =  request.GET.get('video_url')

    # Vérifier si le paramètre 'video_url' est présent
    if video_url_param:
        video_url = 'http://' + video_url_param + '/video'

        # Retourner le flux vidéo avec détection d'objets
        return StreamingHttpResponse(generate_video_fast_rnn(video_url), content_type='multipart/x-mixed-replace; boundary=frame') 
    else:
        # Retourner une réponse BadRequest si le paramètre 'video_url' est manquant
        return HttpResponseBadRequest("Missing or invalid 'video_url' parameter ")


    def detect_objects_fast_rnn(image):
        model_path = "/home/modafa-pc/Bureau/violence-detection/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"
        odapi = DetectorAPI(path_to_ckpt=model_path)
        threshold = 0.7

        boxes, scores, classes, num = odapi.processFrame(image)
        person_count = 0
        max_accuracy = 0
        max_avg_accuracy = 0
        acc_sum = 0

        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > threshold:
                person_count += 1
                acc_sum += scores[i]
                if scores[i] > max_accuracy:
                    max_accuracy = scores[i]

                box = boxes[i]
                cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)

        if person_count > 0:
            max_avg_accuracy = acc_sum / person_count

        return image, person_count, max_accuracy, max_avg_accuracy

    def generate_video_fast_rnn(video_url):
        cap = cv2.VideoCapture(video_url)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_with_boxes, person_count, max_accuracy, max_avg_accuracy = detect_objects_fast_rnn(frame)
            
            text = f"P: {person_count}"
            # text = f"Person count: {person_count}, Max accuracy: {max_accuracy}, Max average accuracy: {max_avg_accuracy}"
            cv2.putText(frame_with_boxes, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            _, jpeg = cv2.imencode('.jpg', frame_with_boxes)

            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

    # Obtenir l'URL de la vidéo de la requête GET
    video_url_param =  request.GET.get('video_url')

    # Vérifier si le paramètre 'video_url' est présent
    if video_url_param:
        video_url = 'http://' + video_url_param + '/video'

        # Retourner le flux vidéo avec détection d'objets
        return StreamingHttpResponse(generate_video_fast_rnn(video_url), content_type='multipart/x-mixed-replace; boundary=frame') 
    else:
        # Retourner une réponse BadRequest si le paramètre 'video_url' est manquant
        return HttpResponseBadRequest("Missing or invalid 'video_url' parameter ")


#suppresion des elment