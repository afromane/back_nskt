from django.http import HttpResponse, JsonResponse,StreamingHttpResponse,HttpResponseBadRequest
import os
from django.views.decorators.csrf import csrf_exempt
import cv2
from .models import RecordedVideo,IndividualSearchFromRecordedVideo,ProfileIndividu,IndividualSearchFromCameraStream
from setting.models import Camera 
from django.db.models import Count
from bson import ObjectId
from django.core.files.storage import FileSystemStorage
from .FaceMatcher import FaceMatcher
import face_recognition
from datetime import datetime, timedelta
from setting.models import Secteur
import json
from django.utils import timezone
def video_from_camera1(request):
    faceMatcher = FaceMatcher()
    
    def generate_video(video_path, reference_image_path, frame_rate=5):
        cap = cv2.VideoCapture(video_path)
        faceMatcher.reference_image = face_recognition.load_image_file(reference_image_path)
        faceMatcher.reference_encoding = face_recognition.face_encodings(faceMatcher.reference_image)[0]

        while True:
            if not ret:
                break

            frame, similarity = faceMatcher.match_faces(frame)
            print(similarity)

            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

    video_path = "http://192.168.100.5:4747/video"
    reference_image_path = "static/profiles/manel.jpeg"
    
    return StreamingHttpResponse(generate_video(video_path, reference_image_path, frame_rate=5), content_type='multipart/x-mixed-replace; boundary=frame')
   
""""
        VIDEO FROM RECORDED
"""

@csrf_exempt
def recordedvideo(request):
    def analysis(video_path, reference_image_path, recorded_video, frame_skip=2):
        faceMatcher = FaceMatcher()
        cap = cv2.VideoCapture(video_path)
        faceMatcher.reference_image = face_recognition.load_image_file(reference_image_path)
        faceMatcher.reference_encoding = face_recognition.face_encodings(faceMatcher.reference_image)[0]

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_base_name = os.path.basename(video_path)
        video_name, _ = os.path.splitext(video_base_name)

        # Ensure the output directory exists
        if not os.path.exists("static/video_analysis"):
            os.makedirs("static/video_analysis")
     
        # Generate the output video path
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_video_path = f"static/video_analysis/{video_name}_{current_time}.mp4"
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264 codc
        out = cv2.VideoWriter(output_video_path, fourcc, fps // frame_skip, (frame_width, frame_height))
        if not out.isOpened():
            print("Error: Could not open VideoWriter.")
            return
    
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                frame, similarities = faceMatcher.match_faces(frame)
                if similarities:
                    
                    frame_path = f'static/person_found/{video_name}'
                    save_frame(frame, frame_path)
                    
                    try:
                        recorded_result = IndividualSearchFromRecordedVideo.objects.get(recorded_video=recorded_video)
                        recorded_result.similarity.extend(similarities)
                        recorded_result.detected_time.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
                        recorded_result.save()
                    except IndividualSearchFromRecordedVideo.DoesNotExist:
                        IndividualSearchFromRecordedVideo.objects.create(
                            similarity=similarities,
                            recorded_video=recorded_video,
                            recognition_path=frame_path,
                            path_video =output_video_path,
                            detected_time=[cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0]

                        )
                
                out.write(frame)

            frame_count += 1

        cap.release()
        out.release()
    if request.method == 'POST':
        if request.FILES or request.POST:
            #upload video file
            video_file= request.FILES['video']
            video_dir = 'static/videos/'
            os.makedirs(video_dir, exist_ok=True)
            fs = FileSystemStorage(location = video_dir)
            filename_video = fs.save(video_file.name,video_file)

            #upload image file
            image_file= request.FILES['image']
            image_dir = 'static/profiles/'
            os.makedirs(image_dir, exist_ok=True)
            fs = FileSystemStorage(location = image_dir)
            filename_image = fs.save(image_file.name,image_file)
            recorded_video  = RecordedVideo.objects.create(
                # name = name,
                description = request.POST.get('description'),
                video_path = video_dir+filename_video,
                image_path = image_dir+filename_image,
            )
            video_path =video_dir+filename_video
            reference_image_path = image_dir+filename_image
            analysis(video_path, reference_image_path, recorded_video,frame_skip=2)
            latest_event = IndividualSearchFromRecordedVideo.objects.order_by('-createdAt').first() 
            return JsonResponse( {
                'message': 'Form data received successfully',
                'id' : str(latest_event._id)
                
                }, status=200)
        else:
            return JsonResponse({'error': 'No form data received'}, status=400)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)

@csrf_exempt
def upload_recorded_video(request):
    if request.method == 'POST':
        if request.FILES or request.POST:
            #upload video file
            video_file= request.FILES['video']
            video_dir = 'static/videos/'
            os.makedirs(video_dir, exist_ok=True)
            fs = FileSystemStorage(location = video_dir)
            filename_video = fs.save(video_file.name,video_file)

            #upload image file
            image_file= request.FILES['image']
            image_dir = 'static/profiles/'
            os.makedirs(image_dir, exist_ok=True)
            fs = FileSystemStorage(location = image_dir)
            filename_image = fs.save(image_file.name,image_file)

             #save to database
            recorded_video  = RecordedVideo.objects.create(
                # name = name,
                description = request.POST.get('description'),
                video_path = video_dir+filename_video,
                image_path = image_dir+filename_image,
            )
            
            latest_event = RecordedVideo.objects.order_by('-createdAt').first()
            return JsonResponse( {
                'message': 'Form data received successfully',
                'id' : str(latest_event._id)
                
                }, status=200)
        else:
            return JsonResponse({'error': 'No form data received'}, status=400)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)

def get_recorded_video_by_id(request,itemId):
    try:
        # Récupérer l'objet Secteur à 
        recorded = RecordedVideo.objects.get(_id=ObjectId(itemId))
        return JsonResponse({'video': 
                             {
                                    "image_path": recorded.image_path,
                                    "video_path": recorded.video_path,
                                    "description": recorded.description
                                    }
                             }, status=200)
    except RecordedVideo.DoesNotExist:
        return JsonResponse({'messge': "Le secteur spécifié n'existe pas."}, status=404)
def save_frame(frame, directory):
        os.makedirs(directory, exist_ok=True)
        num_existing_files = len(os.listdir(directory))
        filename = f"frame{num_existing_files + 1}.jpg"
        filepath = os.path.join(directory, filename)
        cv2.imwrite(filepath, frame)
        return filepath

@csrf_exempt
def analysis_from_recorded_video(request,itemId):
    recorded = RecordedVideo.objects.get(_id=ObjectId(itemId))
    video_path =  recorded.video_path
    reference_image_path = recorded.image_path
    
    
    def generate_video(video_path, reference_image_path, frame_skip=2):
        faceMatcher = FaceMatcher()
        cap = cv2.VideoCapture(video_path)
        faceMatcher.reference_image = face_recognition.load_image_file(reference_image_path)
        faceMatcher.reference_encoding = face_recognition.face_encodings(faceMatcher.reference_image)[0]

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = int(fps / 10)  # Traiter un certain nombre de trames par seconde
        frame_count = 0

            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_base_name = os.path.basename(video_path)
        video_name, _ = os.path.splitext(video_base_name)

        # Ensure the output directory exists
        if not os.path.exists("static/video_analysis"):
            os.makedirs("static/video_analysis")
     
        # Generate the output video path
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_video_path = f"static/video_analysis/{video_name}_{current_time}.mp4"
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264 codc
        out = cv2.VideoWriter(output_video_path, fourcc, fps // frame_skip, (frame_width, frame_height))
        if not out.isOpened():
            print("Error: Could not open VideoWriter.")
            return
    
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            #if frame_count % frame_skip == 0:
            frame, similarities = faceMatcher.match_faces(frame)
            if similarities:
                
                frame_path = f'static/person_found/{video_name}'
                save_frame(frame, frame_path)
                
                try:
                    recorded_result = IndividualSearchFromRecordedVideo.objects.get(recorded_video=recorded)
                    recorded_result.similarity.extend(similarities)
                    recorded_result.detected_time.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
                    recorded_result.save()
                except IndividualSearchFromRecordedVideo.DoesNotExist:
                    IndividualSearchFromRecordedVideo.objects.create(
                        similarity=similarities,
                        recorded_video=recorded,
                        recognition_path=frame_path,
                        path_video =output_video_path,
                        detected_time=[cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0]

                    )
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            out.write(frame)

            frame_count += 1

        cap.release()
        out.release()
    
    return StreamingHttpResponse(generate_video(video_path, reference_image_path,frame_skip=2), content_type='multipart/x-mixed-replace; boundary=frame') 



def getAllResultFromRecordedVideo(request):
    # Récupérer tous les documents avec la date de création égale à aujourd'hui
    items = IndividualSearchFromRecordedVideo.objects.order_by('-createdAt')
    # Créer une liste pour stocker les résultats
    results_list = []
    
    for item in items:
        result_dict = {
            "id": str(item._id),
            'path_video' : item.path_video,
            'recognition_path' : item.recognition_path,
            'description' : item.recorded_video.description,
            'original_image' : item.recorded_video.image_path,
            'original_video' : item.recorded_video.video_path,
            'createdAt' : item.createdAt.strftime("%Y-%m-%d %H:%M:%S"),
            #'detected_time' : item.detected_time,
            'similarity' : item.similarity,


        }
        results_list.append(result_dict)
    
    return JsonResponse(results_list, safe=False)

def getResultFromRecordedVideo(request, event_id):
    try:
        # Retrieve the IndividualSearchFromRecordedVideo event by its _id
        event_id = ObjectId(event_id)
        item = IndividualSearchFromRecordedVideo.objects.get(_id=event_id)
        recognition_frame = os.path.split(item.recognition_path)[-2]
        response_data = {
            '_id': str(item._id),
            'path_video' : item.path_video,
            'recognition_path' : item.recognition_path,
            'description' : item.recorded_video.description,
            'original_image' : item.recorded_video.image_path,
            'original_video' : item.recorded_video.video_path,
            'createdAt' : item.createdAt.strftime("%Y-%m-%d %H:%M:%S"),
            'detected_time' : item.detected_time,
            'similarity' : item.similarity,
        }

        return JsonResponse(response_data)
    except IndividualSearchFromRecordedVideo.DoesNotExist:
        return JsonResponse({'error': 'items not found'}, status=404)

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
def serve_video_with_path(request):
    # Chemin vers le fichier vidéo
    video_path = request.GET.get('path')
    # Vérifie si le fichier vidéo existe
    if os.path.exists(video_path):
        with open(video_path, 'rb') as video_file:
            response = HttpResponse(video_file.read(), content_type='video/mp4')
            response['Content-Disposition'] = f'inline; filename="{video_path}"'
            return response
    else:
        return HttpResponse('La vidéo demandée n\'existe pas', status=404)



@csrf_exempt
def search_individu_by_camera1(request):
    video_url = 'http://'+request.GET.get('video_url')+'/video'
    faceMatcher = FaceMatcher()

    def generate_video(video_url,video_url_param,reference_image_path):
        cap = cv2.VideoCapture(video_url) 
        faceMatcher.reference_image = face_recognition.load_image_file(reference_image_path)
        faceMatcher.reference_encoding = face_recognition.face_encodings(faceMatcher.reference_image)[0]

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame, similarities = faceMatcher.match_faces(frame)
            if similarities:
                current_date = datetime.now().strftime("%Y-%m-%d")
                frame_path = f'static/person_found/{current_date}/{video_url_param}'
                #save_frame(frame, frame_path)
                """
                try:
                    recorded_result = IndividualSearchFromRecordedVideo.objects.get(recorded_video=recorded)
                    recorded_result.similarity.extend(similarities)
                    recorded_result.detected_time.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
                    recorded_result.save()
                except IndividualSearchFromRecordedVideo.DoesNotExist:
                    IndividualSearchFromRecordedVideo.objects.create(
                        similarity=similarities,
                        recorded_video=recorded,
                        recognition_path=frame_path,
                        path_video =output_video_path,
                        detected_time=[cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0]

                    ) """
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    video_url_param = request.GET.get('video_url')
    if video_url_param:
        video_url = 'http://' + video_url_param + '/video'
        reference_image_path = "static/profiles/manel.jpeg"
        # Récupérer tous les individus avec le statut "actif"
        
        return StreamingHttpResponse(generate_video(video_url,video_url_param,reference_image_path), content_type='multipart/x-mixed-replace; boundary=frame')
    else:
        # Gérer le cas où le paramètre video_url est absent ou None
        return HttpResponseBadRequest("Missing or invalid 'video_url' parameter")
@csrf_exempt
def search_individu_by_camera2(request):
    video_url = 'http://'+request.GET.get('video_url')+'/video'

    def generate_video(video_url, video_url_param, individus_actifs):
        cap = cv2.VideoCapture(video_url)
        faceMatcher = FaceMatcher()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            for individu in individus_actifs:
                reference_image_path = individu.path
                faceMatcher.reference_image = face_recognition.load_image_file(reference_image_path)
                faceMatcher.reference_encoding = face_recognition.face_encodings(faceMatcher.reference_image)[0]

                frame, similarities = faceMatcher.match_faces(frame)
                if similarities:
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    reference_name = reference_image_path.split('/')[-1].split('.')[0]
                    frame_path = f'static/person_found/{current_date}/{video_url_param}/{reference_name}'
                    save_frame(frame, frame_path)
                    
                    try:
                        result = IndividualSearchFromCameraStream.objects.get(individu=individu)
                        result.similarity.extend(similarities)
                       # result.detected_time.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
                        result.save() 
                    except IndividualSearchFromCameraStream.DoesNotExist:
                        IndividualSearchFromCameraStream.objects.create(
                             similarity = similarities,
                             path_frame = frame_path,
                             camera = Camera.objects.get(url=video_url_param),
                             individu = individu
                        ) 
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    video_url_param = request.GET.get('video_url')
    if video_url_param:
        video_url = 'http://' + video_url_param + '/video'
        reference_image_paths =[]
        # Récupérer tous les individus avec le statut "actif"
        individus_actifs = ProfileIndividu.objects.filter(status="actif")
        reference_image_paths = [individu.path for individu in individus_actifs]
        
        return StreamingHttpResponse(generate_video(video_url,video_url_param,individus_actifs), content_type='multipart/x-mixed-replace; boundary=frame')
    else:
        # Gérer le cas où le paramètre video_url est absent ou None
        return HttpResponseBadRequest("Missing or invalid 'video_url' parameter")
@csrf_exempt
def search_individu_by_camera(request):
    video_url = 'http://'+request.GET.get('video_url')+'/video'
    def generate_video(video_url, video_url_param, individus_actifs,):
        cap = cv2.VideoCapture(video_url)
        faceMatcher = FaceMatcher()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            for individu in individus_actifs:
                reference_image_path = individu.path
                faceMatcher.reference_image = face_recognition.load_image_file(reference_image_path)
                faceMatcher.reference_encoding = face_recognition.face_encodings(faceMatcher.reference_image)[0]

                frame, similarities = faceMatcher.match_faces(frame)
                if similarities:
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    reference_name = reference_image_path.split('/')[-1].split('.')[0]
                    frame_path = f'static/person_found/{current_date}/{video_url_param}/{reference_name}'
                    save_frame(frame, frame_path)
                    
                    try:
                        result = IndividualSearchFromCameraStream.objects.get(individu=individu)
                        result.similarity.extend(similarities)
                        result.detected_time.append(timezone.now().isoformat())
                        result.save() 
                    except IndividualSearchFromCameraStream.DoesNotExist:
                        IndividualSearchFromCameraStream.objects.create(
                             similarity = similarities,
                             path_frame = frame_path,
                             detected_time = [ timezone.now().isoformat()],
                             camera = Camera.objects.get(url=video_url_param),
                             individu = individu
                        ) 
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    video_url_param = request.GET.get('video_url')
    if video_url_param:
        video_url = 'http://' + video_url_param + '/video'
        # Récupérer tous les individus avec le statut "actif"
        individus_actifs = ProfileIndividu.objects.filter(status="actif")
        return StreamingHttpResponse(generate_video(video_url,video_url_param,individus_actifs), content_type='multipart/x-mixed-replace; boundary=frame')
    else:
        # Gérer le cas où le paramètre video_url est absent ou None
        return HttpResponseBadRequest("Missing or invalid 'video_url' parameter")


@csrf_exempt
def get_cameras_by_secteur_name(request):
    # Get the 'secteur_names' parameter from the request
    secteur_names = request.POST.get('secteur')
    # Check if 'secteur_names' parameter is provided
    if not secteur_names:
        return HttpResponseBadRequest("Missing 'secteur_names' parameter")

    try:
        # Attempt to parse secteur_names as a JSON array
        secteur_names = json.loads(secteur_names)
        if not isinstance(secteur_names, list):
            raise json.JSONDecodeError
    except json.JSONDecodeError:
        # If it's not a valid JSON array, assume it's a comma-separated string
        secteur_names = secteur_names.split(',')

    # Strip whitespace from each sector name
    secteur_names = [name.strip() for name in secteur_names]

    # Retrieve the sectors with the provided names
    secteurs = Secteur.objects.filter(name__in=secteur_names)

    # Retrieve the cameras associated with the retrieved sectors
    cameras = Camera.objects.filter(secteur__in=secteurs)
    
    # Prepare the response data
    cameras_data = [{
        "_id": str(camera._id),
        "name": camera.name,
        "url": camera.url,
        "secteur": camera.secteur.name,
    } for camera in cameras]

    # Return the response data as JSON
    return JsonResponse(cameras_data, safe=False)

def getAllActifSearchFromCameraStream(request):
    # Récupérer tous les individus avec le statut "actif"
    # = ProfileIndividu.objects.filter(status="actif")
    individus_actifs = ProfileIndividu.objects.all()

    # Ensuite, récupérer toutes les entrées dans IndividualSearchFromCameraStream associées à ces individus actifs
    recherche_individus_actifs = IndividualSearchFromCameraStream.objects.filter(individu__in=individus_actifs)

    # Grouper les résultats par l'identifiant unique de l'individu et compter le nombre d'occurrences
    recherche_grouped = recherche_individus_actifs.values('individu').annotate(count=Count('individu'))

    results_list = []
    
    for item in recherche_grouped:
        # Récupérer les détails de l'individu pour chaque groupe
        individu = ProfileIndividu.objects.get(_id=item['individu'])
        
        result_dict = {
            "id": str(individu._id),
            "createdAt": individu.createdAt.strftime("%Y-%m-%d %H:%M:%S"),
            'description': individu.description,
            'path': individu.path,
            'status': individu.status
        }
        results_list.append(result_dict)
    
    return JsonResponse(results_list, safe=False)

def get_searches_by_individu(request, individu_id):
    individu = ProfileIndividu.objects.get(_id=ObjectId(individu_id))

    # Récupérer toutes les entrées de IndividualSearchFromCameraStream pour cet individu
    searches = IndividualSearchFromCameraStream.objects.filter(individu=individu)

    # Préparer la liste des résultats à renvoyer
    results_list = []
    for search in searches:
        result_dict = {
            "id": str(search._id),
            "similarity": search.similarity,
            "createdAt": search.createdAt.strftime("%Y-%m-%d %H:%M:%S"),
            "path_frame": search.path_frame,
            "cameraName": search.camera.name,
            "long": search.camera.longitude,
            "lat": search.camera.latitude,
            "secteur": search.camera.secteur.name,
            "profile" : individu.path,
            "description" : individu.description
        }
        results_list.append(result_dict)

    return JsonResponse(results_list, safe=False)

def count_recent_searches_find(request):
    # Convertir le paramètre 'intervalle' en entier
    intervalle_str = request.GET.get('intervalle')
    try:
        intervalle = int(intervalle_str)
    except (TypeError, ValueError):
        return JsonResponse({"error": "Invalid intervalle parameter"}, status=400)

    # Calculer le temps actuel moins l'intervalle spécifié
    time_limit = timezone.now() - timedelta(seconds=intervalle)

    # Récupérer toutes les instances
    all_searches = IndividualSearchFromCameraStream.objects.all()

    # Initialiser un compteur et une liste pour stocker les informations des individus trouvés
    count = 0
    found_individuals = []

    for search in all_searches:
        # Parcourir chaque detected_time dans chaque instance
        for detected_time_str in search.detected_time:
            if isinstance(detected_time_str, str):
                # Convertir la chaîne de caractères en objet datetime
                try:
                    detected_time = datetime.fromisoformat(detected_time_str)
                    # Comparer avec time_limit
                    if detected_time >= time_limit:
                        count += 1
                        found_individuals.append({
                            "individu_id": str(search.individu._id),
                            "individu_description": search.individu.description,
                            "path": search.individu.path
                        })
                        break  # Nous avons trouvé au moins une correspondance récente pour cette instance
                except ValueError:
                    # Ignorer les chaînes de caractères mal formées
                    continue

    # Retourner le nombre d'éléments trouvés et les informations des individus trouvés
    return JsonResponse({
        "recently": count,
        "found_individuals": found_individuals
    }, safe=False)

"""

@csrf_exempt
def video_from_camera1(request):
    faceMatcher = FaceMatcher()
    
    def generate_video(video_path, reference_image_path,frame_rate=5):
        cap = cv2.VideoCapture(video_path)
        faceMatcher.reference_image = face_recognition.load_image_file(reference_image_path)
        faceMatcher.reference_encoding = face_recognition.face_encodings(faceMatcher.reference_image)[0]

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = int(fps / frame_rate)  # Process specified frames per second
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame,similary = faceMatcher.match_faces(frame)

            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            frame_count += 1

        cap.release      
    video_path = "static/videos/test.mp4"
    reference_image_path = "static/profiles/jaures.jpeg"
    
    return StreamingHttpResponse(generate_video(video_path, reference_image_path,frame_rate=5), content_type='multipart/x-mixed-replace; boundary=frame') 
    if request.method == 'POST':
        if request.FILES or request.POST:
            
            video_file= request.FILES['video']
            video_dir = 'static/videos/'
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir,video_file.name)
            fs = FileSystemStorage(location = video_dir)
            filename = fs.save(video_file.name,video_file)

            #upload image file
            image_file= request.FILES['image']
            image_dir = 'static/profiles/'
            os.makedirs(image_dir, exist_ok=True)
            reference_image_path = os.path.join(image_dir, image_file.name)
            fs = FileSystemStorage(location = image_dir)
            filename = fs.save(image_file.name,image_file) 

            #description = request.POST.get('description')
            return StreamingHttpResponse(generate_video(video_path, reference_image_path,frame_rate=5), content_type='multipart/x-mixed-replace; boundary=frame') 

        else:
            return JsonResponse({'error': 'No form data received'}, status=400)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)"""

def test(request):
    camera1 = Camera.objects.get(_id=ObjectId("6659a04a97f3f56c717998ba"))
    camera2 = Camera.objects.get(_id=ObjectId("6659a06b97f3f56c717998bb"))
    camera3 = Camera.objects.get(_id=ObjectId("6659a0a097f3f56c717998bc"))
    camera4 = Camera.objects.get(_id=ObjectId("6659a0d897f3f56c717998bd"))
    camera5 = Camera.objects.get(_id=ObjectId("6659a11a97f3f56c717998be"))
    camera6 = Camera.objects.get(_id=ObjectId("6659a13f97f3f56c717998bf"))

    ind2Act = ProfileIndividu.objects.get(_id=ObjectId("66549cccaca784d90e3d516f"))


    recorded_video  = IndividualSearchFromCameraStream.objects.create(
        individu = ind2Act,
        camera = camera1

     ) 
    recorded_video  = IndividualSearchFromCameraStream.objects.create(
        individu = ind2Act,
        camera = camera2

     ) 
    recorded_video  = IndividualSearchFromCameraStream.objects.create(
        individu = ind2Act,
        camera = camera3

     )
    
    recorded_video  = IndividualSearchFromCameraStream.objects.create(
        individu = ind2Act,
        camera = camera4

     ) 
    recorded_video  = IndividualSearchFromCameraStream.objects.create(
        individu = ind2Act,
        camera = camera5

     ) 
    recorded_video  = IndividualSearchFromCameraStream.objects.create(
        individu = ind2Act,
        camera = camera6

     )
    return JsonResponse({'error': 'No form data received'}, status=400)

"""

@csrf_exempt
def recordedvideo(request):
    if request.method == 'POST':
        if request.FILES or request.POST:
            
            #upload video file
            video_file= request.FILES['video']
            video_dir = 'static/videos/'
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir,video_file.name)
            fs = FileSystemStorage(location = video_dir)
            filename = fs.save(video_file.name,video_file)

            #upload image file
            image_file= request.FILES['image']
            image_dir = 'static/faces/'
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, image_file.name)
            fs = FileSystemStorage(location = image_dir)
            filename = fs.save(image_file.name,image_file)
            
            threshold = request.POST.get('seuil')
            description = request.POST.get('description')

                         #save to database
            recorded_video  = RecordedVideo.objects.create(
                # name = name,
                description = description,
                video_path = video_path,
                image_path = image_path,
            )
            
            recognizer = FacialRecognition(threshold=threshold)
            image  = cv2.imread(image_path)
            detected_time, similarity_list,output_file_path,RECOGNITION_FRAME_PATH = recognizer.find_image_in_video(image, video_path)
            recorded_result  = IndividualSearchFromRecordedVideo.objects.create(
                threshold = threshold,
                similarity = similarity_list,
                path_video = output_file_path,
                detected_time = detected_time,
                recorded_video = recorded_video,
                recognition_path =RECOGNITION_FRAME_PATH
                        )
            latest_event = IndividualSearchFromRecordedVideo.objects.order_by('-createdAt').first()
            return JsonResponse( {
                'message': 'Form data received successfully',
                
                'id' : str(latest_event._id)
                
                }, status=200)
        else:
            return JsonResponse({'error': 'No form data received'}, status=400)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)




def serve_image(request):
    # Nom du fichier image passé en paramètre GET
    image_name = request.GET.get('image')

    # Chemin complet vers le fichier image
    image_path = os.path.join(settings.BASE_DIR, image_name)

    # Vérifie si le fichier image existe
    if os.path.exists(image_path):
        with open(image_path, 'rb') as image_file:
            response = HttpResponse(image_file.read(), content_type='image/jpeg')  # Modifier le content_type selon le type d'image
            response['Content-Disposition'] = f'inline; filename="{image_name}"'
            return response
    else:
        return HttpResponse('L\'image demandée n\'existe pas', status=404)
@csrf_exempt
def camerastream(request):
    video_url = 'http://'+request.GET.get('video_url')+'/video'
    threshold = request.POST.get('seuil')
    # image_file= request.FILES['image']
    # image_dir = 'static/images/'
    # os.makedirs(image_dir, exist_ok=True)
    # image_path = os.path.join(settings.STATIC_ROOT, 'images', image_file.name)
    # fs = FileSystemStorage(location = image_dir)
    # filename = fs.save(image_file.name,image_file)
    #recognizer = FacialRecognition(threshold=threshold)
    def generate_video(video_url):
        #cap = cv2.VideoCapture(video_url) 
        cap = cv2.VideoCapture(0) 
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame,current_time,similarity= extractor.find_person_in_image(image_source, frame)
            if similarity != 0 :
                image_dir = 'static/images/'
                os.makedirs(image_dir, exist_ok=True)
                filename =image_dir+datetime.now().time()+'.jpg'
                cv2.imwrite(filename, frame)
                recorded_video  = CameraStreamResult.objects.create(
                duration = max_duration,
                threshold = threshold,
                similarity = similarity_list,
                path_image = output_file_path,
                detected_time = detected_time,
                recorded_video = recorded_video
                        ) 

            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    video_url_param = request.GET.get('video_url')
    if video_url_param:
        video_url = 'http://' + video_url_param + '/video'
        return StreamingHttpResponse(generate_video(video_url), content_type='multipart/x-mixed-replace; boundary=frame')
    else:
        # Gérer le cas où le paramètre video_url est absent ou None
        return HttpResponseBadRequest("Missing or invalid 'video_url' parameter")




def getDayFindFromCamera(request):
    # Récupérer tous les documents avec la date de création égale à aujourd'hui
    results_of_the_day = CameraStreamResult.objects.all()
    # Créer une liste pour stocker les résultats
    results_list = []
    
    for result in results_of_the_day:

        if timezone.now().date() ==  result.createdAt.date() :
            result_dict = {
                'threshold': result.threshold,
                'similarity': result.similarity,
                'path_frame': result.path_frame,
                'detected_time': result.detected_time,
                'createdAt': result.createdAt,
            }
            results_list.append(result_dict)
    
    return JsonResponse(results_list, safe=False)

def getAllFromCamera(request):
    # Récupérer tous les documents avec la date de création égale à aujourd'hui
    results_of_the_day = CameraStreamResult.objects.all()
    # Créer une liste pour stocker les résultats
    results_list = []
    
    for result in results_of_the_day:

        result_dict = {
            'threshold': result.threshold,
            'similarity': result.similarity,
            'path_frame': result.path_frame,
            'detected_time': result.detected_time,
            'createdAt': result.createdAt,
        }
        results_list.append(result_dict)
    
    return JsonResponse(results_list, safe=False)
def getStatistiquePerMonth(request):
   
    # Créer un dictionnaire par défaut pour stocker les statistiques par mois
    stats = defaultdict(int)
    
    results_of_the_day = CameraStreamResult.objects.all()
    for result in results_of_the_day:
        mois = result.createdAt.month
        stats[mois] += 1
    results_of_the_day = IndividualSearchFromRecordedVideo.objects.all()
    for result in results_of_the_day:
        mois = result.createdAt.month
        stats[mois] += 1
    
    stats_list = [stats[mois] for mois in range(1, 13)]  
    
    # Renvoyer les statistiques
    return JsonResponse(stats_list, safe=False)



def get_folder_content(request, folder_path):
    directory = "static/recognition_frame/"+folder_path
    image_paths = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_paths.append(file_path)

    return JsonResponse({
                'images': image_paths
                }, status=200) """

""""
       ENDPOINTS TRACKER INDIVIDUAL
"""
""" 
def getAllActifSearchFromCameraStream(request):
    # Récupérer tous les individus avec le statut "actif"
    individus_actifs = ProfileIndividu.objects.filter(status="actif")

    # Ensuite, récupérer toutes les entrées dans IndividualSearchFromCameraStream associées à ces individus actifs
    recherche_individus_actifs = IndividualSearchFromCameraStream.objects.filter(individu__in=individus_actifs)

    # Grouper les résultats par l'identifiant unique de l'individu et compter le nombre d'occurrences
    recherche_grouped = recherche_individus_actifs.values('individu').annotate(count=Count('individu'))

    results_list = []
    
    for item in recherche_grouped:
        # Récupérer les détails de l'individu pour chaque groupe
        individu = ProfileIndividu.objects.get(_id=item['individu'])
        
        result_dict = {
            "id": str(individu._id),
            "createdAt": individu.createdAt.strftime("%Y-%m-%d %H:%M:%S"),
            'description': individu.description,
            'path': individu.path,
        }
        results_list.append(result_dict)
    
    return JsonResponse(results_list, safe=False)

 """

""""
        SAVING PROFILE
"""
@csrf_exempt
def saveProfile(request):
    if request.method == 'POST':
        if request.FILES or request.POST:
            files = request.FILES.getlist('files')
            upload_dir = 'static/profiles/'
            os.makedirs(upload_dir, exist_ok=True)

            fs = FileSystemStorage(location = upload_dir)
            for file in files:
                filename = fs.save(file.name,file)
                profiles  = ProfileIndividu.objects.create(
                description = request.POST.get('description'),
                status = "actif",
                path = upload_dir+filename
            )

            return JsonResponse({
                'message': 'Record successfully',
                "status" :200
                }, status=200)
        else:
            return JsonResponse({'error': 'No form data received'}, status=400)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)

def getAllProfile(request):
    profiles = ProfileIndividu.objects.all()
    results_list = []
    
    for result in profiles:

        result_dict = {
            'status': result.status,
            'description': result.description,
            'path': result.path,
            'createdAt': result.createdAt.strftime("%Y-%m-%d %H:%M:%S"),
            "id": str(result._id),

        }
        results_list.append(result_dict)
    
    return JsonResponse(results_list, safe=False)
def get_image_with_path(request):
  
    image_path = request.GET.get("path")
    if os.path.exists(image_path):
        with open(image_path, 'rb') as image_file:
            # Determine the content type based on the file extension
            extension = os.path.splitext(image_path)[1].lower()
            if extension == '.png':
                content_type = 'image/png'
            elif extension == '.gif':
                content_type = 'image/gif'
            elif extension in ['.jpeg', '.jpg']:
                content_type = 'image/jpeg'
            elif extension == '.bmp':
                content_type = 'image/bmp'
            elif extension == '.webp':
                content_type = 'image/webp'
            else:
                content_type = 'application/octet-stream'  # Default content type for unknown types

            response = HttpResponse(image_file.read(), content_type=content_type)
            response['Content-Disposition'] = f'inline; filename="{os.path.basename(image_path)}"'
            return response
    else:
        return HttpResponse('L\'image demandée n\'existe pas', status=404)

@csrf_exempt
def updateStatus(request):
    if request.method == 'POST':
        if request.POST:
            individu = ProfileIndividu.objects.get(_id=ObjectId(request.POST.get('id')))
            individu.status = request.POST.get('status')

            individu.save()

            return JsonResponse({
                'message': 'Form data received successfully',
                "status" :200
                }, status=200)
        else:
            return JsonResponse({'error': 'No form data received'}, status=400)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)
