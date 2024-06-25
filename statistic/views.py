from django.http import JsonResponse
from datetime import datetime,timedelta
from detect_violence.models import RecordedVideo as RecordedVideoViolence,ViolenceEventFromRecordedVideo,ViolenceEventCameraStream
from individual_search.models import RecordedVideo as RecordedVideoSearch,IndividualSearchFromRecordedVideo,ProfileIndividu,IndividualSearchFromCameraStream
from collections import defaultdict
from setting.models import Camera,Secteur,ContactUrgence
from django.core.mail import send_mail

def send_notification_email(subject, message, recipient_list):
    send_mail(
        subject,
        message,
        'issamanel05@gmail.com',  # L'adresse e-mail de l'expéditeur
        recipient_list,
        fail_silently=False,
    )


def notification(request):

    send_notification_email('Sujet de l\'e-mail', 'Contenu du message', ['issamanel05@gmail.com'])
    return JsonResponse({'succeess': "cool"}, status=200)


def get_top5_violence_secteurs_by_month(request):
    try:
        current_year = datetime.now().year
        events = ViolenceEventCameraStream.objects.filter(createdAt__year=current_year)

        # Dictionnaire pour stocker les données de violence par secteur et par mois
        violence_count_by_secteur_month = defaultdict(lambda: defaultdict(int))

        # Collecter les données de violence par secteur et par mois
        for event in events:
            if event.camera.secteur:
                secteur_name = event.camera.secteur.name
                month = event.createdAt.month
                #violence_count_by_secteur_month[secteur_name][month] += 1
                violence_count_by_secteur_month[secteur_name][month] += event.violence

        # Préparer les données pour chaque secteur
        secteur_data_list = []

        for secteur, month_counts in violence_count_by_secteur_month.items():
            secteur_data = {'name': secteur, 'data': []}
            for month in range(1, 13):
                secteur_data['data'].append(month_counts.get(month, 0))
            secteur_data_list.append(secteur_data)

        # Trier les secteurs en fonction du nombre total d'incidents de violence
        sorted_secteurs = sorted(secteur_data_list, key=lambda x: sum(x['data']), reverse=True)

        # Sélectionner les cinq premiers secteurs
        top5_secteurs = sorted_secteurs[:5]

        return JsonResponse({'top5_secteurs': top5_secteurs}, status=200)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def get_total_count(request):
    try:
        return JsonResponse(
            {
                'camera': Camera.objects.count(),
                'secteur': Secteur.objects.count(),
                'contact': ContactUrgence.objects.count(),
                'individuRecherche' : ProfileIndividu.objects.filter(status="actif").count(),
                'recorded_video': RecordedVideoViolence.objects.count() + RecordedVideoSearch.objects.count(),
                'analysis_video': ViolenceEventFromRecordedVideo.objects.count() + IndividualSearchFromRecordedVideo.objects.count(),
                }, status=200)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def getStatistiquesIndividualSearch(request):
    # Récupérer la date d'aujourd'hui
    today = datetime.now().date()
    
    # Calculer les dates du début et de la fin de la semaine en cours
    start_of_week = today - timedelta(days=(today.weekday() + 1) % 7)
    end_of_week = start_of_week + timedelta(days=6)
    
    # Créer un dictionnaire par défaut pour stocker les statistiques par jour de la semaine
    stats_by_day = defaultdict(int)
    
    # Récupérer les résultats de la recherche d'individus à partir des flux de caméra
    results_of_the_week = IndividualSearchFromCameraStream.objects.filter(createdAt__range=(start_of_week, end_of_week))
    for result in results_of_the_week:
        day_of_week = result.createdAt.weekday()
        stats_by_day[day_of_week] += 1
    
    # Récupérer les résultats de la recherche d'individus à partir des vidéos enregistrées
    results_of_the_week = IndividualSearchFromRecordedVideo.objects.filter(createdAt__range=(start_of_week, end_of_week))
    for result in results_of_the_week:
        day_of_week = result.createdAt.weekday()
        stats_by_day[day_of_week] += 1
    
    # Créer une liste ordonnée des statistiques par jour de la semaine
    stats_by_day_list = [stats_by_day[i] for i in range(7)]
    
    # Calculer les dates du début et de la fin du mois en cours
    start_of_month = today.replace(day=1)
    if start_of_month.month == 12:
        end_of_month = start_of_month.replace(year=start_of_month.year + 1, month=1)
    else:
        end_of_month = start_of_month.replace(month=start_of_month.month + 1)
    end_of_month -= timedelta(days=1)
    
    # Créer un dictionnaire par défaut pour stocker les statistiques par semaine du mois
    stats_by_week = defaultdict(int)
    
    # Récupérer les résultats de la recherche d'individus à partir des flux de caméra
    results_of_the_month = IndividualSearchFromCameraStream.objects.filter(createdAt__range=(start_of_month, end_of_month))
    for result in results_of_the_month:
        week_of_month = (result.createdAt.date() - start_of_week).days // 7 + 1
        if week_of_month > 4:
            week_of_month = 4
        stats_by_week[week_of_month] += 1
    
    # Récupérer les résultats de la recherche d'individus à partir des vidéos enregistrées
    results_of_the_month = IndividualSearchFromRecordedVideo.objects.filter(createdAt__range=(start_of_month, end_of_month))
    for result in results_of_the_month:
        week_of_month = (result.createdAt.date() - start_of_week).days // 7 + 1
        if week_of_month > 4:
            week_of_month = 4
        stats_by_week[week_of_month] += 1
    
    # Créer une liste ordonnée des statistiques par semaine du mois
    stats_by_week_list = [stats_by_week[i] for i in range(1, 5)]
    
    # Calculer les statistiques par mois de l'année en cours
    start_of_year = today.replace(month=1, day=1)
    end_of_year = start_of_year.replace(year=start_of_year.year + 1)
    end_of_year -= timedelta(days=1)
    
    stats_by_month = defaultdict(int)
    
    results_of_the_year = IndividualSearchFromCameraStream.objects.filter(createdAt__range=(start_of_year, end_of_year))
    for result in results_of_the_year:
        month_of_year = result.createdAt.month
        stats_by_month[month_of_year] += 1
    
    results_of_the_year = IndividualSearchFromRecordedVideo.objects.filter(createdAt__range=(start_of_year, end_of_year))
    for result in results_of_the_year:
        month_of_year = result.createdAt.month
        stats_by_month[month_of_year] += 1
    
    stats_by_month_list = [stats_by_month[i] for i in range(1, 13)]
    
    # Renvoyer les statistiques
    return JsonResponse({'weeks': stats_by_day_list, 'month': stats_by_week_list, 'year': stats_by_month_list}, safe=False)

def getStatistiquesViolence(request):
    # Récupérer la date d'aujourd'hui
    today = datetime.now().date()
    
    # Calculer les dates du début et de la fin de la semaine en cours
    start_of_week = today - timedelta(days=(today.weekday() + 1) % 7)
    end_of_week = start_of_week + timedelta(days=6)
    
    # Créer un dictionnaire par défaut pour stocker les statistiques par jour de la semaine
    stats_by_day = defaultdict(int)
    
    # Récupérer les résultats de la recherche d'individus à partir des flux de caméra
    results_of_the_week = ViolenceEventCameraStream.objects.filter(createdAt__range=(start_of_week, end_of_week))
    for result in results_of_the_week:
        day_of_week = result.createdAt.weekday()
        stats_by_day[day_of_week] += 1
    
    # Récupérer les résultats de la recherche d'individus à partir des vidéos enregistrées
    results_of_the_week = ViolenceEventFromRecordedVideo.objects.filter(createdAt__range=(start_of_week, end_of_week))
    for result in results_of_the_week:
        day_of_week = result.createdAt.weekday()
        stats_by_day[day_of_week] += 1
    
    # Créer une liste ordonnée des statistiques par jour de la semaine
    stats_by_day_list = [stats_by_day[i] for i in range(7)]
    
    # Calculer les dates du début et de la fin du mois en cours
    start_of_month = today.replace(day=1)
    if start_of_month.month == 12:
        end_of_month = start_of_month.replace(year=start_of_month.year + 1, month=1)
    else:
        end_of_month = start_of_month.replace(month=start_of_month.month + 1)
    end_of_month -= timedelta(days=1)
    
    # Créer un dictionnaire par défaut pour stocker les statistiques par semaine du mois
    stats_by_week = defaultdict(int)
    
    # Récupérer les résultats de la recherche d'individus à partir des flux de caméra
    results_of_the_month = ViolenceEventCameraStream.objects.filter(createdAt__range=(start_of_month, end_of_month))
    for result in results_of_the_month:
        week_of_month = (result.createdAt.date() - start_of_week).days // 7 + 1
        if week_of_month > 4:
            week_of_month = 4
        stats_by_week[week_of_month] += 1
    
    # Récupérer les résultats de la recherche d'individus à partir des vidéos enregistrées
    results_of_the_month = ViolenceEventFromRecordedVideo.objects.filter(createdAt__range=(start_of_month, end_of_month))
    for result in results_of_the_month:
        week_of_month = (result.createdAt.date() - start_of_week).days // 7 + 1
        if week_of_month > 4:
            week_of_month = 4
        stats_by_week[week_of_month] += 1
    
    # Créer une liste ordonnée des statistiques par semaine du mois
    stats_by_week_list = [stats_by_week[i] for i in range(1, 5)]
    
    # Calculer les statistiques par mois de l'année en cours
    start_of_year = today.replace(month=1, day=1)
    end_of_year = start_of_year.replace(year=start_of_year.year + 1)
    end_of_year -= timedelta(days=1)
    
    stats_by_month = defaultdict(int)
    
    results_of_the_year = ViolenceEventCameraStream.objects.filter(createdAt__range=(start_of_year, end_of_year))
    for result in results_of_the_year:
        month_of_year = result.createdAt.month
        stats_by_month[month_of_year] += 1
    
    results_of_the_year = ViolenceEventFromRecordedVideo.objects.filter(createdAt__range=(start_of_year, end_of_year))
    for result in results_of_the_year:
        month_of_year = result.createdAt.month
        stats_by_month[month_of_year] += 1
    
    stats_by_month_list = [stats_by_month[i] for i in range(1, 13)]
    
    # Renvoyer les statistiques
    return JsonResponse({'weeks': stats_by_day_list, 'month': stats_by_week_list, 'year': stats_by_month_list}, safe=False)



def get_top5_violence_secteurs_by_week_day_and_week_month(request):
    try:
        current_date = datetime.now()
        current_year = current_date.year
        current_month = current_date.month
        current_week_day = current_date.weekday() # 0 = Lundi, 6 = Dimanche

        # Calculer les dates de début et de fin de la semaine courante
        if current_week_day == 0:
            start_date = current_date
        else:
            start_date = current_date - timedelta(days=current_week_day)
        end_date = start_date + timedelta(days=6)

        # Récupérer les événements du mois courant
        start_of_month = datetime(current_year, current_month, 1)
        if current_month == 12:
            end_of_month = datetime(current_year + 1, 1, 1) - timedelta(days=1)
        else:
            end_of_month = datetime(current_year, current_month + 1, 1) - timedelta(days=1)
        events = ViolenceEventCameraStream.objects.filter(createdAt__gte=start_of_month, createdAt__lte=end_of_month)

        # Dictionnaire pour stocker les données de violence par secteur, par jour de la semaine et par semaine du mois
        violence_count_by_secteur_week_day_week_month = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))

        # Collecter les données de violence par secteur, par jour de la semaine et par semaine du mois
        for event in events:
            if event.camera.secteur:
                secteur_name = event.camera.secteur.name
                week_day = event.createdAt.weekday()
                week_of_month = (event.createdAt.day - 1) // 7 + 1  # Commencer à 1 pour la première semaine
                violence_count_by_secteur_week_day_week_month[secteur_name][week_day][week_of_month - 1] += event.violence

        # Préparer les données pour chaque secteur
        stats_by_day_list = []
        stats_by_week_list = []
        stats_by_month_list = []

        for secteur, week_day_data in violence_count_by_secteur_week_day_week_month.items():
            secteur_stats_by_day = {'name': secteur, 'data': []}
            secteur_stats_by_week = {'name': secteur, 'data': []}
            secteur_stats_by_month = {'name': secteur, 'data': []}

            # Données par jour de la semaine
            for week_day in range(7):
                secteur_stats_by_day['data'].append(sum(week_day_data[week_day]))
            stats_by_day_list.append(secteur_stats_by_day)

            # Données par semaine du mois
            for week in range(4):
                secteur_stats_by_week['data'].append(sum(week_day_data[week]))
            stats_by_week_list.append(secteur_stats_by_week)

            # Données par mois
            secteur_stats_by_month['data'].append(sum(sum(week_day_data[week_day]) for week_day in range(7)))
            stats_by_month_list.append(secteur_stats_by_month)

        return JsonResponse({'weeks': stats_by_day_list, 'month': stats_by_week_list, 'year': stats_by_month_list}, safe=False)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)