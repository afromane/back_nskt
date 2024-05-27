from django.shortcuts import render
import os
from django.views.decorators.csrf import csrf_exempt
import cv2
from django.http import StreamingHttpResponse,HttpResponseBadRequest,FileResponse,HttpResponse, JsonResponse
import time
from datetime import datetime
from .models import Camera,Secteur,ContactUrgence
from djongo.models import ObjectIdField
from bson import ObjectId  # Import ObjectId from bson
import json
@csrf_exempt
def save_camera(request):
    if request.method == 'POST':

        camera = Camera.objects.create(
            name= request.POST.get('name'),
            longitude=request.POST.get('longitude'),
            latitude=request.POST.get('latitude'),
            secteur=Secteur.objects.get(name=request.POST.get('secteur')),
            url=request.POST.get('url'),
        )
        return JsonResponse({
            'message': 'Form data received successfully',
            'code' : 200
            }, status=200)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)

def get_all_camera(request):
    try:
        camera = Camera.objects.all()
        camera_list = []

        for camera in camera:
            camera_details = {
                '_id': str(camera._id),
                'url': camera.url,
                'longitude': camera.longitude,
                'latitude': camera.latitude,
                'secteur': camera.secteur.name,
                'name': camera.name,
                #'description': camera.videostream_ptr.description
            }
            camera_list.append(camera_details)

        return JsonResponse({'camera': camera_list}, status=200)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def delete_camera(request):
    try:
        # Récupérer l'objet Secteur à 
        camera = Camera.objects.get(pk=request.POST.get('id'))
        camera.delete()
        return HttpResponse("Le Camera a été supprimé avec succès.")
    except Secteur.DoesNotExist:
        return HttpResponse("Le Camera spécifié n'existe pas.", status=404)


@csrf_exempt
def save_secteur(request):
    if request.method == 'POST':
        camera = Secteur.objects.create(
            name= request.POST.get('name'),
        )
        return JsonResponse({
            'message': 'Form data received successfully',
            'code' : 200
            }, status=200)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)

def get_all_secteur(request):
    try:
        secteur = Secteur.objects.all()
        secteur_list = []

        for secteur in secteur:
            secteur_details = {
                '_id': str(secteur._id),
                'name': secteur.name,
            }
            secteur_list.append(secteur_details)

        return JsonResponse({'secteur': secteur_list}, status=200)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
def delete_secteur(request):
    try:
        # Récupérer l'objet Secteur à 
        secteur = Secteur.objects.get(pk=request.POST.get('id'))
        secteur.delete()
        return HttpResponse("Le secteur a été supprimé avec succès.")
    except Secteur.DoesNotExist:
        return HttpResponse("Le secteur spécifié n'existe pas.", status=404)


@csrf_exempt
def save_contact(request):
    if request.method == 'POST':

        contact = ContactUrgence.objects.create(
            name= request.POST.get('name'),
            responsable=request.POST.get('responsable'),
            telephone=request.POST.get('telephone'),
            email=request.POST.get('email'),
        )
        return JsonResponse({
            'message': 'Form data received successfully',
            'code' : 200
            }, status=200)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)
@csrf_exempt
def autorite_contact(request):
    if request.method == 'POST':
        contact = ContactUrgence.objects.get(_id=ObjectId(request.POST.get('id')))
        contact.name = request.POST.get('name')
        contact.reponsable = request.POST.get('reponsable')
        contact.telephone = request.POST.get('tel')
        contact.email = request.POST.get('email')

        contact.save()
        return JsonResponse({
            'message': 'Form data received successfully',
            'code' : 200
            }, status=200)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)

def get_all_contact(request):
    try:
        contact = ContactUrgence.objects.all()
        contact_list = []

        for contact in contact:
            contact_details = {
                '_id': str(contact._id),
                'name': contact.name,
                'responsable': contact.responsable,
                'telephone': contact.telephone,
                'email': contact.email
                #'description': contact.videostream_ptr.description
            }
            contact_list.append(contact_details)

        return JsonResponse({'contact': contact_list}, status=200)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
