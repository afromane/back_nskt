
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
     path('save_camera', views.save_camera),
     path('findAllCamera', views.get_all_camera),
     path('deleteCamera', views.delete_camera),
     path('update_camera', views.update_camera),

     path('save_secteur', views.save_secteur),
     path('update_secteur', views.update_secteur),
     path('findAllSecteur', views.get_all_secteur),
     path('deleteSecteur/<str:id>', views.delete_secteur),

     path('save_contact', views.save_contact),
     path('findAllContact', views.get_all_contact),
     path('deleteContact/<str:id>', views.delete_contact),
     path('update_contact', views.update_contact),


]
