
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    #path('test', views.home ),
    path('recordedvideo', views.recordedvideo ),
    #path('camerastream', views.camerastream ),
    #path('getDayFindFromCamera', views.getDayFindFromCamera ),
    #path('getAllFromCamera', views.getAllFromCamera ),
    #path('getStatistiquePerMonth', views.getStatistiquePerMonth ),
     #path('getResultFromRecordedVideo/<str:event_id>', views.getResultFromRecordedVideo),
    #path('loadVideo',views.serve_video),
    #path('loadImage',views.serve_image),
    
    #path('getAllResultFromRecordedVideo',views.getAllResultFromRecordedVideo),
    #path('getFolderContent/<str:folder_path>', views.get_folder_content),
    path('loadImageWithPath',views.get_image_with_path),

    path('saveProfile', views.saveProfile ),
    path('getAllProfile', views.getAllProfile ),
    path('updateStatus', views.updateStatus ),
    #path('getAllActifSearchFromCameraStream', views.getAllActifSearchFromCameraStream ),
    #path('searchesIndividuById/<str:individu_id>/', views.get_searches_by_individu),



]
