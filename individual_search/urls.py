
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
   path('test', views.video_from_camera1 ),
    path('recordedvideo', views.recordedvideo ),
    path('uploadrecordedvideo', views.upload_recorded_video ),
    path('getRecordedVideoById/<str:itemId>', views.get_recorded_video_by_id ),
    path('getRecordedVideoById/<str:itemId>', views.get_recorded_video_by_id ),
    path('analysisFromRecordedVideo/<str:itemId>', views.analysis_from_recorded_video ),
    path('getAllResultFromRecordedVideo',views.getAllResultFromRecordedVideo),
    path('getResultFromRecordedVideo/<str:event_id>', views.getResultFromRecordedVideo),
    path('getFolderContentWithPath', views.get_folder_content_with_path),
    path('loadVideoWithPath',views.serve_video_with_path),

    path('searchIndividuBySecteur', views.search_individu_by_secteur ),
    #path('getDayFindFromCamera', views.getDayFindFromCamera ),
    #path('getAllFromCamera', views.getAllFromCamera ),
    #path('getStatistiquePerMonth', views.getStatistiquePerMonth ),
    #path('loadImage',views.serve_image),
    
    #path('getFolderContent/<str:folder_path>', views.get_folder_content),
    path('loadImageWithPath',views.get_image_with_path),

    path('saveProfile', views.saveProfile ),
    path('getAllProfile', views.getAllProfile ),
    path('updateStatus', views.updateStatus ),
    #path('getAllActifSearchFromCameraStream', views.getAllActifSearchFromCameraStream ),
    #path('searchesIndividuById/<str:individu_id>/', views.get_searches_by_individu),



]
