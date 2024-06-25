
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('getTop5ViolenceSecteursByMonth', views.get_top5_violence_secteurs_by_month),
    path('getTop5ViolenceSecteur', views.get_top5_violence_secteurs_by_week_day_and_week_month),
    path('getTotalCount', views.get_total_count),
    path('notification', views.notification),

    path('getStatistiquesIndividualSearch', views.getStatistiquesIndividualSearch ),
    path('getStatistiquesViolence', views.getStatistiquesViolence ),

    

]
