
from djongo import models
from djongo.models.fields import JSONField
from setting.models import Camera

class RecordedVideo(models.Model):
    """
    Modèle pour représenter les enregistrements vidéo.
    """
    _id = models.ObjectIdField(primary_key=True)
    name = models.CharField(max_length=100, blank=True)
    description = models.TextField()
    createdAt = models.DateTimeField(auto_now_add=True, blank=True)

    image_path = models.TextField()
    video_path = models.TextField()

    class Meta:
        verbose_name = "Recorded Video"
        verbose_name_plural = "Recorded Videos"



class IndividualSearchFromRecordedVideo(models.Model):
    """
    Modèle de base pour enregistrer les recherche issu des des enregistrement videos.
    """
    _id = models.ObjectIdField(primary_key=True)
    similarity = JSONField(default=list)
    path_video = models.TextField(blank=True)
    recognition_path = models.TextField(blank=True)
    detected_time = JSONField(default=list)
    createdAt = models.DateTimeField(auto_now_add=True, blank=True)
    recorded_video = models.ForeignKey('RecordedVideo', on_delete=models.CASCADE)

    class Meta:
        verbose_name = "Recorded Video Result"
        verbose_name_plural = "Recorded Video Results"

class ProfileIndividu(models.Model):
    """
    Modèle pour représenter les enregistrements vidéo.
    """
    _id = models.ObjectIdField(primary_key=True)
    status = models.CharField(max_length=100, blank=True)
    description = models.TextField()
    createdAt = models.DateTimeField(auto_now_add=True, blank=True)

    path = models.TextField()

    class Meta:
        verbose_name = "Profile individu"
        verbose_name_plural = "Profiles Individu"

class IndividualSearchFromCameraStream(models.Model):
    """
    Modèle de base pour enregistrer les recherche issu des camera.
    """
    _id = models.ObjectIdField(primary_key=True)
    similarity = JSONField(default=list)
    detected_time = JSONField(default=list)
    path_frame = models.TextField(blank=True)
    createdAt = models.DateTimeField(auto_now_add=True, blank=True)
    individu = models.ForeignKey('ProfileIndividu', on_delete=models.CASCADE)
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE)

    class Meta:
        verbose_name = "Individual search from camera stream"
        verbose_name_plural = "Individual search from camera stream"
