from djongo import models
from djongo.models.fields import JSONField
from setting.models import Camera

class RecordedVideo(models.Model):
    """
    Modèle pour représenter les enregistrements vidéo.
    """
    _id = models.ObjectIdField(primary_key=True)
    name = models.CharField(max_length=100, blank=True)
    description = models.TextField(blank=True)
    createdAt = models.DateTimeField(auto_now_add=True, blank=True)
    path = models.TextField()
    class Meta:
        verbose_name = "Recorded Video"
        verbose_name_plural = "Recorded Videos"

class ViolenceEventFromRecordedVideo(models.Model):
    """
    Modèle de base pour les evenements violente dans les enregistrements vidéo.
    """
    _id = models.ObjectIdField(primary_key=True)
    violence = models.FloatField()  
    non_violence = models.FloatField(blank=True)
    path_detected = models.TextField(blank=True)
    path_analysis_video = models.TextField(blank=True)
    path_person = models.TextField(blank=True)
    times_detect = JSONField(default=list)
    createdAt = models.DateTimeField(auto_now_add=True, blank=True)
    video_stream = models.ForeignKey('RecordedVideo', on_delete=models.CASCADE)

    class Meta:
        verbose_name = "Violence Event from recorded video"
        verbose_name_plural = "Violence Events from recorded video"


class ViolenceEventCameraStream(models.Model):
    """
    Modèle de base pour les evenements violente.
    """
    _id = models.ObjectIdField(primary_key=True)
    violence = models.FloatField()  
    path_frame = models.TextField(blank=True)
    times_detect = models.TextField()  
    createdAt = models.DateTimeField(auto_now_add=True, blank=True)
    #video_stream = models.ForeignKey(CameraStream, on_delete=models.CASCADE)
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE)

    class Meta:
        verbose_name = "Violence Event camera strem"
        verbose_name_plural = "Violence Events camera stream"
