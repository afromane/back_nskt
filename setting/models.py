from djongo import models

class Camera(models.Model):
    """
    Modèle pour représenter les flux de caméra en direct.
    """
    _id = models.ObjectIdField(primary_key=True)

    name = models.TextField()
    url = models.TextField()
    secteur = models.TextField()
    longitude = models.TextField( blank=True)
    latitude = models.TextField( blank=True)
    createdAt = models.DateTimeField(auto_now_add=True, blank=True)
    secteur= models.ForeignKey('Secteur', on_delete=models.CASCADE)


    class Meta:
        verbose_name = "Camera Stream"
        verbose_name_plural = "Camera Streams"

class Secteur(models.Model):
    """
    Modèle pour représenter un secteur.
    """
    name = models.TextField()
    _id = models.ObjectIdField(primary_key=True)
    createdAt = models.DateTimeField(auto_now_add=True, blank=True)

    class Meta:
        verbose_name = "Secteur "
        verbose_name_plural = "Secteur "

class ContactUrgence(models.Model):
    """
    Modèle pour représenter un contact d'urgence.
    """
    name = models.TextField()
    responsable = models.TextField()
    email = models.TextField()
    telephone = models.TextField()
    _id = models.ObjectIdField(primary_key=True)
    createdAt = models.DateTimeField(auto_now_add=True, blank=True)

    class Meta:
        verbose_name = "Contact "
        verbose_name_plural = "Contact "
