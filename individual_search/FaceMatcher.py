import cv2
import face_recognition

class FaceMatcher:
    def __init__(self, reference_image_path,distance_threshold=0.5):
        # Charger l'image de référence et en extraire les encodages de visage
        self.reference_image = face_recognition.load_image_file(reference_image_path)
        self.reference_encoding = face_recognition.face_encodings(self.reference_image)[0]
        self.distance_threshold = distance_threshold=0.5

    def match_faces(self, target_image_path):
        # Charger l'image cible en utilisant OpenCV pour conserver la couleur originale
        target_image = cv2.imread(target_image_path)

        # Convertir l'image cible de BGR (OpenCV) à RGB (face_recognition)
        rgb_target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

        # Trouver tous les visages et leurs encodages dans l'image cible
        face_locations = face_recognition.face_locations(rgb_target_image)
        print("Nombre de visages détectés dans l'image cible :", len(face_locations))

        # Extraire les encodages de visage
        face_encodings = face_recognition.face_encodings(rgb_target_image, face_locations)

        # Créer une copie de l'image cible pour dessiner les résultats
        output_image = target_image.copy()

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Calculer la distance euclidienne entre les encodages de visage
            distance = 1 - face_recognition.face_distance([self.reference_encoding], face_encoding)[0]

            # Convertir la distance en pourcentage
            match_percentage = distance * 100

            # Déterminer si le visage correspond à l'image de référence en fonction des critères
            if match_percentage > distance_threshold:
                # Si le pourcentage est supérieur au seuil, considérer le visage comme une correspondance
                match_text = f"Match ({match_percentage:.2f}%)"
                rectangle_color = (0, 255, 0)  # Vert
            else:
                # Sinon, considérer le visage comme une non-correspondance
                match_text = f"No Match ({match_percentage:.2f}%)"
                rectangle_color = (0, 0, 255)  # Rouge

            # Encadrer le visage avec la couleur appropriée
            top, right, bottom, left = face_location
            cv2.rectangle(output_image, (left, top), (right, bottom), rectangle_color, 2)
            # Afficher le texte indiquant si le visage correspond ou non à l'image de référence
            cv2.putText(output_image, match_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rectangle_color, 2)

        return output_image

# Utilisation de la classe FaceMatcher
matcher = FaceMatcher("../adel.jpeg")
output_image = matcher.match_faces("../adel.jpeg")
cv2.imwrite("output_image.jpg", output_image)
