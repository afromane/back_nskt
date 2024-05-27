import cv2
import face_recognition
from datetime import datetime
import os

class FaceMatcher:
    def __init__(self, distance_threshold=0.5):
        # Initialize the FaceMatcher with a distance threshold
        self.distance_threshold = distance_threshold
        self.output_video_path = "static/video_analysis"  # Path for saving output video
        self.person_found_path = "static/person_found"  # Path for saving output video
    def save_frame(self,frame, directory):
        os.makedirs(directory, exist_ok=True)
        num_existing_files = len(os.listdir(directory))
        filename = f"frame{num_existing_files + 1}.jpg"

        filepath = os.path.join(directory, filename)
        cv2.imwrite(filepath, frame)
        return filepath  
    def match_faces(self, frame):
        # Convert the frame from BGR (OpenCV) to RGB (face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all faces and their encodings in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        #print("Number of faces detected in the frame:", len(face_locations))

        # Extract face encodings
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Create a copy of the frame to draw the results
        output_frame = frame.copy()

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Calculate the Euclidean distance between the face encodings
            distance = 1 - face_recognition.face_distance([self.reference_encoding], face_encoding)[0]

            # Calculer la distance euclidienne entre les encodages de visage
            distance_threshold = self.distance_threshold * 100
            # Convertir la distance en pourcentage
            match_percentage = distance * 100

            # Determine if the face matches the reference image based on the criteria
            if match_percentage > distance_threshold:
                # If the percentage is above the threshold, consider it a match
                match_text = f"Match ({match_percentage:.2f}%)"
                rectangle_color = (0, 255, 0)  # Green

                # Draw a rectangle around the face with the appropriate color
                top, right, bottom, left = face_location
                cv2.rectangle(output_frame, (left, top), (right, bottom), rectangle_color, 2)
                # Display the text indicating if the face matches the reference image
                cv2.putText(output_frame, match_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rectangle_color, 2)
                self.save_frame(output_frame,self.person_found_path)
            else:
                # Otherwise, consider it a non-match
                match_text = f"No Match ({match_percentage:.2f}%)"
                rectangle_color = (0, 0, 255)  # Red

                # Draw a rectangle around the face with the appropriate color
                top, right, bottom, left = face_location
                cv2.rectangle(output_frame, (left, top), (right, bottom), rectangle_color, 2)
                # Display the text indicating if the face matches the reference image
                cv2.putText(output_frame, match_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rectangle_color, 2)


        return output_frame
   
    def search_in_video(self, reference_image_path, video_path):
        # Load the reference image and extract its face encodings
        self.reference_image = face_recognition.load_image_file(reference_image_path)
        self.reference_encoding = face_recognition.face_encodings(self.reference_image)[0]

        # Get the base name of the video file
        video_base_name = os.path.basename(video_path)
        video_name, _ = os.path.splitext(video_base_name)

        # Get the current time for generating the output file name
        formatted_time = datetime.now().time().strftime("%H-%M-%S")

        # Generate the output file name with the current timestamp and original video file name
        self.output_file_path = f'{self.output_video_path}/{video_name}_{formatted_time}.mp4'

        # Open the video for reading
        video_capture = cv2.VideoCapture(video_path)

        # Retrieve video properties
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))

        # Define the codec and create the VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_file_path, fourcc, fps, (frame_width, frame_height))

        # Read the video frame by frame
        while True:
            ret, frame = video_capture.read()

            if not ret:
                break

            # Search for matching faces in the current frame
            output_frame = self.match_faces(frame)
            # Write the frame with the detected faces to the output video
            out.write(output_frame)

        # Release resources
        video_capture.release()
        out.release()


