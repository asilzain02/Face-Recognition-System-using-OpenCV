import face_recognition
import cv2
import os
import numpy as np

def load_known_faces(folder=r"E:\Zain collage file\SEM V\face_recognition-master (1)\face recognition try\image\known_faces"):
    """Loads known faces and their encodings from a folder."""
    known_encodings = []
    known_names = []

    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])  # Use filename as name

    return known_encodings, known_names

def detect_and_recognize_faces(target_image_path, known_encodings, known_names):
    """Detect and recognize faces in the target image."""
    # Load the target image
    image = face_recognition.load_image_file(target_image_path)

    # Detect face locations and encodings in the target image
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # Convert image to RGB (for OpenCV display)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the detected face with known faces
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        
        # Find the best match (if any)
        best_match_index = np.argmin(face_distances) if matches else None

        # Label the face with the best match or "Unknown"
        name = known_names[best_match_index] if matches[best_match_index] else "Unknown"

        # Draw a rectangle around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        # Draw the name below the face
        cv2.putText(image, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the image with the detected faces
    cv2.imshow("Face Recognition", image)
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()  # Close the window

if __name__ == "__main__":
    # Load known faces
    print("Loading known faces...")
    known_encodings, known_names = load_known_faces()

    # Detect and recognize faces in the target image
    print("Processing target image...")
    detect_and_recognize_faces(r"E:\Zain collage file\SEM V\face_recognition-master (1)\face recognition try\image\target\farewell.jpg", known_encodings, known_names)
    detect_and_recognize_faces(r"E:\Zain collage file\SEM V\face_recognition-master (1)\face recognition try\image\target\frnds.jpg", known_encodings, known_names)