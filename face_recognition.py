import face_recognition
import cv
import os
import numpy as np

# Load known faces
def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(known_faces_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Load image
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)

            # Get face encoding
            face_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)

            # Get the name from the filename (remove extension)
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)

    return known_face_encodings, known_face_names

# Recognize faces in the given image
def recognize_faces_in_image(image_path, known_face_encodings, known_face_names):
    # Load image
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Find faces and encodings in the image
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    # Initialize lists for recognized names and rectangles
    face_names = []
    face_rectangles = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Find the best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)
        face_rectangles.append((top, right, bottom, left))

    return face_names, face_rectangles

# Draw rectangles around recognized faces
def draw_face_rectangles(image_path, face_names, face_rectangles):
    image = cv2.imread(image_path)
    for (top, right, bottom, left), name in zip(face_rectangles, face_names):
        # Draw a rectangle around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with the name below the face
        cv2.putText(image, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Save the result
    output_image_path = "recognized_faces.jpg"
    cv2.imwrite(output_image_path, image)
    print(f"Results saved to {output_image_path}")

if __name__ == "__main__":
    known_faces_dir = "known_faces"  # Directory with known faces
    unknown_image_path = "unknown.jpg"  # Path to the image to recognize faces in

    # Load known faces
    known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

    # Recognize faces in the unknown sss
    face_names, face_rectangles = recognize_faces_in_image(unknown_image_path, known_face_encodings, known_face_names)

    # Draw rectangles and names on the image
    draw_face_rectangles(unknown_image_path, face_names, face_rectangles)
