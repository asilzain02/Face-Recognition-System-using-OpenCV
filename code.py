import face_recognition
import cv2
import os
import numpy as np

def load_images_from_folder(folder):
    images = []
    names = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            images.append(encodings[0])
            names.append(os.path.splitext(filename)[0])  # Save name without extension
    return images, names

def recognize_faces(known_encodings, known_names):
    for filename in os.listdir(r"E:\Zain collage file\SEM V\face_recognition-master (1)\face recognition try\image\unknown_faces"):
        img_path = os.path.join(r"E:\Zain collage file\SEM V\face_recognition-master (1)\face recognition try\image\unknown_faces", filename)
        unknown_image = face_recognition.load_image_file(img_path)
        unknown_encodings = face_recognition.face_encodings(unknown_image)

        # Detect faces and compare with known encodings
        for unknown_encoding in unknown_encodings:
            results = face_recognition.compare_faces(known_encodings, unknown_encoding)
            face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)

            # Find the closest match
            best_match_index = np.argmin(face_distances)
            if results[best_match_index]:
                print(f"{filename}: Match found - {known_names[best_match_index]}")
            else:
                print(f"{filename}: No match found")

if __name__ == "__main__":
    # Load known images and their encodings
    print("Loading known images...")
    known_encodings, known_names = load_images_from_folder(r"E:\Zain collage file\SEM V\face_recognition-master (1)\face recognition try\image\known_faces")

    # Recognize faces in unknown images
    print("Recognizing faces in unknown images...")
    recognize_faces(known_encodings, known_names)
