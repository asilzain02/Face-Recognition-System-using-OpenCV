👤 Face Recognition using OpenCV
📌 Overview

This project implements a face recognition system using the face_recognition library (built on dlib) and OpenCV.
The system loads known faces from a dataset folder, detects and recognizes faces in target images, and labels them with names or "Unknown".

📂 Project Structure
<img width="814" height="445" alt="image" src="https://github.com/user-attachments/assets/08303669-4dfe-41c5-bda8-415f0d3e744b" />


⚙️ Installation

Clone the repository or download the project.

git clone (https://github.com/asilzain02/Face-Recognition-System-using-OpenCV.git)

cd Face-Recognition-System-using-OpenCV


Create a virtual environment (optional but recommended).

python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows


Install dependencies.

pip install -r requirements.txt

📦 Requirements

Add this to requirements.txt:

face_recognition
opencv-python
numpy


⚠️ Note: dlib is required by face_recognition. On Windows, install it via:

pip install cmake
pip install dlib

🚀 Usage

Place images of known people in the known_faces/ folder.

File name will be used as the person’s label.

Example: Zain.jpg → Recognized as "Zain".

Place test images in the target/ folder.

Run the script:

python face_recognition_app.py


The program will:

Load known faces.

Detect faces in target images.

Draw bounding boxes & names.

Display the recognized image in a pop-up window.

🖼 Example Output

Input: frnds.jpg

Known dataset: Zain.jpg, Asil.jpg

Output (Displayed Image):

Bounding box around Zain → Labeled as "Zain"

Bounding box around Asil → Labeled as "Asil"

Any unknown face → Labeled as "Unknown"

🔮 Future Enhancements

✅ Real-time face recognition using webcam (cv2.VideoCapture).

✅ Attendance system (log recognized names + timestamps to CSV).

✅ Integrate with a Flask/Django web app for uploading & recognition.

✅ Improve accuracy using SVM/KNN classifier on encodings.

👨‍💻 Author

Developed by Asil Zain ✨
