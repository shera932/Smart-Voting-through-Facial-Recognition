import face_recognition as fr  # Library for face recognition tasks
import cv2  # OpenCV library for computer vision tasks
import numpy as np  # NumPy for numerical operations
from firebase_admin import credentials, storage, initialize_app
import os

# Initialize Firebase app
cred = credentials.Certificate("credentials.json")
initialize_app(cred, {
    'storageBucket': 'flaskwebapp-e3844.appspot.com'
})

# Reference to the default Firebase Storage bucket
bucket = storage.bucket()

# List of Firebase Storage paths of images
firebase_storage_paths = ["Images/Pushpendra.jpg", "Images/Devanand.jpg"]  # Add more paths as needed

# Initialize lists to store known names and their corresponding encodings
known_names = []
known_name_encodings = []

# Load images directly from Firebase Storage and store encodings
for firebase_path in firebase_storage_paths:
    blob = bucket.blob(firebase_path)

    # Read the image from the Firebase Storage blob
    image_bytes = blob.download_as_bytes()
    image_np = cv2.imdecode(np.asarray(bytearray(image_bytes), dtype="uint8"), cv2.IMREAD_COLOR)

    # Extract face encoding for the first face in the image
    encoding = fr.face_encodings(image_np)[0]

    # Append the encoding and corresponding name to the lists
    known_name_encodings.append(encoding)
    known_names.append(os.path.splitext(os.path.basename(firebase_path))[0].capitalize())

# Display the known names
print(known_names)

# Open a video capture object for the default camera (index 2)
cap = cv2.VideoCapture(0)

# Start an infinite loop for capturing and processing video frames
while True:
    # Read a frame from the video capture object
    ret, frame = cap.read()

    # Find face locations and encodings in the current frame
    face_locations = fr.face_locations(frame)
    face_encodings = fr.face_encodings(frame, face_locations)

    # Iterate through detected faces and their encodings
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare face encodings with known face encodings
        matches = fr.compare_faces(known_name_encodings, face_encoding)
        name = ""

        # Calculate face distances and find the best match
        face_distances = fr.face_distance(known_name_encodings, face_encoding)
        best_match = np.argmin(face_distances)

        # If a match is found, set the name to the corresponding known name
        if matches[best_match]:
            name = known_names[best_match]

        # Draw rectangles around the detected faces and display names
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the frame in a window named "Cam"
    cv2.imshow("Cam", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
