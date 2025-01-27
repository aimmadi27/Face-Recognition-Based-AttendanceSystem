import cv2
import os
import face_recognition
import numpy as np
import pickle

# Load images from the dataset folder and encode faces
dataset_path = "dataset"
known_encodings = []
known_names = []

for user_name in os.listdir(dataset_path):
    user_folder = os.path.join(dataset_path, user_name)
    if not os.path.isdir(user_folder):
        continue

    for image_name in os.listdir(user_folder):
        image_path = os.path.join(user_folder, image_name)
        image = face_recognition.load_image_file(image_path)
        try:
            encoding = face_recognition.face_encodings(image)[0]
            known_encodings.append(encoding)
            known_names.append(user_name)
        except IndexError:
            print(f"Face not found in image {image_path}")

# Save the encodings and names
with open("face_encodings.pkl", "wb") as file:
    pickle.dump({"encodings": known_encodings, "names": known_names}, file)

print("Model training complete. Encodings saved.")
