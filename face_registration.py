import cv2
import os

# Create a folder to store images if it doesn't exist
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Press 'q' to stop capturing images.")
user_name = input("Enter the name of the person: ")
user_path = os.path.join("dataset", user_name)

if not os.path.exists(user_path):
    os.makedirs(user_path)

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (200, 200))  # Resize to 200x200 for consistency
        file_name_path = os.path.join(user_path, f"{count}.jpg")
        cv2.imwrite(file_name_path, face_resized)
        count += 1

    cv2.imshow("Face Registration in progress...", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:  # Capture 50 images per user
        print(f"Face registration is completed successfully for {user_name}")
        break

cap.release()
cv2.destroyAllWindows()
