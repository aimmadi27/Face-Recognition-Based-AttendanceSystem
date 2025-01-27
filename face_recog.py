import cv2
import dlib
import face_recognition
import pickle
from datetime import datetime
from scipy.spatial import distance as dist
import time  # Import time module

# Load the face encodings and names
with open("face_encodings.pkl", "rb") as file:
    data = pickle.load(file)

known_encodings = data["encodings"]
known_names = data["names"]

# Initialize video capture
cap = cv2.VideoCapture(0)

# Create a new attendance file with a timestamp in its name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
attendance_log = f"attendance_{timestamp}.csv"

# Write header to the CSV file
with open(attendance_log, "w") as log:
    log.write("Name,Timestamp\n")

# Dictionary to keep track of people who have already been marked
marked_names = set()

# Load dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye aspect ratio to detect blink
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Thresholds and consecutive frame counters for blink detection
EYE_AR_THRESHOLD = 0.25
EYE_AR_CONSEC_FRAMES = 3
blink_counter = 0
blink_detected = False

# Variables to control the text display duration
display_time = 3  # Time in seconds
text_display_start = None  # To track when the text was first shown
displayed_name = ""  # To track the name being displayed

print("Press 'q' to stop the recognition process.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Detect faces using dlib
    faces = detector(gray_frame)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

        # Perform liveness detection (blink detection)
        for face in faces:
            landmarks = predictor(gray_frame, face)
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

            # Calculate eye aspect ratio
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Check if the eye aspect ratio is below the blink threshold
            if ear < EYE_AR_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= EYE_AR_CONSEC_FRAMES:
                    blink_detected = True
                blink_counter = 0

        # Only mark attendance if the person is recognized and liveness is confirmed
        if name != "Unknown" and name not in marked_names and blink_detected:
            marked_names.add(name)
            # Mark attendance in the log file
            with open(attendance_log, "a") as log:
                time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log.write(f"{name},{time_now}\n")
            
            # Start the timer for text display
            text_display_start = time.time()
            displayed_name = name  # Store the name to display
            
            # Notify that attendance has been marked
            print(f"Attendance marked for {name} at {time_now}")
            blink_detected = False  # Reset blink detection for the next person

        # Draw a rectangle around the face and label it
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Check if we need to display the "attendance marked" message
    if text_display_start:
        elapsed_time = time.time() - text_display_start
        if elapsed_time < display_time:
            cv2.putText(frame, f"Attendance marked for {displayed_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            text_display_start = None  # Reset after the duration has passed

    # Display the current time on the top-right corner
    current_time = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, current_time, (frame.shape[1] - 120, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Attendance Recording...", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
