
# Face Recognition Based Attendance System

This project implements an automated attendance system using facial recognition technology. It identifies individuals from video or image feeds and logs their attendance in real-time.

## Features
- **Face Registration:** Enroll users by capturing and storing their facial data.
- **Real-Time Recognition:** Detects and recognizes registered faces in live or recorded videos.
- **Attendance Logging:** Automatically logs attendance in CSV files with timestamps.

## Project Structure
- **Scripts:**
  - `face_registration.py`: Registers new users and stores their face encodings.
  - `model_training.py`: Trains the face recognition model using registered faces.
  - `face_recog.py`: The face recognition module to record the attendance.
- **Data Files:**
  - `face_encodings.pkl`: Contains serialized face encodings.
  - `shape_predictor_68_face_landmarks.dat`: Pre-trained model for facial landmarks.
- **Attendance Logs:** Multiple CSV files storing attendance records.

## Requirements
- **Python Packages:**
  - `numpy`
  - `opencv-python`
  - `dlib`
  - `face_recognition`

## Setup
1. **Install Dependencies:**
   ```bash
   pip install numpy opencv-python dlib face_recognition
   ```
2. **Run Face Registration:**
   ```bash
   python face_registration.py
   ```
3. **Train the model:**
   ```bash
   python model_training.py
   ```
4. **Start Attendance System:**
   ```bash
   python face_recog.py
   ```

## Usage
1. **Register Faces:** Use the registration script to capture and store new faces.
2. **Monitor Attendance:** Launch the recognition script to detect and log attendance.
3. **View Logs:** Check the CSV files for attendance data

## Contributing
Contributions are welcome! Please fork the repository and create a pull request.

## License
This project is licensed under the MIT License.
