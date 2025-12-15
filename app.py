import cv2
import os
import joblib
import numpy as np
import pandas as pd
from datetime import date, datetime
from flask import Flask, request, render_template, redirect, url_for, Response, flash

# --- CONFIGURATION ---
class Config:
    BASE_DIR = os.getcwd()
    ATTENDANCE_DIR = os.path.join(BASE_DIR, 'Attendance')
    STATIC_DIR = os.path.join(BASE_DIR, 'static')
    FACES_DIR = os.path.join(STATIC_DIR, 'faces')
    MODEL_PATH = os.path.join(STATIC_DIR, 'face_recognition_model.pkl')
    HAAR_CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
    IMG_SIZE = (50, 50)

for dir_path in [Config.ATTENDANCE_DIR, Config.STATIC_DIR, Config.FACES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# --- GLOBAL STATE ---
# We use this to let the 'Mark Attendance' button know who is currently visible
current_detected_user = None

# --- BACKEND LOGIC ---
class FaceRecognitionSystem:
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier(Config.HAAR_CASCADE_PATH)
        self.model = None
        if os.path.exists(Config.MODEL_PATH):
            self.model = joblib.load(Config.MODEL_PATH)

    def get_todays_file(self):
        filename = f'Attendance-{date.today().strftime("%m_%d_%y")}.csv'
        filepath = os.path.join(Config.ATTENDANCE_DIR, filename)
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                f.write('Name,Roll,Time')
        return filepath

    def mark_attendance(self, label):
        """Marks attendance for the specific label (Name_ID)."""
        try:
            username, userid = label.split('_')
        except ValueError:
            return "Error: Invalid Label Format"

        current_time = datetime.now().strftime("%H:%M:%S")
        file_path = self.get_todays_file()
        
        df = pd.read_csv(file_path)
        if str(userid) not in df['Roll'].astype(str).tolist():
            with open(file_path, 'a') as f:
                f.write(f'\n{username},{userid},{current_time}')
            return f"Attendance Marked for {username}"
        return f"Attendance already marked for {username}"

    def get_attendance_data(self):
        file_path = self.get_todays_file()
        df = pd.read_csv(file_path)
        return df['Name'].tolist(), df['Roll'].tolist(), df['Time'].tolist(), len(df)

fr_system = FaceRecognitionSystem()

# --- FLASK APP ---
app = Flask(__name__)
app.secret_key = "secret"

# 1. THE GENERATOR FUNCTION
# This yields video frames one by one
def generate_frames():
    global current_detected_user
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Face Detection Logic
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = fr_system.face_detector.detectMultiScale(gray, 1.1, 5)
        
        current_detected_user = None # Reset every frame

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_img = cv2.resize(frame[y:y+h, x:x+w], Config.IMG_SIZE)
            
            if fr_system.model:
                prediction = fr_system.model.predict(face_img.reshape(1, -1))
                name = prediction[0]
                current_detected_user = name # Update global state
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield the frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
               
    cap.release()

@app.route('/')
def home():
    names, rolls, times, l = fr_system.get_attendance_data()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, 
                           datetoday2=date.today().strftime("%d-%B-%Y"))

# 2. THE VIDEO FEED ROUTE
# The HTML <img> tag will point to this URL
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 3. THE "MARK ATTENDANCE" BUTTON ROUTE
# Instead of pressing 'a' on keyboard, the user clicks a button
@app.route('/mark_attendance_action')
def mark_attendance_action():
    global current_detected_user
    
    if current_detected_user:
        msg = fr_system.mark_attendance(current_detected_user)
    else:
        msg = "No face detected or face not recognized."
        
    return render_template('home.html', mess=msg, **_get_home_context())

def _get_home_context():
    names, rolls, times, l = fr_system.get_attendance_data()
    return {'names': names, 'rolls': rolls, 'times': times, 'l': l, 'datetoday2': date.today().strftime("%d-%B-%Y")}

if __name__ == '__main__':
    app.run(debug=True, port=1000)