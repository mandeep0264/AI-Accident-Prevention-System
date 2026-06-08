import cv2 
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import pygame
import threading
import tkinter as tk
import time
import sys
import os

# ================= RESOURCE PATH =================
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # PyInstaller path
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ================= GLOBAL VARIABLES =================
running = False
alarm_on = False
detection_thread = None
yawn_alarm_start = None

# ================= INITIALIZE PYGAME =================
pygame.mixer.init()
pygame.mixer.music.load(resource_path("music.wav"))

# ================= HELPER FUNCTIONS =================
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

def sound_alarm():
    global alarm_on
    if not alarm_on:
        pygame.mixer.music.play(-1)
        alarm_on = True

def stop_alarm():
    global alarm_on
    if alarm_on:
        pygame.mixer.music.stop()
        alarm_on = False

# ================= DETECTION FUNCTION =================
def run_detection():
    global running, yawn_alarm_start

    # Thresholds
    EYE_THRESHOLD = 0.25
    EYE_FRAMES = 20
    MAR_THRESHOLD = 0.6
    YAWN_FRAMES = 15

    eye_frame_counter = 0
    mouth_frame_counter = 0
    yawn_counter = 0

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(resource_path("shape_predictor_68_face_landmarks.dat"))

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    cap = cv2.VideoCapture(0)
    time.sleep(1)
    if not cap.isOpened():
        print("Cannot access camera!")
        return

    running = True

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]

            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
            mar = mouth_aspect_ratio(mouth)

            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0,255,0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0,255,0), 1)
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255,0,0), 1)

            # Eye detection
            if ear < EYE_THRESHOLD:
                eye_frame_counter += 1
                sound_alarm()
            else:
                eye_frame_counter = 0
                stop_alarm()

            # Yawn detection
            if mar > MAR_THRESHOLD:
                mouth_frame_counter += 1
            else:
                if mouth_frame_counter >= YAWN_FRAMES:
                    yawn_counter += 1
                mouth_frame_counter = 0

            # Yawn alarm logic
            if yawn_counter >= 3:
                if yawn_alarm_start is None:
                    yawn_alarm_start = time.time()
                    sound_alarm()
                elif time.time() - yawn_alarm_start >= 5:
                    stop_alarm()
                    yawn_alarm_start = None
                    yawn_counter = 0
                cv2.putText(frame, "YAWN ALERT DRIVER FATIGUE!", (120, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            else:
                if eye_frame_counter < EYE_FRAMES:  # avoid stopping alarm if eyes closed
                    stop_alarm()
                yawn_alarm_start = None

        # ================= FATIGUE SCORE =================
        # Combine eye closure frames and yawn count to get percentile (0-100)
        fatigue_score = min(100, (eye_frame_counter / EYE_FRAMES * 50) + (yawn_counter / 3 * 50))
        cv2.putText(frame, f"Fatigue Score: {fatigue_score:.0f}%", (20,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # Display counters
        cv2.putText(frame, f"Eye Frames: {eye_frame_counter}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, f"Yawn Count: {yawn_counter}", (20,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.imshow("Driver Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    stop_alarm()
    running = False

# ================= GUI FUNCTIONS =================
def start_detection_thread():
    global detection_thread
    if not running:
        detection_thread = threading.Thread(target=run_detection)
        detection_thread.start()

def stop_app():
    global running, detection_thread
    running = False
    if detection_thread and detection_thread.is_alive():
        detection_thread.join()
    stop_alarm()
    pygame.mixer.quit()
    cv2.destroyAllWindows()
    root.destroy()

# ================= GUI =================
root = tk.Tk()
root.title("Driver Drowsiness Detection System")
root.geometry("320x220")
root.resizable(False, False)

frame = tk.Frame(root)
frame.pack(expand=True)

start_btn = tk.Button(frame, text="Start Detection",
                      command=start_detection_thread,
                      width=20, height=2)
start_btn.pack(pady=15)

exit_btn = tk.Button(frame, text="Exit",
                     command=stop_app,
                     width=20, height=2)
exit_btn.pack(pady=10)

root.protocol("WM_DELETE_WINDOW", stop_app)
root.mainloop()
