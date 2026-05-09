import serial
import cv2
import time
import threading
from flask import Flask, jsonify, Response
from pathlib import Path
from Algorithms.Haar_Cascade.Haar_Cascade_main import detect_face
from Algorithms.Eye_Aspect_Ratio.Face_Landmark_Detector import FaceLandmarkDetector
from Algorithms.Eye_Aspect_Ratio.Eye_Aspect_Ratio_main import eye_aspect_ratio, put_text
from Sample_Alarm.play_sound_alarm import play_alert
from Algorithms.Arduino.Arduino_Signal import check_arduino_connection, send_to_arduino
from Algorithms.Blink_Rate.Blink_Rate_main import BlinkRateDetector
from Algorithms.Logs.main import setup_logger
import Algorithms.Server.shared_state as shared_state
from Algorithms.Server.server import run_server
from collections import deque



# Initialize System Logger for Post-Trip Review
logger = setup_logger()

# Paths
_ROOT = Path(__file__).resolve().parent
_CASCADE_PREDICTOR = (
    _ROOT
    / "Algorithms"
    / "Eye_Aspect_Ratio"
    / "Models"
    / "shape_predictor_68_face_landmarks.dat"
)

# Load face cascades
face_cascades = [
    cv2.CascadeClassifier(
        str(_ROOT / "Algorithms" / "Haar_Cascade" /"Models" / "haarcascade_frontalface_default.xml")
    ),
    cv2.CascadeClassifier(
        str(_ROOT / "Algorithms" / "Haar_Cascade" /"Models" / "haarcascade_frontalface_alt.xml")
    ),
    cv2.CascadeClassifier(
        str(_ROOT / "Algorithms" / "Haar_Cascade" /"Models" / "haarcascade_frontalface_alt2.xml")
    ),
]

# Initialize landmark detector
detector = FaceLandmarkDetector(str(_CASCADE_PREDICTOR))

# Video capture
video_capture = cv2.VideoCapture(0)

# Arduino communication 
# CONNECT TO ARDUINO FOR MULTIMODAL ALERTS
check_arduino_connection()
# if an error occured due to wrong port, change them in Arduino.Arduino_signal def check_arduino_connection

# FPS variables
prev_time = 0
fps = 0.0
fps_smoothing = 0.9  # higher = smoother FPS

# Adaptive Sensitivity (ECR) - Calibration Variables
CALIBRATION_FRAMES = 100
calibration_counter = 0
baseline_ears = []
is_calibrated = False

# EAR & PERCLOS Configuration Parameters
EAR_THRESHOLD = 0.25 # Will be overwritten by calibration
PERCLOS_WINDOW = 30.0 # 30-second window
PERCLOS_THRESHOLD = 0.20 # 20% eye closure threshold

frame_counter = 0 # Counter for consecutive closed-eye frames (micro-sleep detection)
closure_history = deque() # Stores tuples of (timestamp, is_closed)
shared_state.fatigue_level = "SAFE"

# Blink rate Configuration Parameters
blink_detector = BlinkRateDetector(
    threshold=EAR_THRESHOLD,
    min_frames_closed=2,   # IMPORTANT for 15 FPS
    window_seconds=22      # shorter window = more responsive
)

# Start Server
threading.Thread(target=run_server, daemon=True).start()

last_heartbeat_time = 0

# Start Video Capture
while True:
    head_state = 3  # DEFAULT: UNKNOWN each frame
    ret, img = video_capture.read()
    if not ret:
        break

    img = cv2.flip(img, 1)

    # FPS calculation
    current_time = time.time()
    dt = current_time - prev_time
    prev_time = current_time

    # FPS display (upper-right)
    h, w = img.shape[:2]
    cv2.putText(
        img,
        f"FPS: {fps:.1f}",
        (w - 150, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    if dt > 0:
        current_fps = 1.0 / dt
        fps = fps_smoothing * fps + (1 - fps_smoothing) * current_fps
        #Sharing FPS to flask 
        shared_state.fps_value = round(fps, 1)
    
    # Face detection using Haar Cascade and extracting ROI (Region of Interest)
    img, face_roi, face_bbox = detect_face(img, face_cascades, return_roi=True)
    blink_rate = 0.0

    # Keeps looking for detected Face
    if face_bbox is not None:
        # Detect Facial Landmarks and draw on them
        faces_landmarks = detector.detect_landmarks(
            img,
            face_bbox,
            draw_connections=True,
            draw_points=False,
            draw_rect=False,
        )

        if faces_landmarks:
            for landmarks in faces_landmarks:
                #Sharing distance to Flask
                shared_state.bpm_value = round(blink_rate, 1)
                # Extracting eye landmarks for left eyebrow and eye to calculate eye height
                point1 = landmarks[19]  # left eyebrow
                point2 = landmarks[41]  # left eye
                distance = detector.calculate_distance(point1, point2)
                #Sharing distance to Flask
                shared_state.eye_height_value = round(distance, 2)

                # Eye indices from face_landmark_detector
                right_eye_idxs = detector.FACIAL_LANDMARKS_IDXS["right_eye"]
                left_eye_idxs = detector.FACIAL_LANDMARKS_IDXS["left_eye"]

                # Extract eye coordinates
                right_eye = landmarks[right_eye_idxs[0] : right_eye_idxs[1]]
                left_eye_ = landmarks[left_eye_idxs[0] : left_eye_idxs[1]]

                # Computer EAR
                right_ear = eye_aspect_ratio(right_eye)
                left_ear = eye_aspect_ratio(left_eye_)
                ear = (left_ear + right_ear) / 2.0

                # Adaptive Sensitivity (ECR) Calibration Phase
                if not is_calibrated:
                    baseline_ears.append(ear)
                    calibration_counter += 1
                    put_text(img, f"CALIBRATING ECR: {calibration_counter}/{CALIBRATION_FRAMES}", (50, 100), color=(0, 255, 255))
                    
                    if calibration_counter >= CALIBRATION_FRAMES:
                        baseline_ear = sum(baseline_ears) / len(baseline_ears)
                        # Adaptive threshold: 75% of the user's normal baseline EAR
                        EAR_THRESHOLD = baseline_ear * 0.75
                        is_calibrated = True
                        print(f"Calibration Complete! Baseline EAR: {baseline_ear:.2f}, Set Threshold: {EAR_THRESHOLD:.2f}")
                        logger.info(f"CALIBRATION_COMPLETE,100%,0.0,{baseline_ear:.2f},0.0,{fps:.1f}")
                    continue  # Skip detection logic until calibrated

                # Postural Deviation Check
                is_looking_down, pitch_ratio = detector.check_postural_deviation(landmarks)
                 # HEAD POSTURE CLASSIFICATION
                head_state = 3  # default: IGNORE / UNKNOWN

                # HEAD STATE CLASSIFICATION (ONLY ONCE)
                if pitch_ratio is None:
                    head_state = 3   # UNKNOWN
                elif 2.4 < pitch_ratio <= 2.7:
                    head_state = 2   # NORMAL
                elif pitch_ratio <= 2.4:
                    head_state = 1   # HEAD DOWN
                else:
                    head_state = 3
                    
                # HEAD STATE DISPLAY (REAL OUTPUT)
                # ALWAYS SHOW HEAD STATE (INSIDE LOOP, NOT OUTSIDE)

                if head_state == 1:
                    put_text(img, "HEAD DOWN DETECTED", (10, 150), color=(0, 0, 255))

                elif head_state == 2:
                    put_text(img, "NORMAL HEAD POSITION", (10, 150), color=(0, 255, 0))

                #Sending EAR to Flask
                shared_state.ear_value = round(ear, 2)
                
                #Sharing blink rate to flask
                blink_rate = blink_detector.update(ear)
                shared_state.bpm_value = round(blink_rate, 1)
                
                # Print Blink Rate
                put_text(img, f"BPM: {blink_rate:.1f}", (10, 90), color=(255, 255, 0))
                # Print EAR
                put_text(img, f"EAR: {ear:.2f}", (10, 60), color=(0, 255, 0))
                # Print EAR & Threshold
                put_text(img, f"EAR: {ear:.2f} (Thresh: {EAR_THRESHOLD:.2f})", (10, 60), color=(0, 255, 0))

                # ========================================================
                # Check for manual overrides from the App
                with shared_state.lock:
                    if getattr(shared_state, 'clear_history_flag', False):
                        closure_history.clear() # Instantly drops PERCLOS to 0%
                        shared_state.clear_history_flag = False

                # PERCLOS and Drowsiness check
                current_time = time.time()
                is_closed = ear < EAR_THRESHOLD
                
                # Update closure history for PERCLOS
                closure_history.append((current_time, is_closed))
                
                # Remove old frames outside the 30-second window
                while closure_history and current_time - closure_history[0][0] > PERCLOS_WINDOW:
                    closure_history.popleft()
                
                # Calculate PERCLOS (Percentage of Eye Closure)
                if len(closure_history) > 0:
                    closed_frames = sum(1 for _, closed in closure_history if closed)
                    perclos = closed_frames / len(closure_history)
                else:
                    perclos = 0.0
                    # Send PERCLOS to mobile app
                    shared_state.perclos_value = round(perclos * 100, 1)
                    
                # Fast Recovery Mechanism: Track how long eyes have been consistently open
                if not is_closed and head_state != 1:
                    shared_state.frames_open = getattr(shared_state, 'frames_open', 0) + 1
                    if shared_state.frames_open > 60: # If awake for ~2 seconds straight
                        # mathematically safe reset: convert all history to "eyes open"
                        # This preserves the queue length so a single blink won't spike the math to 100%
                        for i in range(len(closure_history)):
                            closure_history[i] = (closure_history[i][0], False)
                        perclos = 0.0 # Force update for this frame
                else:
                    shared_state.frames_open = 0

                # Print perclos in camera
                put_text(img, f"PERCLOS: {perclos:.1%}", (10, 120), color=(0, 165, 255))

                # Alert Escalation Finite-State Machine & Confidence Scoring
                # Confidence score is based directly on the PERCLOS percentage, head pitch and blink rate
                confidence_score = min(
                    100,
                    int(
                        (perclos * 100)
                        + (20 if head_state == 1 else 0)
                        + (10 if blink_rate < 10 else 0)
                        + (5 if blink_rate > 30 else 0)
                    )
                )
                
                if perclos >= 0.80:
                    # SEVERE CRITICAL: 80% Fatigue - Triggers locked Scent cycle
                    # play_alert()
                    if shared_state.fatigue_level != "SEVERE_SCENT":
                        print(f"SEVERE FATIGUE! Confidence: {confidence_score}%", time.ctime())
                        shared_state.fatigue_level = "SEVERE_SCENT"
                        logger.info(f"SEVERE_FATIGUE,{confidence_score}%,{perclos:.2f},{ear:.2f},{pitch_ratio:.2f},{fps:.1f}")
                        send_to_arduino('S') 
                    put_text(img, f"SEVERE FATIGUE ({confidence_score}%) - SCENT", (150, 100), color=(0, 0, 255))

                elif perclos >= 0.70 or (head_state == 1 and perclos >= 0.50):
                    # CRITICAL STATE: 70% Fatigue - Triggers Buzzer and Vibration
                    # play_alert()
                    if shared_state.fatigue_level != "CRITICAL_BUZZER":
                        print(f"CRITICAL FATIGUE! Confidence: {confidence_score}%", time.ctime())
                        shared_state.fatigue_level = "CRITICAL_BUZZER"
                        logger.info(f"CRITICAL_FATIGUE,{confidence_score}%,{perclos:.2f},{ear:.2f},{pitch_ratio:.2f},{fps:.1f}")
                        send_to_arduino('B') 
                    put_text(img, f"CRITICAL FATIGUE ({confidence_score}%) - BUZZER", (150, 100), color=(0, 0, 255))

                elif is_closed:
                    frame_counter += 1
                    # WARNING STATE: Micro-sleep detected (>3s of closed eyes. Assumes ~15-30fps, 45 frames = ~1.5 to 3 seconds)
                    if frame_counter >= 20: 
                        # play_alert()
                        if shared_state.fatigue_level != "WARNING_HAPTIC":
                            print(f"DROWSY WARNING (Micro-sleep) Confidence: {confidence_score}%", time.ctime())
                            shared_state.fatigue_level = "WARNING_HAPTIC"
                            logger.info(f"MICRO_SLEEP_WARNING,{confidence_score}%,{perclos:.2f},{ear:.2f},{pitch_ratio:.2f},{fps:.1f}")
                            send_to_arduino('H') 
                        put_text(img, "Warning: Micro-sleep! - HAPTIC", (200, 100), color=(0, 165, 255))
                
                else:
                    # SAFE STATE: Auto-OFF immediately when fatigue is normal and eyes are open
                    frame_counter = 0
                    if shared_state.fatigue_level != "SAFE":
                        print("Driver Alerted and Safe.", time.ctime())
                        shared_state.fatigue_level = "SAFE"
                        send_to_arduino('N') # Sends Auto-OFF. (Arduino keeps Scent locked).
                        logger.info(f"SAFE_STATE,100%,{perclos:.2f},{ear:.2f},{pitch_ratio:.2f},{fps:.1f}")

                # Print Eye Height
                cv2.putText(
                    img,
                    f"Eye Height: {distance:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )

                # Arduino Heartbeat Failsafe (Send current state every 1 second)
                if current_time - last_heartbeat_time > 1.0:
                    last_heartbeat_time = current_time
                    current_state = shared_state.fatigue_level
                    if current_state == "SEVERE_SCENT":
                        send_to_arduino('S')
                    elif current_state == "CRITICAL_BUZZER":
                        send_to_arduino('B')
                    elif current_state == "WARNING_HAPTIC":
                        send_to_arduino('H')
                    else:
                        send_to_arduino('N')

    # video in mobile
    with shared_state.lock:
        shared_state.output_frame = img.copy()

    cv2.imshow("Wake&Brake Drowsiness Detector", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
