import serial
import cv2
import time
from pathlib import Path
from Algorithms.Haar_Cascade.Haar_Cascade_main import detect_face
from Algorithms.Eye_Aspect_Ratio.Face_Landmark_Detector import FaceLandmarkDetector
from Algorithms.Eye_Aspect_Ratio.Eye_Aspect_Ratio_main import eye_aspect_ratio, put_text
from Sample_Alarm.play_sound_alarm import play_alert

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
    cv2.CascadeClassifier(str(_ROOT / "Algorithms" / "Haar_Cascade" /"Models" / "haarcascade_frontalface_default.xml")),
    cv2.CascadeClassifier(str(_ROOT / "Algorithms" / "Haar_Cascade" /"Models" / "haarcascade_frontalface_alt.xml")),
    cv2.CascadeClassifier(str(_ROOT / "Algorithms" / "Haar_Cascade" /"Models" / "haarcascade_frontalface_alt2.xml")),
]

# Initialize landmark detector
detector = FaceLandmarkDetector(str(_CASCADE_PREDICTOR))

# Video capture
video_capture = cv2.VideoCapture(0)

# FPS variables
prev_time = 0
fps = 0.0
fps_smoothing = 0.9  

# EAR Configuration Parameters
EAR_THRESHOLD = 0.30 
CONSECUTIVE_FRAMES_THRESHOLD = 20 
frame_counter = 0 
drowsy = False
arduino = None # Initialize variable safely

# --- 1. CONNECT TO ARDUINO ---
try:
    print("Connecting to Arduino on COM4...")
    arduino = serial.Serial(port='COM4', baudrate=9600, timeout=0.1)
    
    # CRITICAL FIX: The Arduino resets when Serial connects. 
    # We MUST wait 2 seconds for it to finish booting before sending data.
    time.sleep(2) 
    print("Arduino Connected successfully!")
except Exception as e:
    print(f"Arduino connection failed: {e}")

# Start Video Capture Loop
while True:
    ret, img = video_capture.read()
    if not ret:
        break

    img = cv2.flip(img, 1)

    # Face detection
    img, face_roi, face_bbox = detect_face(img, face_cascades, return_roi=True)

    if face_bbox is not None:
        faces_landmarks = detector.detect_landmarks(
            img, face_bbox, draw_connections=True, draw_points=False, draw_rect=False
        )

        if faces_landmarks:
            for landmarks in faces_landmarks:
                point1 = landmarks[19]  
                point2 = landmarks[41]  
                distance = detector.calculate_distance(point1, point2)

                right_eye_idxs = detector.FACIAL_LANDMARKS_IDXS["right_eye"]
                left_eye_idxs = detector.FACIAL_LANDMARKS_IDXS["left_eye"]

                right_eye = landmarks[right_eye_idxs[0] : right_eye_idxs[1]]
                left_eye_ = landmarks[left_eye_idxs[0] : left_eye_idxs[1]]

                right_ear = eye_aspect_ratio(right_eye)
                left_ear = eye_aspect_ratio(left_eye_)
                ear = (left_ear + right_ear) / 2.0

                put_text(img, f"EAR: {ear:.2f}", (10, 60), color=(0, 255, 0))

                # --- 2. THE ARDUINO LOGIC INSIDE THE LOOP ---
                if ear < EAR_THRESHOLD:
                    frame_counter += 1
                    if frame_counter >= CONSECUTIVE_FRAMES_THRESHOLD:
                        if not drowsy: # State change: Awake -> Drowsy
                            if arduino:
                                arduino.write(b'1') # Send '1' ONLY ONCE when crossing threshold
                        drowsy = True
                        put_text(img, "Fatigue Detected!", (200, 220))
                        play_alert()
                else:
                    if drowsy: # State change: Drowsy -> Awake
                        if arduino:
                            arduino.write(b'0') # Send '0' ONLY ONCE when waking up
                    frame_counter = 0
                    drowsy = False
                      
                # Print Eye Height
                cv2.putText(
                    img, f"Eye Height: {distance:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2,
                )

    # FPS calculation
    current_time = time.time()
    dt = current_time - prev_time
    prev_time = current_time

    if dt > 0:
        current_fps = 1.0 / dt
        fps = fps_smoothing * fps + (1 - fps_smoothing) * current_fps

    h, w = img.shape[:2]
    cv2.putText(
        img, f"FPS: {fps:.1f}", (w - 150, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
    )

    cv2.imshow("Wake&Brake Drowsiness Detector", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- 3. CLEANUP ---
if arduino:
    arduino.write(b'0') # Ensure alarm is off when quitting
    arduino.close()     # Close serial port cleanly

video_capture.release()
cv2.destroyAllWindows()