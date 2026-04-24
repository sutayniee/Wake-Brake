import serial
import cv2
import time
import threading
from flask import Flask, jsonify

from pathlib import Path
from Algorithms.Haar_Cascade.Haar_Cascade_main import detect_face
from Algorithms.Eye_Aspect_Ratio.Face_Landmark_Detector import FaceLandmarkDetector
from Algorithms.Eye_Aspect_Ratio.Eye_Aspect_Ratio_main import eye_aspect_ratio, put_text
from Sample_Alarm.play_sound_alarm import play_alert
from Arduino.Arduino_Signal import check_arduino_connection, send_to_arduino

# ================= FLASK SERVER =================
app = Flask(__name__)

# Shared fatigue variable
fatigue_level = "LOW"

# -------- SIMPLE AUTO-REFRESH PAGE --------
@app.route('/fatigue')
@app.route('/fatigue')
def get_fatigue():
    return jsonify({"level": fatigue_level})

# -------- REAL-TIME STREAM --------
@app.route('/stream')
def stream():
    def generate():
        while True:
            yield f"data: {fatigue_level}\n\n"
            time.sleep(1)

    return app.response_class(generate(), mimetype='text/event-stream')

# -------- LIVE DASHBOARD --------
@app.route('/')
def index():
    return """
    <html>
    <body style="text-align:center;">
        <h1>Wake&Brake Live Monitor</h1>
        <h2 id="status">Waiting...</h2>

        <script>
            var source = new EventSource("/stream");
            source.onmessage = function(event) {
                document.getElementById("status").innerHTML = event.data;
            };
        </script>
    </body>
    </html>
    """

# -------- RUN SERVER --------
def run_server():
    app.run(host="0.0.0.0", port=5000)

# ================= PATHS =================
_ROOT = Path(__file__).resolve().parent
_CASCADE_PREDICTOR = (
    _ROOT
    / "Algorithms"
    / "Eye_Aspect_Ratio"
    / "Models"
    / "shape_predictor_68_face_landmarks.dat"
)

# ================= LOAD CASCADES =================
face_cascades = [
    cv2.CascadeClassifier(
        str(_ROOT / "Algorithms" / "Haar_Cascade" / "Models" / "haarcascade_frontalface_default.xml")
    ),
    cv2.CascadeClassifier(
        str(_ROOT / "Algorithms" / "Haar_Cascade" / "Models" / "haarcascade_frontalface_alt.xml")
    ),
    cv2.CascadeClassifier(
        str(_ROOT / "Algorithms" / "Haar_Cascade" / "Models" / "haarcascade_frontalface_alt2.xml")
    ),
]

# ================= INITIALIZE =================
detector = FaceLandmarkDetector(str(_CASCADE_PREDICTOR))
video_capture = cv2.VideoCapture(0)

check_arduino_connection()

# FPS
prev_time = 0
fps = 0.0
fps_smoothing = 0.9

# EAR CONFIG
EAR_THRESHOLD = 0.30
CONSECUTIVE_FRAMES_THRESHOLD = 20
frame_counter = 0
drowsy = False

# ================= START SERVER =================
threading.Thread(target=run_server, daemon=True).start()

# ================= MAIN LOOP =================
while True:
    ret, img = video_capture.read()
    if not ret:
        break

    img = cv2.flip(img, 1)

    # FPS
    current_time = time.time()
    dt = current_time - prev_time
    prev_time = current_time

    if dt > 0:
        current_fps = 1.0 / dt
        fps = fps_smoothing * fps + (1 - fps_smoothing) * current_fps

    # Face detection
    img, face_roi, face_bbox = detect_face(img, face_cascades, return_roi=True)

    if face_bbox is not None:
        faces_landmarks = detector.detect_landmarks(
            img,
            face_bbox,
            draw_connections=True,
            draw_points=False,
            draw_rect=False,
        )

        if faces_landmarks:
            for landmarks in faces_landmarks:

                # Eye height (optional)
                point1 = landmarks[19]
                point2 = landmarks[41]
                distance = detector.calculate_distance(point1, point2)

                # Eyes
                right_eye_idxs = detector.FACIAL_LANDMARKS_IDXS["right_eye"]
                left_eye_idxs = detector.FACIAL_LANDMARKS_IDXS["left_eye"]

                right_eye = landmarks[right_eye_idxs[0]: right_eye_idxs[1]]
                left_eye_ = landmarks[left_eye_idxs[0]: left_eye_idxs[1]]

                # EAR
                right_ear = eye_aspect_ratio(right_eye)
                left_ear = eye_aspect_ratio(left_eye_)
                ear = (left_ear + right_ear) / 2.0

                put_text(img, f"EAR: {ear:.2f}", (10, 60), color=(0, 255, 0))

                # ================= DROWSINESS LOGIC =================
                if ear < EAR_THRESHOLD:
                    frame_counter += 1

                    if frame_counter >= CONSECUTIVE_FRAMES_THRESHOLD:
                        drowsy = True
                        fatigue_level = "HIGH"

                        put_text(img, "Fatigue Detected!", (200, 220))
                        print("Drowsiness Detected!", time.ctime())

                        send_to_arduino('1')
                        play_alert()

                    else:
                        fatigue_level = "LOW"

                else:
                    if drowsy:
                        send_to_arduino('0')

                    drowsy = False
                    frame_counter = 0
                    fatigue_level = "LOW"

                    print("Driver Alerted!", time.ctime())

                # Eye height
                cv2.putText(
                    img,
                    f"Eye Height: {distance:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )

    # FPS display
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

    cv2.imshow("Wake&Brake Drowsiness Detector", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()