import cv2
import time
from pathlib import Path
from Algorithms.Haar_Cascade.Haar_Cascade_main import detect_face
from Algorithms.Eye_Aspect_Ratio.Face_Landmark_Detector import FaceLandmarkDetector

# -------------------------
# Paths
# -------------------------
_ROOT = Path(__file__).resolve().parent
_CASCADE_PREDICTOR = (
    _ROOT
    / "Algorithms"
    / "Eye_Aspect_Ratio"
    / "Models"
    / "shape_predictor_68_face_landmarks.dat"
)

# -------------------------
# Load face cascades
# -------------------------
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

# -------------------------
# Initialize landmark detector
# -------------------------
detector = FaceLandmarkDetector(str(_CASCADE_PREDICTOR))

# -------------------------
# Video capture
# -------------------------
video_capture = cv2.VideoCapture(0)

# -------------------------
# FPS variables
# -------------------------
prev_time = 0
fps = 0.0
fps_smoothing = 0.9  # higher = smoother FPS

while True:
    ret, img = video_capture.read()
    if not ret:
        break

    img = cv2.flip(img, 1)

    # -------------------------
    # Face detection
    # -------------------------
    img, face_roi, face_bbox = detect_face(img, face_cascades, return_roi=True)

    if face_bbox is not None:
        faces_landmarks = detector.detect_landmarks(
            img,
            face_bbox,
            draw_connections=True,
            draw_points=False,
            draw_rect=True,
        )

        if faces_landmarks:
            for landmarks in faces_landmarks:
                point1 = landmarks[19]  # left eyebrow
                point2 = landmarks[41]  # left eye
                distance = detector.calculate_distance(point1, point2)

                cv2.putText(
                    img,
                    f"Eye Height: {distance:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )

    # -------------------------
    # FPS calculation
    # -------------------------
    current_time = time.time()
    dt = current_time - prev_time
    prev_time = current_time

    if dt > 0:
        current_fps = 1.0 / dt
        fps = fps_smoothing * fps + (1 - fps_smoothing) * current_fps

    # -------------------------
    # FPS display (upper-right)
    # -------------------------
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

    cv2.imshow("Wake&Brake Face Landmark Detection", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
