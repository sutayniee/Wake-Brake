import cv2
import time
from pathlib import Path
from Algorithms.Haar_Cascade.Haar_Cascade_main import detect_face
from Algorithms.Eye_Aspect_Ratio.Face_Landmark_Detector import FaceLandmarkDetector
from Algorithms.Eye_Aspect_Ratio.Eye_Aspect_Ratio_main import eye_aspect_ratio, put_text


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

# FPS variables
prev_time = 0
fps = 0.0
fps_smoothing = 0.9  # higher = smoother FPS

# EAR Configuration Parameters
EAR_THRESHOLD = 0.30 # Below this, eyes are considered closed
CONSECUTIVE_FRAMES_THRESHOLD = 20 # Number of frames before drowsiness is detected
frame_counter = 0 # Counter for consecutive closed-eye frames
drowsy = False

# Start Video Capture
while True:
    ret, img = video_capture.read()
    if not ret:
        break

    img = cv2.flip(img, 1)

    # Face detection using Haar Cascade and extracting ROI (Region of Interest)
    img, face_roi, face_bbox = detect_face(img, face_cascades, return_roi=True)

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
                # Extracting eye landmarks for left eyebrow and eye to calculate eye height
                point1 = landmarks[19]  # left eyebrow
                point2 = landmarks[41]  # left eye
                distance = detector.calculate_distance(point1, point2)

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

                # Print EAR
                put_text(img, f"EAR: {ear:.2f}", (10, 60), color=(0, 255, 0))

                #Drowsiness check
                if ear < EAR_THRESHOLD:
                    frame_counter += 1
                    if frame_counter >= CONSECUTIVE_FRAMES_THRESHOLD:
                        drowsy = True
                        put_text(img, "WAKE THE FUCK UP!!", (200, 220))
                        print("Drowsiness Detected!")
                else:
                    frame_counter = 0
                    drowsy = False
                      
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

    # FPS calculation
    current_time = time.time()
    dt = current_time - prev_time
    prev_time = current_time

    if dt > 0:
        current_fps = 1.0 / dt
        fps = fps_smoothing * fps + (1 - fps_smoothing) * current_fps

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

    cv2.imshow("Wake&Brake Drowsiness Detector", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
