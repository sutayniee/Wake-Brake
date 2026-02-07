import cv2
from pathlib import Path
from Algorithms.Haar_Cascade.Haar_Cascade_main import detect
from Algorithms.Eye_Aspect_Ratio.Face_Landmark_Detector import FaceLandmarkDetector

_ROOT = Path(__file__).resolve().parent
_CASCADE_FACE = _ROOT / "Algorithms" / "Haar_Cascade" / "haarcascade_frontalface_default.xml"
_CASCADE_PREDICTOR = _ROOT / "Algorithms" / "Eye_Aspect_Ratio" / "Models" / "shape_predictor_68_face_landmarks.dat"
detector = FaceLandmarkDetector(str(_CASCADE_PREDICTOR))
faceCascade = cv2.CascadeClassifier(str(_CASCADE_FACE))

video_capture = cv2.VideoCapture(0)

while True:
    _, img = video_capture.read()
    img = cv2.flip(img, 1)
    img, face_roi, face_bbox = detect(img, faceCascade, return_roi=True)
    if face_bbox is None:
        cv2.imshow("Live Face Landmark Detection", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    faces = detector.detect_landmarks(
            img, face_bbox, draw_connections=True, draw_points=False, draw_rect=True
        )
     # Example: measure eye height between two points (e.g. between eyebrow and eye)
    if faces:
        for face_landmarks in faces:
            point1 = face_landmarks[19]  # Upper part of left eyebrow
            point2 = face_landmarks[41]  # Lower part of left eye
            distance = detector.calculate_distance(point1, point2)

            # Draw distance as text
            cv2.putText(
                img,
                f"Eye Height: {distance:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )
    # Display the result
    cv2.imshow("Live Face Landmark Detection", img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
