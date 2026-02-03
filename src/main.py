import cv2
from pathlib import Path
from Algorithms.Haar_Cascade.Haar_Cascade_main import detect

_ROOT = Path(__file__).resolve().parent
_CASCADE_FACE = _ROOT / "Algorithms" / "Haar_Cascade" / "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(str(_CASCADE_FACE))

video_capture = cv2.VideoCapture(0)

while True:
    _, img = video_capture.read()
    img = detect(img, faceCascade)
    cv2.imshow("Test", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
