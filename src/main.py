import cv2

video_capture = cv2.VideoCapture(0)

while True:
    _, img = video_capture.read()
    cv2.imshow("Test", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.read.release()
cv2.destroyAllWindows()