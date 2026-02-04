import cv2

def _preprocess_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)

def _select_largest(bboxes):
    if len(bboxes) == 0:
        return None
    return max(bboxes, key=lambda b: b[2] * b[3])

def _pad_box(x, y, w, h, pad, max_w, max_h):
    px = max(0, x - pad)
    py = max(0, y - pad)
    pw = min(max_w - px, w + 2 * pad)
    ph = min(max_h - py, h + 2 * pad)
    return px, py, pw, ph

def detect(img, faceCascade, return_roi=False):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0)}

    gray_img = _preprocess_gray(img)
    h_img, w_img = gray_img.shape[:2]

    faces = faceCascade.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=8,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(80, 80),
    )

    face = _select_largest(faces)
    if face is None:
        if return_roi:
            return img, None, None
        return img

    x, y, w, h = face
    x, y, w, h = _pad_box(x, y, w, h, pad=10, max_w=w_img, max_h=h_img)

    cv2.rectangle(img, (x, y), (x + w, y + h), color["blue"], 2)
    cv2.putText(img, "Face", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color["blue"], 1, cv2.LINE_AA)

    face_roi = img[y:y + h, x:x + w]
    if return_roi:
        return img, face_roi, (x, y, w, h)
    return img