import cv2

def _preprocess_gray(img):
    """Convert image to grayscale and equalize histogram for better detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)

def _select_largest(bboxes):
    """Select the bounding box with the largest area."""
    if len(bboxes) == 0:
        return None
    return max(bboxes, key=lambda b: b[2] * b[3])

def _pad_box(x, y, w, h, pad, max_w, max_h):
    """Add padding to a bounding box without exceeding image bounds."""
    px = max(0, x - pad)
    py = max(0, y - pad)
    pw = min(max_w - px, w + 2 * pad)
    ph = min(max_h - py, h + 2 * pad)
    return px, py, pw, ph

def detect_face(img, faceCascades, return_roi=False, pad=10):
    """
    Detect the largest face using multiple cascades.

    Args:
        img: input BGR image
        faceCascades: list of cv2.CascadeClassifier objects
        return_roi: whether to return cropped face ROI
        pad: padding around detected face

    Returns:
        img: image with rectangle drawn
        face_roi: cropped face region (optional)
        bbox: (x, y, w, h) of detected face (optional)
    """
    if isinstance(faceCascades, cv2.CascadeClassifier):
        faceCascades = [faceCascades]

    gray_img = _preprocess_gray(img)
    h_img, w_img = gray_img.shape[:2]
    all_faces = []

    # Detect faces using all cascades
    for cascade in faceCascades:
        faces = cascade.detectMultiScale(
            gray_img,
            scaleFactor=1.1,
            minNeighbors=8,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(90, 90),
        )
        if len(faces) > 0:
            all_faces.extend(faces)

    face = _select_largest(all_faces)
    if face is None:
        if return_roi:
            return img, None, None
        return img

    x, y, w, h = _pad_box(*face, pad=pad, max_w=w_img, max_h=h_img)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(img, "Driver", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)

    face_roi = img[y:y + h, x:x + w]
    if return_roi:
        return img, face_roi, (x, y, w, h)
    return img
