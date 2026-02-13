import cv2
import dlib
import numpy as np
from pathlib import Path
import sys

model_path = (
    Path(__file__).resolve().parent / "Models" / "shape_predictor_68_face_landmarks.dat"
)

class FaceLandmarkDetector:
    """
    A class for detecting 68 facial landmarks using dlib's pre-trained models.

    This class provides functionality to:
    - Predict 68 facial landmarks within a provided face ROI
    - Optionally draw connections between landmarks or just points
    - Draw bounding boxes around provided face ROI
    - Compute distances between specific landmark points
    """

    # Facial landmark indexes (based on dlib's 68-point shape predictor)
    FACIAL_LANDMARKS_IDXS = {
        "jaw": (0, 17),
        "right_eyebrow": (17, 22),
        "left_eyebrow": (22, 27),
        "nose": (27, 36),
        "right_eye": (36, 42),
        "left_eye": (42, 48),
        "outer_mouth": (48, 60),
        "inner_mouth": (60, 68),
    }

    def __init__(self, model_path: str = "shape_predictor_68_face_landmarks.dat"):
        """
        Initializes the FaceLandmarkDetector class.

        Args:
            model_path (str): Path to the pre-trained dlib 68-point facial landmark model.
        """
        self.landmark_predictor = dlib.shape_predictor(model_path)

    def detect_landmarks(
        self,
        image: np.ndarray,
        face_bbox: tuple | None,
        draw_connections: bool = False,
        draw_points: bool = False,
        draw_rect: bool = False,
    ) -> list:
        """
        Detects facial landmarks in an image and optionally draws them.

        Args:
            image (np.ndarray): The input BGR image.
            face_bbox (tuple | None): Face ROI as (x, y, w, h) from Haar detector. If None, returns [].
            draw_connections (bool): Whether to draw lines between connected landmark points.
            draw_points (bool): Whether to draw landmark points as circles.
            draw_rect (bool): Whether to draw a bounding box around each detected face.

        Returns:
            List of np.ndarray: Each array is shape (68, 2) representing landmark coordinates for one face.
        """
        # dlib expects a contiguous 8-bit grayscale or RGB image.
        if image is None:
            return []
        if hasattr(image, "get"):
            image = image.get()
        image = np.asarray(image)
        if image.size == 0:
            return []
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8, copy=False)

        if image.ndim == 2:
            gray_image = image
        elif image.ndim == 3 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3 and image.shape[2] == 4:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            return []

        gray_image = np.ascontiguousarray(gray_image, dtype=np.uint8)
        print(
            f"[FaceLandmarkDetector] image type={type(image)} "
            f"dtype={image.dtype} shape={getattr(image, 'shape', None)} "
            f"gray dtype={gray_image.dtype} shape={gray_image.shape}"
        )

        all_faces_landmarks = []

        if face_bbox is None:
            h, w = gray_image.shape[:2]
            rect = dlib.rectangle(left=0, top=0, right=w - 1, bottom=h - 1)
        else:
            x, y, w, h = face_bbox
            rect = dlib.rectangle(left=x, top=y, right=x + w - 1, bottom=y + h - 1)

        landmarks = self.landmark_predictor(gray_image, rect)
        landmark_points = np.zeros((68, 2), dtype=int)

        for i in range(68):
            landmark_points[i] = (landmarks.part(i).x, landmarks.part(i).y)
        all_faces_landmarks.append(landmark_points)

        if draw_connections:
            self._draw_landmark_connections(image, landmark_points)
        if draw_points:
            self._draw_landmark_points(image, landmark_points)
        if draw_rect:
            self._draw_bounding_box(image, rect)

        return all_faces_landmarks

    def _draw_landmark_connections(self, image: np.ndarray, points: np.ndarray):
        """
        Draws lines connecting facial landmark points based on facial regions.

        Args:
            image (np.ndarray): The image on which to draw.
            points (np.ndarray): Array of landmark points (shape: 68, 2).
        """
        colors = [
            (255, 255, 0),  # jaw
            (0, 255, 255),  # right eyebrow
            (0, 255, 255),  # left eyebrow
            (0, 255, 0),  # nose
            (0, 0, 255),  # right eye
            (0, 0, 255),  # left eye
            (0, 255, 0),  # outer mouth
            (0, 255, 0),  # inner mouth
        ]

        for i, (start, end) in enumerate(self.FACIAL_LANDMARKS_IDXS.values()):
            isClosed = i >= 4  # close eye and mouth regions
            cv2.polylines(image, [points[start:end]], isClosed, colors[i], 2)

    def _draw_landmark_points(self, image: np.ndarray, points: np.ndarray):
        """
        Draws small circles for each facial landmark point.

        Args:
            image (np.ndarray): The image on which to draw.
            points (np.ndarray): Array of landmark points (shape: 68, 2).
        """
        for x, y in points:
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

    def _draw_bounding_box(self, image: np.ndarray, face: dlib.rectangle):
        """
        Draws a bounding box around the detected face.

        Args:
            image (np.ndarray): The image on which to draw.
            face (dlib.rectangle): The rectangle representing the face.
        """
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    @staticmethod
    def calculate_distance(point1: tuple, point2: tuple) -> float:
        """
        Calculates the Euclidean distance between two 2D points.

        Args:
            point1 (tuple): First point (x, y).
            point2 (tuple): Second point (x, y).

        Returns:
            float: Euclidean distance.
        """
        return np.linalg.norm(np.array(point1) - np.array(point2))
