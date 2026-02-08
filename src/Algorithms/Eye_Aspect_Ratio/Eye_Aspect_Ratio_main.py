"""
Drowsiness Detection using Eye Aspect Ratio (EAR)

This script performs real-time drowsiness detection using facial landmarks.
If the eyes remain closed for a set number of consecutive frames, a warning is triggered.

Requirements:
- OpenCV (cv2)
- dlib
- numpy
- face_landmark_detector.py (custom class)
- Pretrained dlib model: shape_predictor_68_face_landmarks.dat
"""

import cv2
import numpy as np

# Helper Functions
# -----------------
def eye_aspect_ratio(eye: np.ndarray) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) to detect eye closure.
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)


    Args:
        eye (np.ndarray): Array of 6 (x, y) coordinates representing the eye landmarks.

    Returns:
        float: Eye Aspect Ratio (EAR) value.
    """
    # Vertical distances
    vertical_1 = np.linalg.norm(eye[1] - eye[5])
    vertical_2 = np.linalg.norm(eye[2] - eye[4])

    # Horizontal distance
    horizontal = np.linalg.norm(eye[0] - eye[3])

    # EAR formula
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def put_text(frame, text, position, color=(0, 0, 255), font_scale=0.7, thickness=2):
    """
    Utility function to draw text on a video frame.

    Args:
        frame (np.ndarray): Image frame.
        text (str): Text to draw.
        position (tuple): (x, y) position on the frame.
        color (tuple): Text color (BGR).
        font_scale (float): Font size.
        thickness (int): Text thickness.
    """
    cv2.putText(
        frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness
    )
