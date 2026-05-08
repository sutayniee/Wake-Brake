"""
Blink Rate Detection using EAR

This file imports the EAR function from the existing EAR module
and uses it to determine whether the eyes are closed.

Condition:
- If EAR < 0.10 → eye is considered closed
- Uses consecutive frame counting for blink detection
"""

from Algorithms.Eye_Aspect_Ratio.Eye_Aspect_Ratio_main import eye_aspect_ratio

# EAR threshold
EAR_THRESHOLD = 0.30

# Number of consecutive frames required
CONSECUTIVE_FRAMES = 3

# Counter for closed-eye frames
closed_eye_frames = 0


def check_blink_rate(eye_landmarks):
    """
    Uses EAR from the existing EAR module
    and determines if the eye is closed.

    Args:
        eye_landmarks: Array of 6 eye landmark points

    Returns:
        tuple:
            ear (float)
            eye_closed (bool)
    """

    global closed_eye_frames

    # Call EAR function from your existing file
    ear = eye_aspect_ratio(eye_landmarks)

    # Check threshold
    if ear < EAR_THRESHOLD:
        closed_eye_frames += 1
    else:
        closed_eye_frames = 0

    # Determine if eye is considered closed
    eye_closed = closed_eye_frames >= CONSECUTIVE_FRAMES

    return ear, eye_closed