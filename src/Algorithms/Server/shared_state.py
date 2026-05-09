import threading

lock = threading.Lock()

# Frame
output_frame = None

# Metrics
fatigue_level = "LOW"
ear_value = 0.0
eye_height_value = 0.0
fps_value = 0.0
bpm_value = 0.0
frames_open = 0

# ADD THESE (IMPORTANT)
head_pose_status = "CENTER"
fatigue_log = ""
perclos_value = 0.0

# Used to reset PERCLOS history during a manual panic override
clear_history_flag = False