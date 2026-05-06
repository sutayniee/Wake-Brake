import time

class BlinkRateDetector:
    def __init__(self, threshold=0.30, min_frames_closed=2, window_seconds=60):
        self.threshold = threshold
        self.min_frames_closed = min_frames_closed
        self.window_seconds = window_seconds

        self.frame_counter = 0
        self.blink_count = 0
        self.last_blink_state = False

        self.start_time = time.time()
        self.blink_rate = 0

    def update(self, ear):
        is_closed = ear < self.threshold

        # Count consecutive closed frames
        if is_closed:
            self.frame_counter += 1
        else:
            # Detect blink (closed → open transition)
            if self.frame_counter >= self.min_frames_closed:
                self.blink_count += 1
            self.frame_counter = 0

        # Compute blink rate every window
        current_time = time.time()
        elapsed = current_time - self.start_time

        if elapsed >= self.window_seconds:
            self.blink_rate = self.blink_count / (elapsed / 60)

            # Reset window
            self.blink_count = 0
            self.start_time = current_time

        return self.blink_rate