import logging
import os
import psutil
import time
import Algorithms.Server.shared_state as shared_state


# =========================
# LOGGER SETUP (FATIGUE + SYSTEM USE SAME FILE OR SEPARATE)
# =========================
def setup_logger():
    LOG_FILE = "system_fatigue_log.csv"
    file_exists = os.path.isfile(LOG_FILE)

    logger = logging.getLogger("fatigue_logger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(LOG_FILE)

        formatter = logging.Formatter(
            "%(asctime)s,%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # CSV HEADER (ONLY ONCE)
        if not file_exists:
            logger.info(
                "EVENT_TYPE,CONFIDENCE_SCORE,PERCLOS,EAR,BLINK_RATE,PITCH_RATIO,FPS,CPU (%),MEMORY (%)"
            )

    return logger


# =========================
# SYSTEM PERFORMANCE LOGGER THREAD
# =========================
def log_system_performance(logger):

    print("[SYSTEM LOGGER] Started successfully")

    while True:
        try:
            cpu = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory().percent

            # SAFE FPS READ
            fps = getattr(shared_state, "fps_value", 0.0)

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            print(f"[SYSTEM LOGGER] FPS={fps}, CPU={cpu}, MEM={memory}")

            # IMPORTANT: correct CSV alignment
            logger.info(
                f"SYSTEM,,,,,,{fps},{cpu},{memory}"
            )

            time.sleep(10)

        except Exception as e:
            print("[SYSTEM LOGGER ERROR]", e)
            time.sleep(2)