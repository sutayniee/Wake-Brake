import logging
import os
import psutil
import time
import Algorithms.Server.shared_state as shared_state


# =========================
# FATIGUE LOGGER SETUP
# =========================
def setup_logger():
    LOG_FILE = "system_fatigue_log.csv"
    file_exists = os.path.isfile(LOG_FILE)

    logger = logging.getLogger("fatigue_logger")
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():  # FIXED SAFER CHECK
        handler = logging.FileHandler(LOG_FILE)

        formatter = logging.Formatter(
            "%(asctime)s,%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

        if not file_exists:
            logger.info(
                "EVENT_TYPE,CONFIDENCE_SCORE,PERCLOS,EAR,BLINK_RATE,PITCH_RATIO,FPS,CPU_PERCENT,MEMORY_PERCENT"
            )

    return logger


# =========================
# SYSTEM PERFORMANCE LOGGER
# =========================
def log_system_performance(logger):

    print("[SYSTEM LOGGER] Started")

    while True:
        try:
            cpu = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory().percent
            fps = getattr(shared_state, "fps_value", 0.0)

            logger.info(
                f"SYSTEM,0,0,0,0,0,{fps},{cpu},{memory}"
            )

            time.sleep(10)

        except Exception as e:
            print("[SYSTEM LOGGER ERROR]", e)
            time.sleep(2)


# =========================
# LATENCY LOGGER SETUP
# =========================
def setup_latency_logger():
    LOG_FILE = "arduino_latency_log.csv"
    file_exists = os.path.isfile(LOG_FILE)

    logger = logging.getLogger("latency_logger")
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        handler = logging.FileHandler(LOG_FILE)

        formatter = logging.Formatter(
            "%(asctime)s,%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

        if not file_exists:
            logger.info("EVENT_TYPE,DETECTION_TIME,ARDUINO_TIME,LATENCY_MS")

    return logger


# =========================
# ARDUINO LATENCY TRACKER
# =========================
def log_arduino_latency(logger, event_type, send_func):

    detection_time = time.time()

    send_func()

    arduino_time = time.time()

    latency_ms = (arduino_time - detection_time) * 1000

    logger.info(
        f"{event_type},"
        f"{detection_time:.6f},"
        f"{arduino_time:.6f},"
        f"{latency_ms:.2f}"
    )