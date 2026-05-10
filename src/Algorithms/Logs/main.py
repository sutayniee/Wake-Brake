import logging
import os

def setup_logger():
    LOG_FILE = 'fatigue_log.csv'
    file_exists = os.path.isfile(LOG_FILE)

    logger = logging.getLogger("fatigue_logger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(LOG_FILE)
        formatter = logging.Formatter(
            '%(asctime)s,%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        if not file_exists:
            logger.info("EVENT_TYPE,CONFIDENCE_SCORE,PERCLOS,EAR,BLINK_RATE,PITCH_RATIO,FPS")

    return logger