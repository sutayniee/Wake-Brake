import playsound3 as ps3
import threading
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_CASCADE_SOUND = (_ROOT / "soundbeat.mp3")

alert_playing = False
lock = threading.Lock()

def play_sound(_CASCADE_SOUND):
    global alert_playing
    try:
        ps3.playsound(_CASCADE_SOUND)
    finally:
        with lock:
            alert_playing = False
#CHANGE MP3/XAV file to prefered sound
def play_alert():
    global alert_playing
    with lock:
        if alert_playing:
            return
        alert_playing = True

    threading.Thread(
        target=play_sound,
        args=(_CASCADE_SOUND,),
        daemon=True
    ).start()