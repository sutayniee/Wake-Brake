import playsound3 as ps3
import threading

alert_playing = False
lock = threading.Lock()

def play_sound(Path_Alarm):
    global alert_playing
    try:
        ps3.playsound(Path_Alarm)
    finally:
        with lock:
            alert_playing = False

def play_alert(audio_path="pogi-gising-na.mp3"):
    global alert_playing
    with lock:
        if alert_playing:
            return
        alert_playing = True

    threading.Thread(
        target=play_sound,
        args=(audio_path,),
        daemon=True
    ).start()