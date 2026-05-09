import time
import cv2
from flask import Flask, jsonify, Response
import Algorithms.Server.shared_state as shared_state  # IMPORTANT: import the module, not the variables
import socket
import Algorithms.Server.shared_state as shared_state
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody

app = Flask(__name__)

# ----------------------------------
# VIDEO STREAM
# ----------------------------------
def generate_frames():
    while True:
        with shared_state.lock:
            if shared_state.output_frame is None:
                continue

            # encode frame as jpg
            ret, buffer = cv2.imencode('.jpg', shared_state.output_frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes +
            b'\r\n'
        )


# ----------------------------------
# FATIGUE JSON API
# ----------------------------------
@app.route('/fatigue')
@app.route("/fatigue")
def fatigue():
    return jsonify({
        "bpm": shared_state.bpm_value,
        "ear": shared_state.ear_value,
        "eye_height": shared_state.eye_height_value,
        "fps": shared_state.fps_value,
        "level": shared_state.fatigue_level,
        "head_pose": shared_state.head_pose_status
    })
# ----------------------------------
# VIDEO ROUTE
# ----------------------------------
@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ----------------------------------
# LIVE STATUS STREAM (AUTO UPDATE)
# ----------------------------------
@app.route('/stream')
def stream():
    def generate():
        last_value = ""

        while True:
            with shared_state.lock:
             current_value = shared_state.fatigue_level

            # only send update if changed
            if current_value != last_value:
                yield f"data: {current_value}\n\n"
                last_value = current_value

            time.sleep(1)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )
from flask import request

#Arduino Updates
@app.route('/config', methods=['POST'])
def update_config():

    data = request.json

    sound = 1 if data.get("sound") else 0
    vibration = 1 if data.get("vibration") else 0
    scent = 1 if data.get("scent") else 0

    command = f"CFG,SOUND:{sound},VIB:{vibration},SCENT:{scent}\n"

    print("Sending config:", command)

    send_to_arduino(command)

    return jsonify({
        "status": "success",
        "command": command
    })


# ----------------------------------
# WEB PAGE
# ----------------------------------
@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Wake&Brake Monitor</title>
    </head>

    <body style="text-align:center; font-family:Arial;">

        <h1>Wake&Brake Live Monitor</h1>

        <h2 id="status">Loading...</h2>

        <p id="metrics">FPS: -- | EAR: -- | BPM: -- | Eye Height: --</p>

        <img src="/video" width="640" style="border:2px solid black;" />

        <script>
            const source = new EventSource("/stream");

            source.onmessage = function(event) {
                document.getElementById("status").innerHTML =
                    "Fatigue Level: " + event.data;
            };

            setInterval(async () => {
                const res = await fetch("/fatigue");
                const data = await res.json();

                document.getElementById("metrics").innerHTML =
                "FPS: " + data.fps +
                " | EAR: " + data.ear +
                " | BPM: " + data.bpm +
                " | Eye Height: " + data.eye_height +
                " | PERCLOS: " + (data.perclos * 100).toFixed(1) + "%";

         }, 500);

        </script>

    </body>
    </html>
    """
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # doesn't actually send data
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

# ----------------------------------
# RUN SERVER
# ----------------------------------
def run_server():
    import logging
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

    ip = get_local_ip()

    print(f"\nServer running on:")
    print(f"  http://127.0.0.1:5000")
    print(f"  http://{ip}:5000\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )


if __name__ == "__main__":
    run_server()