import time
import cv2
from flask import Flask, jsonify, Response
import shared_state  # IMPORTANT: import the module, not the variables

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
def get_fatigue():
    return jsonify({
        "level": shared_state.fatigue_level
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


# ----------------------------------
# WEB PAGE
# ----------------------------------
@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Wake&Brake Live Monitor</title>
    </head>
    <body style="text-align:center; font-family:Arial;">

        <h1>Wake&Brake Live Monitor</h1>

        <h2 id="status">Waiting for detection...</h2>

        <img src="/video" width="640" style="border:2px solid black;"/>

        <script>
            const source = new EventSource("/stream");

            source.onmessage = function(event) {
                document.getElementById("status").innerHTML =
                    "Fatigue Level: " + event.data;
            };

            source.onerror = function(error) {
                console.log("Stream error:", error);
            };
        </script>

    </body>
    </html>
    """


# ----------------------------------
# RUN SERVER
# ----------------------------------
def run_server():
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )


if __name__ == "__main__":
    run_server()