import time
import cv2
from flask import Flask, jsonify, Response
from shared_state import fatigue_level, output_frame, lock

app = Flask(__name__)

# -------- VIDEO STREAM --------
def generate_frames():
    global output_frame

    while True:
        with lock:
            if output_frame is None:
                continue

            _, buffer = cv2.imencode('.jpg', output_frame)
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes + b'\r\n')

# -------- ROUTES --------
@app.route('/fatigue')
def get_fatigue():
    return jsonify({"level": fatigue_level})

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream')
def stream():
    def generate():
        while True:
            yield f"data: {fatigue_level}\n\n"
            time.sleep(1)

    return app.response_class(generate(), mimetype='text/event-stream')

@app.route('/')
def index():
    return """
    <html>
    <body style="text-align:center;">
        <h1>Wake&Brake Live Monitor</h1>
        <h2 id="status">Waiting...</h2>
        <img src="/video" width="640"/>

        <script>
            var source = new EventSource("/stream");
            source.onmessage = function(event) {
                document.getElementById("status").innerHTML = event.data;
            };
        </script>
    </body>
    </html>
    """

def run_server():
    app.run(host="0.0.0.0", port=5000)