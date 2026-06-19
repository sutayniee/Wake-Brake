# Wake&Brake

Wake&Brake is a real-time driver fatigue detection system designed to run on a Raspberry Pi with an Arduino acting as a haptic actuator. It combines multiple computer-vision algorithms (Haar cascades, dlib facial landmarks, eye-aspect-ratio, blink-rate and PERCLOS calculations) to detect micro-sleeps and sustained drowsiness and escalate alerts via sound, vibration, buzzer and an optional scent actuator.

---

## Key features

- Real‑time face detection (Haar cascade)
- Facial landmark detection (dlib 68-point predictor)
- Eye Aspect Ratio (EAR) for eye-closure detection
- Blink-rate estimation
- PERCLOS (percentage of eye closure over a time window) for macro-fatigue detection
- Head posture (pitch) estimation for head-down detection
- Escalating multimodal alerts via Arduino (vibration, buzzer, scent)
- Local Flask server to stream stats and video frames
- Performance logging for post-trip analysis

---

## Repository structure

- src/
  - main.py                — main application loop (capture, detection, alerting, streaming)
  - Algorithms/            — detection and hardware integration modules
    - Haar_Cascade/        — face detection utilities
    - Eye_Aspect_Ratio/    — EAR computation and landmark wrapper
    - Blink_Rate/          — blink-rate estimator
    - Arduino/             — Arduino/serial communication helpers
    - Logs/                — logging helpers
    - Server/              — Flask server and shared state
  - Sample_Alarm/          — example alarm playback utility

---

## Requirements

This project uses Python and has been developed to run on Raspberry Pi-class hardware. The primary dependencies are listed in `requirements.txt` and include:

- OpenCV (cv2)
- dlib
- numpy
- flask
- pyserial

Install dependencies (prefer a virtual environment):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Notes for Raspberry Pi: use the OS package manager to install any system dependencies (build tools) required for dlib and OpenCV before pip-installing.

---

## Hardware setup

- Raspberry Pi (tested on Raspberry Pi 3/4)
- USB-connected Arduino (haptic actuator, buzzer, optional scent module)

Connect your Arduino and identify its serial device (e.g. `/dev/ttyACM0` or `/dev/ttyUSB0`). The default serial port used by the code is `/dev/ttyACM0` — edit `src/Algorithms/Arduino/Arduino_Signal.py` if your port differs, or set up a udev rule / symlink for consistent naming.

---

## Configuration

- Adjust thresholds and windows inside `src/main.py`:
  - `EAR_THRESHOLD` — eye-closure threshold (adapted after calibration)
  - `PERCLOS_WINDOW` — window in seconds for PERCLOS calculation
  - `PERCLOS_THRESHOLD` — PERCLOS threshold to trigger macro alerts

- Enable/disable actuators via `Algorithms/Server/shared_state.py` values (sound_enabled, vibration_enabled, scent_enabled).

---

## Usage

Run the main application (preferably inside the virtual environment):

```bash
cd src
python3 main.py
```

The app opens a camera window and starts a small Flask server (background thread) exposing monitoring endpoints and a video stream. Press `q` in the OpenCV window to quit.

If the app fails to find the Arduino, it will print available serial ports. Connect the Arduino and restart the app.

---

## Performance tips (embedded / Raspberry Pi)

This project performs CPU-intensive computer-vision tasks. If you experience low fps or high CPU usage:

- Lower the camera capture resolution (set `cv2.VideoCapture` properties).
- Downscale frames for detection (run detection on a smaller copy) and run the dlib predictor on a cropped face ROI.
- Run heavy detection/landmarking less frequently (every N frames) and track between detections.
- Share JPEG-encoded frames instead of raw BGR images when streaming to reduce memory/copy costs.
- Use asynchronous logging or rate-limit prints; avoid blocking I/O on the main loop.

Profiling on the target hardware (cProfile, py-spy) is recommended to find the biggest bottlenecks.

---

## License

This repository is licensed under the MIT License. See the `LICENSE` file in the repository root for details.

---

## Contributing

Contributions are welcome. Please open an issue or submit a pull request with a clear description of the change. For larger changes (new detection backends, performance refactors), open an issue first so we can discuss design.

---

## Contact

If you have questions or want to collaborate, create an issue or contact the repository owner: `sutayniee`.
