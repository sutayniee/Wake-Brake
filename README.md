# Wake-Brake
Wake&Brake is a real-time driver fatigue detection system developed as an undergraduate thesis project.  
The system uses computer vision techniques to monitor visual fatigue indicators and triggers a tri-modal alert system (haptic, olfactory, and auditory) when driver drowsiness is detected.

🛠 Requirements
- Python 3.9+
- Git
- OpenCV
- NumPy

1️⃣ Clone the repository
- git clone https://github.com/yourusername/WakeBrake.git
- cd WakeBrake

2️⃣ Create a virtual environment
- python -m venv venv

3️⃣ Activate the virtual environment
- Windows (PowerShell):
- venv\Scripts\Activate
- If activation is successful, you should see:
  (venv)

4️⃣ Install dependencies
- pip install -r requirements.txt

5️⃣ Verify installation
python -c "import cv2; import numpy; print('Setup successful')"

6️⃣ Select the virtual environment in VS Code (Important)
- Press Ctrl + Shift + P
- Select Python: Select Interpreter
- Choose the interpreter inside the venv folder

🔁 Basic Git Commands (Quick Reference)
Check repository status
- git status

Stage changes
- git add .

Commit changes
- git commit -m "Your commit message"

Push changes to GitHub
- git push

Pull latest updates
- git pull