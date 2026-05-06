# Wake-Brake
Wake&Brake is a real-time driver fatigue detection system developed as an undergraduate thesis project.  
The system uses computer vision techniques to monitor visual fatigue indicators and triggers a tri-modal alert system (haptic, olfactory, and auditory) when driver drowsiness is detected.

🛠 Requirements
- Python 3.9+
- Git
- OpenCV
- NumPy
- dlib 
- pyserial

1. Clone the repository
- git clone https://github.com/sutayniee/Wake-Brake.git
- cd WakeBrake

2. Create a virtual environment
- python -m venv venv

3. Activate the virtual environment
▶ Command Prompt (CMD)
- venv\Scripts\activate

▶ PowerShell (VS Code default)
- venv\Scripts\Activate.ps1

If PowerShell blocks it, run once:
- Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

Then try activating again.
If activation is successful, you should see:
  (venv)

4. Install dependencies
- pip install -r requirements.txt

5. Installing dlib
- cd dlib
- python -m pip install dlib-19.24.1-cp311-cp311-win_amd64.whl

5. Verify installation
- python -c "import cv2, numpy, dlib; print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__); print('dlib:', dlib.__version__)"

6. Select the virtual environment in VS Code (Important)
- Press Ctrl + Shift + P
- Select Python: Select Interpreter
- Choose the interpreter inside the venv folder

💿 Arduino IDE Setup (Firmware)

Installation
1. Download the IDE Go to the Official Arduino Website (https://www.arduino.cc/en/software) and download *Arduino IDE 2.0* (or higher) for Windows.

2. Install: Run the .exe and follow the prompts. Ensure you allow the installation of "USB Drivers" if Windows asks.

3. Install LiquidCrystal I2C Library:
   - Open Arduino IDE.
   - Go to Sketch > Include Library > Manage Libraries...
   - Search for *"LiquidCrystal I2C"* by Frank de Brabander.
   - Click Install.


Uploading the Code
1. Connect: Plug your Arduino Uno into your laptop via USB.

2. Open Firmware: In the Arduino IDE, go to File > Open and select the .ino file located in `arduino_code/` folder.
Note: You can either open the .ino file directly from the arduino_code/ folder or simply copy-paste the code from VS Code into a new sketch in the Arduino IDE. Ensure the previous default code is deleted before pasting and saving.

3. Select Board & Port:
   - Go to Tools > Board and select Arduino Uno.
   - Go to Tools > Port and select the active COM port (e.g., COM4).
4. Upload: Click the Upload arrow button (top left).


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

Switch branch
- git switch (name of existing branch)

Create and switch to the branch
- git checkout -b (name of branch)

Delete branch
- git branch -d (branch_name)

Steps to Merge:
- Switch to the main branch: git checkout main
- Pull latest changes: git pull origin main (to avoid conflicts)
- Merge the feature branch: git merge your-feature-branch
- Push the changes: git push origin main 
