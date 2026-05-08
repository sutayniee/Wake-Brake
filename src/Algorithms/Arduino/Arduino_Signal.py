import serial 
import time
import serial.tools.list_ports

arduino = None
arduino_connected = False 
ports = list(serial.tools.list_ports.comports())

def check_arduino_connection():
    try:
        # Replace 'COM' value with your actual port!
        arduino = serial.Serial(port='COM3', baudrate=9600, timeout=1)
        time.sleep(2)  # Allow Arduino to initialize
        arduino_connected = True
        print("Arduino connected successfully.")
    except Exception as e:
        print("Arduino not connected. Running in standalone mode.")
        for p in ports:
            print(p.device)
            
def send_to_arduino(signal):
    if arduino_connected and arduino is not None:
        try:
            arduino.write(signal.encode())
        except:
            print("Warning: Failed to send data to Arduino.")