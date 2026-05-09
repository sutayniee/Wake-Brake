import tkinter as tk
from datetime import datetime
import threading
import queue

class HardwareDashboard:
    def __init__(self):
        self.root = None
        self.q = queue.Queue()
        
        # Configuration Tracking
        self.use_buzzer = True
        self.use_vibration = True
        self.use_scent = True
        
        # Run Tkinter in its own thread to prevent blocking OpenCV main loop
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        self.root = tk.Tk()
        self.root.title("Wake&Brake | Virtual Hardware Dashboard")
        self.root.geometry("500x400")
        self.root.configure(bg="#1e1e1e")

        # Table Headers
        tk.Label(self.root, text="COMPONENT", fg="white", bg="#1e1e1e", font=("Arial", 12, "bold")).grid(row=0, column=0, padx=20, pady=10)
        tk.Label(self.root, text="STATUS", fg="white", bg="#1e1e1e", font=("Arial", 12, "bold")).grid(row=0, column=1, padx=20, pady=10)

        # Status Labels
        self.buzzer_label = tk.Label(self.root, text="[LOW 🔇]", fg="#555555", bg="#1e1e1e", font=("Arial", 12))
        self.buzzer_label.grid(row=1, column=1)
        tk.Label(self.root, text="BUZZER (Pin 7)", fg="white", bg="#1e1e1e").grid(row=1, column=0)

        self.vib_label = tk.Label(self.root, text="[LOW 💤]", fg="#555555", bg="#1e1e1e", font=("Arial", 12))
        self.vib_label.grid(row=2, column=1)
        tk.Label(self.root, text="VIBRATION (Pin 2)", fg="white", bg="#1e1e1e").grid(row=2, column=0)

        self.scent_label = tk.Label(self.root, text="[IDLE 🛑]", fg="#555555", bg="#1e1e1e", font=("Arial", 12))
        self.scent_label.grid(row=3, column=1)
        tk.Label(self.root, text="HUMIDIFIER (Pin 8)", fg="white", bg="#1e1e1e").grid(row=3, column=0)

        # Live Log
        self.log_box = tk.Text(self.root, height=10, width=55, bg="black", fg="#00ff00", font=("Consolas", 10))
        self.log_box.grid(row=4, column=0, columnspan=2, padx=10, pady=20)
        
        self._update_log("Dashboard Initialized. Standby for Serial...")
        
        # Schedule the queue processor
        self.root.after(100, self._process_queue)
        
        self.root.mainloop()

    def _process_queue(self):
        try:
            while True:
                command = self.q.get_nowait()
                self._update_state_internal(command)
        except queue.Empty:
            pass
        finally:
            if self.root:
                self.root.after(100, self._process_queue)

    def _update_log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_box.see(tk.END)

    def update_state(self, command):
        # Thread-safe entry point for updating the dashboard
        command = command.strip()
        self.q.put(command)

    def _update_state_internal(self, command):
        if command.startswith('CFG'):
            self.use_buzzer = 'SOUND:1' in command
            self.use_vibration = 'VIB:1' in command
            self.use_scent = 'SCENT:1' in command
            
            self._update_log("Configuration Updated")
            
            if not self.use_buzzer:
                self.buzzer_label.config(text="[DISABLED ❌]", fg="#555555")
            else:
                self.buzzer_label.config(text="[LOW 🔇]", fg="#555555")
                
            if not self.use_vibration:
                self.vib_label.config(text="[DISABLED ❌]", fg="#555555")
            else:
                self.vib_label.config(text="[LOW 💤]", fg="#555555")
                
            if not self.use_scent:
                self.scent_label.config(text="[DISABLED ❌]", fg="#555555")
            else:
                self.scent_label.config(text="[IDLE 🛑]", fg="#555555")
            return

        if command == 'X':
            self._update_log("🚨 MANUAL OVERRIDE: KILLING ALL PINS")
            if self.use_buzzer: self.buzzer_label.config(text="[LOW 🔇]", fg="#555555")
            if self.use_vibration: self.vib_label.config(text="[LOW 💤]", fg="#555555")
            if self.use_scent: self.scent_label.config(text="[IDLE 🛑]", fg="#555555")
        elif command == 'S':
            self._update_log("SEVERE: Scent Spraying")
            if self.use_scent: self.scent_label.config(text="[SPRAYING 💨]", fg="#33ff33")
            if self.use_buzzer: self.buzzer_label.config(text="[LOW 🔇]", fg="#555555")
            if self.use_vibration: self.vib_label.config(text="[LOW 💤]", fg="#555555")
        elif command == 'B':
            self._update_log("CRITICAL: Buzzer Activated")
            if self.use_buzzer: self.buzzer_label.config(text="[HIGH 🔊]", fg="#ff3333")
            if self.use_vibration: self.vib_label.config(text="[LOW 💤]", fg="#555555")
            if self.use_scent: self.scent_label.config(text="[IDLE 🛑]", fg="#555555")
        elif command == 'H':
            self._update_log("WARNING: Vibration Activated")
            if self.use_buzzer: self.buzzer_label.config(text="[LOW 🔇]", fg="#555555")
            if self.use_vibration: self.vib_label.config(text="[HIGH 📳]", fg="#ff9900")
            if self.use_scent: self.scent_label.config(text="[IDLE 🛑]", fg="#555555")
        elif command == 'N' or command == '0':
            self._update_log("SAFE: System Normal")
            if self.use_buzzer: self.buzzer_label.config(text="[LOW 🔇]", fg="#555555")
            if self.use_vibration: self.vib_label.config(text="[LOW 💤]", fg="#555555")
            if self.use_scent: self.scent_label.config(text="[IDLE 🛑]", fg="#555555")

# Create a singleton instance that initializes when the module is imported
dashboard = HardwareDashboard()
