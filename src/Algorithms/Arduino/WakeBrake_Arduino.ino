// Pins based on your configuration
const int buzzerPin = 7;
const int vibrationPin =
    2; // Recommended: Use a PWM pin like D3 if you want intensity control
const int diffuserPin = 8;

// Timing Constants (in milliseconds)
const unsigned long scentOnDuration = 30000;  // 30 seconds
const unsigned long scentOffDuration = 90000; // 90 seconds

// Scent State Machine Definition
enum ScentState {
  IDLE,
  SPRAYING,
  COOLDOWN
};

// Global State Variables
bool fatigueActive = false;
ScentState currentScentState = IDLE;
unsigned long scentStartTime = 0;
unsigned long lastCommandTime = 0;

// Feature Toggles (Can be updated via Serial from your Mobile App)
bool useBuzzer = true;
bool useVibration = true;
bool useScent = true;

void setup() {
  Serial.begin(9600);
  pinMode(buzzerPin, OUTPUT);
  pinMode(vibrationPin, OUTPUT);
  pinMode(diffuserPin, OUTPUT);

  Serial.println("Wake&Brake System Ready - FSM Mode Active");
}

void loop() {
  handleSerialInput();

  // FAILSAFE: If no serial communication for 3 seconds, kill alarms
  if (fatigueActive && (millis() - lastCommandTime > 3000)) {
    fatigueActive = false;
    Serial.println("FAILSAFE: Connection to Python lost. Alarms disabled.");
  }

  if (fatigueActive) {
    executeAlerts();
  } else {
    stopAllAlerts();
  }

  // Scent cycle runs independently of fatigueActive to enforce the Locked Cycle
  handleScentCycle();
}

void handleSerialInput() {

  if (Serial.available()) {

    String command = Serial.readStringUntil('\n');

    command.trim();
    
    lastCommandTime = millis(); // Reset the watchdog timer on ANY command

    Serial.print("Received: ");
    Serial.println(command);

    // =========================
    // CONFIG COMMANDS
    // =========================

    if (command.startsWith("CFG")) {

  useBuzzer = command.indexOf("SOUND:1") != -1;
  useVibration = command.indexOf("VIB:1") != -1;
  useScent = command.indexOf("SCENT:1") != -1;

  Serial.println("=== CONFIG UPDATED ===");

  Serial.print("Sound: ");
  Serial.println(useBuzzer);

  Serial.print("Vibration: ");
  Serial.println(useVibration);

  Serial.print("Scent: ");
  Serial.println(useScent);

  // 🔥 FORCE APPLY IMMEDIATELY (CRITICAL FIX)
  if (!useBuzzer) digitalWrite(buzzerPin, LOW);
  if (!useVibration) digitalWrite(vibrationPin, LOW);
  if (!useScent) digitalWrite(diffuserPin, LOW);
}

    // =========================
    // ALERT SIGNALS
    // =========================

    else if (command == "S") {

      fatigueActive = true;

      if (useScent && currentScentState == IDLE) {

        Serial.println("SCENT: Triggering Active Spray (30s)");

        currentScentState = SPRAYING;

        scentStartTime = millis();

        digitalWrite(diffuserPin, HIGH);
      }
    }

    else if (command == "B") {

      fatigueActive = true;
    }

    else if (command == "H") {

      fatigueActive = true;
    }

    else if (command == "N" || command == "0") {

      fatigueActive = false;
    }

    else if (command == "X" || command == "KILL") {

      fatigueActive = false;
      currentScentState = IDLE;
      digitalWrite(diffuserPin, LOW);
      Serial.println("SYSTEM OVERRIDDEN: All alerts stopped.");
    }
  }
}

void executeAlerts() {

  if (useBuzzer)
    digitalWrite(buzzerPin, HIGH);
  else
    digitalWrite(buzzerPin, LOW);

  if (useVibration)
    digitalWrite(vibrationPin, HIGH);
  else
    digitalWrite(vibrationPin, LOW);
}

void handleScentCycle() {
  // 2. Pulsed Olfactory State Machine (30s ON / 90s OFF Locked Cycle)
  unsigned long currentTime = millis();

  switch (currentScentState) {
    case IDLE:
      digitalWrite(diffuserPin, LOW); // Ensure it stays OFF
      break;

    case SPRAYING:
      digitalWrite(diffuserPin, HIGH); // Ensure it stays ON
      if (currentTime - scentStartTime >= scentOnDuration) {
        // Transition to COOLDOWN
        Serial.println("SCENT: Entering Cooldown Lockout (90s)");
        currentScentState = COOLDOWN;
        scentStartTime = currentTime;
        digitalWrite(diffuserPin, LOW);
      }
      break;

    case COOLDOWN:
      digitalWrite(diffuserPin, LOW); // Ensure it stays OFF
      if (currentTime - scentStartTime >= scentOffDuration) {
        // Transition back to IDLE
        Serial.println("SCENT: Cooldown Complete. Ready.");
        currentScentState = IDLE;
      }
      break;
  }
}

void stopAllAlerts() {

  digitalWrite(buzzerPin, LOW);
  digitalWrite(vibrationPin, LOW);
}