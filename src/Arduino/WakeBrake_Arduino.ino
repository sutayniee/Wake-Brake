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

  if (fatigueActive) {
    executeAlerts();
  } else {
    stopAllAlerts();
  }

  // Scent cycle runs independently of fatigueActive to enforce the Locked Cycle
  handleScentCycle();
}

void handleSerialInput() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    Serial.print("Received: "); Serial.println(cmd); // Debugging

    if (cmd == 'S') { // SEVERE - Scent trigger
      fatigueActive = true;
      useBuzzer = true; useVibration = true;
      // Trigger the locked cycle if it's currently idle
      if (currentScentState == IDLE) {
          Serial.println("SCENT: Triggering Active Spray (30s)");
          currentScentState = SPRAYING;
          scentStartTime = millis();
          digitalWrite(diffuserPin, HIGH);
      }
    } 
    else if (cmd == 'B') { // CRITICAL - Buzzer and Vibration
      fatigueActive = true;
      useBuzzer = true; useVibration = true;
    }
    else if (cmd == 'H') { // WARNING - Haptic Only
      fatigueActive = true;
      useBuzzer = false; useVibration = true;
    }
    else if (cmd == 'N' || cmd == '0') { // SAFE
      fatigueActive = false;
      // stopAllAlerts() will be called in loop, but scent will continue!
    }
  }
}

void executeAlerts() {
  // 1. Immediate Alerts (Auditory and Haptic)
  if (useBuzzer)
    digitalWrite(buzzerPin, HIGH);
  if (useVibration)
    digitalWrite(vibrationPin, HIGH);
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
  // diffuserPin is NOT turned off here! It is managed exclusively by handleScentCycle()
}