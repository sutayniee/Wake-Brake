#include <Wire.h> 
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);

// Pin Definitions
const int buzzerPin = 7; 
const int buzzerNeg = 4; 
const int MoPin = 2;    

// Global state variables
bool isDrowsy = false;
bool lastState = true; // Set to true so it forces an update on the very first loop

void setup() {
  lcd.init();
  lcd.backlight();
  
  pinMode(buzzerPin, OUTPUT);
  pinMode(buzzerNeg, OUTPUT);
  digitalWrite(buzzerNeg, LOW); // Acting as GND
  pinMode(MoPin, OUTPUT);  
  
  // High-speed communication with Python
  Serial.begin(9600); 

  // Startup sequence
  lcd.setCursor(0, 0);
  lcd.print(" WAKE & BRAKE  ");
  lcd.setCursor(0, 1);
  lcd.print(" AI CAM ACTIVE ");
  delay(2000);
  lcd.clear();
}

void loop() {
  // 1. --- Listen for Python Signal ---
  if (Serial.available() > 0) {
    char data = Serial.read();
    
    // Clear out any extra characters in the buffer (like newline characters)
    while(Serial.available() > 0) { Serial.read(); } 
    
    if (data == '1') {
      isDrowsy = true;
    } 
    else if (data == '0') {
      isDrowsy = false;
    }
  }

  // 2. --- Display and Hardware Logic (ONLY UPDATES IF STATE CHANGES) ---
  if (isDrowsy != lastState) {
    
    if (isDrowsy) {
      // --- Alert State ---
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("!!! DROWSY !!!  "); 
      lcd.setCursor(0, 1);
      lcd.print("   WAKE UP!     ");

      digitalWrite(buzzerPin, HIGH); 
      digitalWrite(MoPin, HIGH);     
    } 
    else {
      // --- Normal State ---
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("  DRIVING MODE  "); 
      lcd.setCursor(0, 1);
      lcd.print("STATUS: ALERT   ");
      
      digitalWrite(buzzerPin, LOW);
      digitalWrite(MoPin, LOW);      
    }
    
    // Save the current state so we don't update again until it changes
    lastState = isDrowsy; 
  }
  
  delay(10); // Keep the small delay for loop stability
}