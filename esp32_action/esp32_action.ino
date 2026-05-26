const int relayPin = 4;

// Your relay was behaving as active LOW earlier:
const int RELAY_ON = LOW;
const int RELAY_OFF = HIGH;

void setup() {
  pinMode(relayPin, OUTPUT);
  digitalWrite(relayPin, RELAY_OFF);

  Serial.begin(115200);
  delay(1000);

  Serial.println("ESP32-S3 ready");
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command == "ON") {
      digitalWrite(relayPin, RELAY_ON);
      Serial.println("Relay ON");
    }

    else if (command == "OFF") {
      digitalWrite(relayPin, RELAY_OFF);
      Serial.println("Relay OFF");
    }
  }
}