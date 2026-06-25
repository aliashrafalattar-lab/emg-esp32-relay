const int relayPin = 6;
const unsigned long baudRate = 230400;

void setup() {
  pinMode(relayPin, OUTPUT);
  digitalWrite(relayPin, LOW);  // LOW = OFF at startup

  Serial.begin(baudRate);
  Serial.println("ESP32 relay ready. Send ON or OFF.");
}

void loop() {
  if (!Serial.available()) {
    return;
  }

  String command = Serial.readStringUntil('\n');
  command.trim();
  command.toUpperCase();

  if (command == "ON") {
    digitalWrite(relayPin, HIGH);
    Serial.println("Relay ON");
  } else if (command == "OFF") {
    digitalWrite(relayPin, LOW);
    Serial.println("Relay OFF");
  } else if (command.length() > 0) {
    Serial.print("Unknown command: ");
    Serial.println(command);
  }
}
