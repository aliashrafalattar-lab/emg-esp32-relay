# EMG ESP32 Relay

Adaptive EMG signal viewer and ESP32 relay control prototype for supervised stimulation testing.

This project reads EMG data from a Backyard Brains-style serial device, displays the live signal in a browser, and sends ON / OFF serial commands to an ESP32 relay controller.

## Files

- byb_signal_viewer.py - Web app for live EMG viewing, measurement, and adaptive stimulation workflow
- esp32_action/esp32_action.ino - ESP32 sketch that controls a relay from serial commands
- unnamed.png - Project image / interface asset

## Features

- Live EMG signal plotting
- Manual mode for supervised relay testing
- Smart mode for automated stimulation/rest timing
- RMS and median frequency analysis
- Serial communication with an ESP32 relay controller
- Setup, safety, science, and troubleshooting views

## Hardware

This prototype expects:

- EMG acquisition device connected over serial
- ESP32 or ESP32-S3 connected over serial
- Relay module connected to the ESP32
- Relay signal pin connected to GPIO 6 by default

The ESP32 sketch uses:

const int relayPin = 6;
const unsigned long baudRate = 230400;

Serial commands:

ON
OFF

## Software Requirements

Python 3.10+ recommended.

Install dependencies:

pip install numpy scipy pyserial dash plotly

## ESP32 Setup

1. Open esp32_action/esp32_action.ino in the Arduino IDE.
2. Select the correct ESP32 board and serial port.
3. Upload the sketch.
4. Confirm the relay starts OFF.
5. The ESP32 listens at 230400 baud for ON and OFF.

## Running the App

Default ports:

- EMG device: COM4
- ESP32: COM5
- Web app: http://127.0.0.1:8060

Run:

python byb_signal_viewer.py --port COM4 --esp-port COM5

If your devices use different ports, change the command:

python byb_signal_viewer.py --port COM3 --esp-port COM7

Then open:

http://127.0.0.1:8060

## Modes

### Manual Mode

Manual mode is for supervised setup, debugging, and relay testing. Use it to verify that the EMG signal is visible and that the ESP32 relay responds correctly.

### Smart Mode

Smart mode runs an automated stimulation/rest protocol using EMG-derived measurements such as RMS and median frequency.

## Safety Notice

This is a research and educational prototype. It is not a medical device and should not be used for clinical treatment or unsupervised stimulation.

Use appropriate electrical isolation, current limiting, supervision, and institutional safety procedures when working with human subjects or stimulation hardware.

## Troubleshooting

If the app cannot connect:

- Check that the EMG and ESP32 serial ports are correct.
- Close Arduino Serial Monitor, Serial Plotter, Spike Recorder, or any other program using the same port.
- Confirm the ESP32 sketch was uploaded successfully.
- Confirm the ESP32 baud rate is 230400.
- Confirm the EMG device baud rate matches the Python app default.

If the relay does not switch:

- Check the relay signal wire is connected to GPIO 6.
- Check relay power and ground.
- Send ON and OFF manually from a serial monitor at 230400 baud.
- Adjust relayPin in esp32_action.ino if using a different GPIO.

## Repository

https://github.com/aliashrafalattar-lab/emg-esp32-relay
