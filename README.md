# Edge AI-based Predictive Street Lighting — TinyML Regression

## Overview

This repository contains an ESP32 implementation of a TinyML-based predictive street lighting controller. A trained TensorFlow Lite model is embedded in the firmware (src/model.h). The ESP32 reads three analog inputs (simulated traffic, distance and ambient light), normalizes them to the training range, runs an on-device inference and sets PWM to drive a street-light LED. Data logging to ThingSpeak is included.

## Status

- Source: src/main.ino (Arduino/PlatformIO)
- Embedded model: src/model.h (g_model array)
- PlatformIO configuration: platformio.ini (env: esp32, lib_deps includes ThingSpeak and TensorFlowLite_ESP32)
- Wokwi simulation files: src/wokwi-* (if you want to run in Wokwi)
- Note: temp_preprocessed.cpp appears empty and can be removed or filled if required.

## Requirements

- Hardware: ESP32 development board (board "esp32dev" in platformio.ini) or compatible
- Software:
  - Visual Studio Code + PlatformIO extension (recommended) OR PlatformIO CLI
- PlatformIO will automatically fetch these library dependencies declared in platformio.ini:
  - ThingSpeak
  - TensorFlowLite_ESP32

## Quick setup (recommended)

1. Install Visual Studio Code
2. Install the PlatformIO extension
3. Open this project folder in VS Code
4. Let PlatformIO index the project and install libraries declared in platformio.ini

## Configure device-specific settings

Before building, configure these values in src/main.ino (or prefer the secrets approach below):

- WiFi SSID / password:
    const char* ssid = "Wokwi-GUEST";
    const char* password = "";

- ThingSpeak channel and API key (used for data logging):
    unsigned long myChannelNumber = 3437665;
    const char* myWriteAPIKey = "QVXPDIWR83IXXUH8";

Recommended: move secrets to src/secrets.h (gitignored)

## Create a new file src/secrets.h (do NOT commit it):

#ifndef SECRETS_H
#define SECRETS_H

const char* WIFI_SSID = "your-ssid";
const char* WIFI_PASSWORD = "your-password";
unsigned long THINGSPEAK_CHANNEL = 1234567;
const char* THINGSPEAK_WRITE_API_KEY = "YOUR_WRITE_API_KEY";

#endif

Then update src/main.ino to include "secrets.h" and use these constants.

## Build (PlatformIO)

Using VS Code PlatformIO UI:
- Open the PlatformIO sidebar (left) -> Project Tasks -> esp32 -> General -> Build

Using PlatformIO CLI (from the project root):
- Build:  platformio run
