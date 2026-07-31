#include <WiFi.h>
#include <ThingSpeak.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"

const int POT1_PIN = 34; // Simulated Input 1 (traffic volume)
const int POT2_PIN = 35; // Simulated Input 2 (object distance)
const int LDR_PIN  = 32; // Simulated LDR (ambient brightness)
const int LED_PIN  = 2;  // Simulated street light (PWM)

const char* ssid = "Wokwi-GUEST";
const char* password = "";

unsigned long myChannelNumber = 3437665;
const char* myWriteAPIKey = "QVXPDIWR83IXXUH8";

WiFiClient client;
unsigned long lastLogTime = 0;
const unsigned long logInterval = 15000; // Send data to ThingSpeak every 15 seconds

// --- TFLite Micro Globals ---
const int kTensorArenaSize = 4 * 1024; // 4 KB arena size for the compact MLP
uint8_t tensor_arena[kTensorArenaSize];

tflite::AllOpsResolver tflite_resolver;
tflite::MicroErrorReporter error_reporter;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// --- Scaling parameters from Python training (Min/Max) ---
const float X_min[3] = {29.0f, 15.0f, 97.0f};       // Traffic, Distance, LDR minimum values
const float X_max[3] = {4013.0f, 3995.0f, 3986.0f}; // Traffic, Distance, LDR maximum values

void setup() {
  Serial.begin(115200);

  // Configure pins
  pinMode(LED_PIN, OUTPUT);
  analogWriteResolution(8);

  // Connect to WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");

  int retries = 0;
  while (WiFi.status() != WL_CONNECTED && retries < 10) {
    delay(500);
    Serial.print(".");
    retries++;
  }
  Serial.println();

  ThingSpeak.begin(client); // Initialize ThingSpeak client

  // --- Initialize TensorFlow Lite ---
  tflite_model = tflite::GetModel(g_model);

  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Error: Model schema version mismatch!");
    return;
  }

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(
      tflite_model, tflite_resolver, tensor_arena, kTensorArenaSize,
      &error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensor memory
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Error: AllocateTensors failed!");
    return;
  }

  // Get input and output tensor pointers
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.println("System started. TinyML model loaded successfully.");
}

int calculate_lighting_level_ml(int pot1, int pot2, int ldr) {
  // 1. Convert raw values to float
  float raw_inputs[3] = {(float)pot1, (float)pot2, (float)ldr};

  // 2. Min-Max normalization to [0,1], identical to Python training script
  for (int i = 0; i < 3; i++) {
    float normalized = (raw_inputs[i] - X_min[i]) /
                       (X_max[i] - X_min[i]);

    // Clamp values if they exceed training limits
    if (normalized < 0.0f) normalized = 0.0f;
    if (normalized > 1.0f) normalized = 1.0f;

    // Write normalized values into the input tensor
    // For INT8 models, quantization may be required.
    // This implementation assumes a Float32 input tensor.
    input_tensor->data.f[i] = normalized;
  }

  // 3. Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Error during model inference (Invoke)!");
    return 0;
  }

  // 4. Read predicted value
  float predicted_pwm = output_tensor->data.f[0];

  // 5. Clamp result to valid PWM range [0,255] and round
  int pwm_output = (int)(round(predicted_pwm));

  if (pwm_output < 0) pwm_output = 0;
  if (pwm_output > 255) pwm_output = 255;

  return pwm_output;
}

void loop() {
  // 1. Read sensor values
  int valPot1 = analogRead(POT1_PIN);
  int valPot2 = analogRead(POT2_PIN);
  int valLdr  = analogRead(LDR_PIN);

  // 2. Machine Learning inference (replaces the previous if/else logic)
  int pwm_value = calculate_lighting_level_ml(valPot1, valPot2, valLdr);

  // 3. Control LED brightness
  analogWrite(LED_PIN, pwm_value);

  // 4. Data logging
  if (millis() - lastLogTime >= logInterval) {
    if (WiFi.status() == WL_CONNECTED) {

      ThingSpeak.setField(1, valPot1);
      ThingSpeak.setField(2, valPot2);
      ThingSpeak.setField(3, valLdr);
      ThingSpeak.setField(4, pwm_value);

      int httpCode =
          ThingSpeak.writeFields(myChannelNumber, myWriteAPIKey);

      if (httpCode == 200) {
        Serial.println("ThingSpeak update successful (ML-based control).");
      } else {
        Serial.println(
            "ThingSpeak update failed. HTTP code: " +
            String(httpCode));
      }

    } else {
      Serial.println(
          "No WiFi connection. Lighting control continues, logging paused.");
    }

    lastLogTime = millis();
  }

  delay(50); // Short delay for simulation stability
}