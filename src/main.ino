#include <WiFi.h>
#include <ThingSpeak.h>
#include <TensorFlowLite_ESP32.h>
#include <math.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"

const int POT1_PIN = 34; // Simulated Input 1 (traffic volume)
const int POT2_PIN = 35; // Simulated Input 2 (object distance)
const int LDR_PIN  = 32; // Simulated LDR (ambient brightness)
const int LED_PIN  = 2;  // Simulated street light (PWM)

#ifdef __has_include
  #if __has_include("secrets.h")
    #include "secrets.h"
  #endif
#endif

const char* ssid = WIFI_SSID;
const char* password = WIFI_PASSWORD;
unsigned long myChannelNumber = THINGSPEAK_CHANNEL;
const char* myWriteAPIKey = THINGSPEAK_WRITE_API_KEY;

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

static void write_quantized_input_value(TfLiteTensor* tensor, int index, float value) {
  if (tensor == nullptr) {
    return;
  }

  if (tensor->type == kTfLiteFloat32) {
    tensor->data.f[index] = value;
    return;
  }

  if (tensor->quantization.type != kTfLiteAffineQuantization ||
      tensor->quantization.params == nullptr) {
    Serial.println("Error: input tensor has no affine quantization parameters.");
    return;
  }

  TfLiteAffineQuantization* affine =
      reinterpret_cast<TfLiteAffineQuantization*>(tensor->quantization.params);
  if (affine->scale == nullptr || affine->scale->size <= 0 ||
      affine->zero_point == nullptr || affine->zero_point->size <= 0) {
    Serial.println("Error: quantization scale/zero-point missing.");
    return;
  }

  const float scale = affine->scale->data[0];
  const int32_t zero_point = affine->zero_point->data[0];
  int32_t quantized = (int32_t)roundf((value / scale) + zero_point);

  if (quantized < -128) quantized = -128;
  if (quantized > 127) quantized = 127;

  if (tensor->type == kTfLiteUInt8) {
    if (quantized < 0) quantized = 0;
    if (quantized > 255) quantized = 255;
    tensor->data.uint8[index] = (uint8_t)quantized;
  } else {
    tensor->data.int8[index] = (int8_t)quantized;
  }
}

static float read_quantized_output_value(const TfLiteTensor* tensor, int index) {
  if (tensor == nullptr) {
    return 0.0f;
  }

  if (tensor->type == kTfLiteFloat32) {
    return tensor->data.f[index];
  }

  if (tensor->quantization.type != kTfLiteAffineQuantization ||
      tensor->quantization.params == nullptr) {
    return 0.0f;
  }

  TfLiteAffineQuantization* affine =
      reinterpret_cast<TfLiteAffineQuantization*>(tensor->quantization.params);
  if (affine->scale == nullptr || affine->scale->size <= 0 ||
      affine->zero_point == nullptr || affine->zero_point->size <= 0) {
    return 0.0f;
  }

  const float scale = affine->scale->data[0];
  const int32_t zero_point = affine->zero_point->data[0];
  int32_t quantized = 0;

  if (tensor->type == kTfLiteUInt8) {
    quantized = tensor->data.uint8[index];
  } else {
    quantized = tensor->data.int8[index];
  }

  return scale * (static_cast<float>(quantized) - static_cast<float>(zero_point));
}

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

    // Write normalized values into the input tensor using the tensor's
    // quantization parameters when necessary.
    write_quantized_input_value(input_tensor, i, normalized);
  }

  // 3. Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Error during model inference (Invoke)!");
    return 0;
  }

  // 4. Read predicted value and dequantize it if the output tensor is quantized.
  float predicted_pwm = read_quantized_output_value(output_tensor, 0);

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