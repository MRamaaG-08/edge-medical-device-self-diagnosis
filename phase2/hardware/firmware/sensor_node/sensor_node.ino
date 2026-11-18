/* Sensor node: ESP32-WROOM-32 + MAX30102.
 *
 * Every sample (100 per second) goes through the same front end as phase2/frontend.py.
 * After the 2-minute calibration, every 2.5 s the node:
 *   1. takes the last 5 s window,
 *   2. works out the 13 Phase 1 features and runs the compressed forest ON THE CHIP,
 *   3. sends one signed line to the Pi: verdict + 5 probabilities + telemetry + the window.
 *
 * Wiring (MAX30102 breakout -> ESP32): VIN->3V3, GND->GND, SDA->GPIO21, SCL->GPIO22.
 * Library: "SparkFun MAX3010x Pulse and Proximity Sensor Library" (Arduino Library Manager).
 * Files: copy edge_config.h, edge_filter.h, edge_rf.h, edge_frontend.c/.h, edge_features.c/.h,
 *        edge_model.c/.h from phase2/edge/c into THIS folder (python -m phase2.edge.b4_make_sketches).
 * Commands from the Pi (one per line): L<0-255> sets the IR LED drive (used to make a
 * controlled, physical "weak sensor" fault), R restarts calibration.
 * Key: python -m phase2.hardware.hub.make_key writes node_key.h next to this file. It is in .gitignore:
 *      the real key never reaches GitHub.
 */
#include <Wire.h>
#include "MAX30105.h"
#include "mbedtls/md.h"
extern "C" {
#include "edge_config.h"
#include "edge_frontend.h"
#include "edge_features.h"
#include "edge_model.h"
}

#include "node_key.h"            // defines NODE_KEY; made by make_key.py, never committed
const uint8_t LED_DEFAULT = 0x1F;           // 0x1F = about 6.4 mA

MAX30105 sensor;
edge_window_t win;
float feat[EDGE_NFEAT], proba[5];
uint32_t seq = 0;
uint8_t ledNow = LED_DEFAULT;
static char line[9000];

void hmacHex(const char *msg, char *hexOut) {
  uint8_t mac[32];
  mbedtls_md_context_t ctx;
  mbedtls_md_init(&ctx);
  mbedtls_md_setup(&ctx, mbedtls_md_info_from_type(MBEDTLS_MD_SHA256), 1);
  mbedtls_md_hmac_starts(&ctx, (const unsigned char *)NODE_KEY, strlen(NODE_KEY));
  mbedtls_md_hmac_update(&ctx, (const unsigned char *)msg, strlen(msg));
  mbedtls_md_hmac_finish(&ctx, mac);
  mbedtls_md_free(&ctx);
  for (int i = 0; i < 32; ++i) sprintf(hexOut + 2 * i, "%02x", mac[i]);
  hexOut[64] = 0;
}

void sendSigned() {
  char tag[65];
  hmacHex(line, tag);
  Serial.print(line); Serial.print(' '); Serial.println(tag);
  seq++;
}

void sendStatus() {             // once a second while warming up / calibrating
  snprintf(line, sizeof(line), "{\"seq\":%lu,\"t_ms\":%lu,\"state\":%d,\"led\":%u}",
           (unsigned long)seq, (unsigned long)millis(), edge_frontend_state(), ledNow);
  sendSigned();
}

void sendWindow(int cls, uint32_t procUs) {
  float temp = sensor.readTemperature();
  int n = snprintf(line, sizeof(line),
      "{\"seq\":%lu,\"t_ms\":%lu,\"state\":2,\"cls\":%d,\"p\":[%.4f,%.4f,%.4f,%.4f,%.4f],"
      "\"tele\":{\"dc_level\":%.1f,\"perfusion_index\":%.4f,\"signal_amplitude\":%.4f,"
      "\"die_temp_c\":%.2f,\"led_current\":%u},\"proc_us\":%lu,\"w\":[",
      (unsigned long)seq, (unsigned long)millis(), cls,
      proba[0], proba[1], proba[2], proba[3], proba[4],
      win.dc_level, win.perfusion_index, win.signal_amplitude, temp, ledNow, (unsigned long)procUs);
  for (int i = 0; i < EDGE_WIN; ++i)
    n += snprintf(line + n, sizeof(line) - n, i ? ",%.3f" : "%.3f", win.window[i]);
  snprintf(line + n, sizeof(line) - n, "]}");
  sendSigned();
}

void handleCommands() {
  static char cmd[16]; static int k = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      cmd[k] = 0;
      if (cmd[0] == 'L' && k > 1) { ledNow = (uint8_t)constrain(atoi(cmd + 1), 0, 255); sensor.setPulseAmplitudeIR(ledNow); }
      if (cmd[0] == 'R') edge_frontend_reset();
      k = 0;
    } else if (k < 15) cmd[k++] = c;
  }
}

void setup() {
  Serial.begin(921600);
  if (!sensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not found: check SDA=GPIO21, SCL=GPIO22, 3V3 and GND");
    while (1) delay(1000);
  }
  // LED drive 0x1F, no averaging, red+IR mode, 100 samples/s, 411 us pulse, 16384 nA range
  sensor.setup(LED_DEFAULT, 1, 2, 100, 411, 16384);
  sensor.enableDIETEMPRDY();
  edge_frontend_reset();
  edge_features_init();
}

void loop() {
  static uint32_t lastStatus = 0;
  handleCommands();
  sensor.check();
  while (sensor.available()) {
    float ir = (float)sensor.getFIFOIR();
    sensor.nextSample();
    if (edge_frontend_push(ir, &win)) {
      uint32_t t0 = micros();
      edge_features(win.window, feat);
      int cls = edge_model_predict(feat, proba);
      sendWindow(cls, micros() - t0);
    }
  }
  if (edge_frontend_state() < 2 && millis() - lastStatus > 1000) { lastStatus = millis(); sendStatus(); }
}
