/* Day-one hardware check: is the MAX30102 wired right, and does it really give 100 samples/s?
 * Prints one line per sample: millis,ir,red. Open Serial Monitor at 921600 baud to look,
 * then close it and run:  python -m phase2.hardware.hub.check_stream --port COM5
 * Wiring: VIN->3V3, GND->GND, SDA->GPIO21, SCL->GPIO22. Library: SparkFun MAX3010x. */
#include <Wire.h>
#include "MAX30105.h"

MAX30105 sensor;

void setup() {
  Serial.begin(921600);
  if (!sensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not found: check SDA=GPIO21, SCL=GPIO22, 3V3 and GND");
    while (1) delay(1000);
  }
  sensor.setup(0x1F, 1, 2, 100, 411, 16384);   // same settings as the real node
}

void loop() {
  sensor.check();
  while (sensor.available()) {
    Serial.printf("%lu,%lu,%lu\n", (unsigned long)millis(),
                  (unsigned long)sensor.getFIFOIR(), (unsigned long)sensor.getFIFORed());
    sensor.nextSample();
  }
}
