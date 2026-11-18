/* ESP32 timing benchmark (Bhumika writes it, Ramaa flashes it).
 * Board: "ESP32 Dev Module". Open Serial Monitor at 115200 baud and copy everything it prints.
 * The .c/.h files from phase2/edge/c must sit in THIS folder (b4_make_sketches.py copies them). */
#include <Arduino.h>
#include "edge_bench.h"
#include "edge_config.h"

static uint32_t clock_us(void) { return (uint32_t)micros(); }

void setup() {
  Serial.begin(115200);
  delay(1500);
  edge_bench_result_t r;
  uint32_t heap_before = ESP.getFreeHeap();
  edge_bench_run(clock_us, 20, &r);
  Serial.println("=== ESP32 edge benchmark (microseconds) ===");
  Serial.printf("{\"board\":\"ESP32-WROOM-32\",\"cpu_mhz\":%u,\"reps\":%u,"
                "\"frontend_us_per_sample\":%.2f,\"features_us\":%.1f,\"model_us\":%.1f,"
                "\"total_us_per_window\":%.1f,\"window_budget_us\":2500000,"
                "\"free_heap_bytes\":%u,\"sketch_bytes\":%u,\"last_class\":%d}\n",
                getCpuFrequencyMhz(), r.reps, r.frontend_per_sample, r.features, r.model,
                r.total_per_window, heap_before, ESP.getSketchSize(), r.last_class);
  Serial.printf("busy %.3f%% of the time\n", 100.0 * r.total_per_window / 2500000.0);
}

void loop() { delay(10000); }
