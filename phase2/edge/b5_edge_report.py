"""B5. Turn the three benchmark printouts into one comparison table for the paper.

Inputs (all go in results/phase2/):
    host_bench.txt      made by this script on your laptop (it compiles and runs the bench)
    esp32_bench.txt     what Ramaa copies from the ESP32 Serial Monitor
    stm32_bench.json    what Ramaa reads from the STM32 debugger, typed in as
                        {"cpu_hz": 168000000, "frontend_per_sample": ..., "features": ...,
                         "model": ..., "total_per_window": ..., "flash_bytes": ..., "ram_bytes": ...}

    python -m phase2.edge.b5_edge_report

Writes results/phase2/b5_edge_report.json and prints a table.
"""
import json
import os
import re
import shutil
import subprocess

import config
from phase2.common import ensure_dir

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = ensure_dir(os.path.join(config.RESULTS, "phase2"))

HOST_MAIN = r'''
#include <stdio.h>
#include <stdint.h>
#include <windows.h>
#include "edge_bench.h"

static uint32_t clock_us(void)
{
    static LARGE_INTEGER frequency;
    LARGE_INTEGER counter;

    if (frequency.QuadPart == 0)
        QueryPerformanceFrequency(&frequency);

    QueryPerformanceCounter(&counter);

    return (uint32_t)(
        (counter.QuadPart * 1000000ULL) / frequency.QuadPart
    );
}

int main(void)
{
    edge_bench_result_t r;
    edge_bench_run(clock_us, 200, &r);

    printf("{\"board\":\"laptop\",\"frontend_us_per_sample\":%.3f,"
           "\"features_us\":%.2f,\"model_us\":%.2f,"
           "\"total_us_per_window\":%.2f}\n",
           r.frontend_per_sample, r.features,
           r.model, r.total_per_window);

    return 0;
}
'''

rows = {}
gcc = shutil.which("gcc")
if gcc:
    src = os.path.join(HERE, "host_test", "bench_host.c")
    open(src, "w").write(HOST_MAIN)
    exe = os.path.join(HERE, "host_test", "bench_host.exe" if os.name == "nt" else "bench_host")
    c = [os.path.join(HERE, "c", f) for f in ("edge_bench.c", "edge_frontend.c", "edge_features.c", "edge_model.c")]
    subprocess.run([gcc, "-O2", "-I" + os.path.join(HERE, "c"), src, *c, "-lm", "-o", exe], check=True)
    txt = subprocess.run([exe], capture_output=True, text=True).stdout
    open(os.path.join(OUT, "host_bench.txt"), "w").write(txt)
for name in ("host_bench.txt", "esp32_bench.txt"):
    p = os.path.join(OUT, name)
    if os.path.exists(p):
        m = re.search(r"\{.*\}", open(p).read())
        if m:
            rows[name.split("_")[0]] = json.loads(m.group(0))
p = os.path.join(OUT, "stm32_bench.json")
if os.path.exists(p):
    s = json.load(open(p))
    hz = s.get("cpu_hz", 168_000_000) / 1e6       # cycles -> microseconds
    rows["stm32"] = {"board": "STM32F407 Discovery", "cpu_mhz": hz,
                     "frontend_us_per_sample": s["frontend_per_sample"] / hz,
                     "features_us": s["features"] / hz, "model_us": s["model"] / hz,
                     "total_us_per_window": s["total_per_window"] / hz,
                     "flash_bytes": s.get("flash_bytes"), "ram_bytes": s.get("ram_bytes")}
print(f"{'board':<10}{'front end/sample':>18}{'features':>12}{'model':>10}{'per window':>13}{'busy %':>9}")
for k, r in rows.items():
    print(f"{k:<10}{r['frontend_us_per_sample']:>15.2f} us{r['features_us']:>9.0f} us{r['model_us']:>7.1f} us"
          f"{r['total_us_per_window']:>10.0f} us{100 * r['total_us_per_window'] / 2.5e6:>8.3f}")
json.dump(rows, open(os.path.join(OUT, "b5_edge_report.json"), "w"), indent=2)
print("saved results/phase2/b5_edge_report.json")
if "esp32" not in rows or "stm32" not in rows:
    print("still waiting for:", ", ".join(x for x in ("esp32", "stm32") if x not in rows), "(from Ramaa)")
