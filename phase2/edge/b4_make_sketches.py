"""B4. Copy the C files next to each benchmark, ready for Ramaa to build and flash.

Arduino and STM32CubeIDE both want the source files in the project folder, so the
same files are copied into:
    phase2/edge/esp32_bench/                     (Arduino sketch, ready to open)
    phase2/edge/stm32_bench/copy_into_cubeide/   (drop into Core/Src and Core/Inc)
and into Ramaa's node firmware folder if it exists in your copy of the repo.

    python -m phase2.edge.b4_make_sketches
"""
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, "c")
files = sorted(f for f in os.listdir(C) if f.endswith((".c", ".h")))
targets = [os.path.join(HERE, "esp32_bench"), os.path.join(HERE, "stm32_bench", "copy_into_cubeide")]
node = os.path.join(HERE, "..", "hardware", "firmware", "sensor_node")
if os.path.isdir(node):
    targets.append(node)
for t in targets:
    os.makedirs(t, exist_ok=True)
    for f in files:
        if os.path.basename(t) == "sensor_node" and f.startswith("edge_bench"):
            continue
        shutil.copy2(os.path.join(C, f), os.path.join(t, f))
    print(f"copied {len(files)} files into {os.path.relpath(t, os.path.join(HERE, '..', '..'))}")
