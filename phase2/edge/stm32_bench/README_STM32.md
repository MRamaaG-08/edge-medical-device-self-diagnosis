# STM32F407 Discovery benchmark (Ramaa runs this, about 30 minutes)

Purpose: run the exact same C model as the ESP32 on a second, different microcontroller
(ARM Cortex-M4 at 168 MHz) and measure its cost. Nothing is wired to the board, so it
cannot be damaged. Never change option bytes or read-out protection.

1. STM32CubeIDE > File > New > STM32 Project > Board Selector > STM32F407G-DISC1 > Next.
   Name `edge_bench`, C language > Finish > "Initialize all peripherals with their default mode?" Yes.
2. Copy every file in `copy_into_cubeide/` that ends in `.h` into `Core/Inc`,
   and every `.c` file into `Core/Src`. Also copy `bench_stm32.c` into `Core/Src`.
3. Open `Core/Src/main.c`. Between `/* USER CODE BEGIN 2 */` and `/* USER CODE END 2 */` add:
       extern void bench_stm32_run(void);
       bench_stm32_run();
4. Project > Properties > C/C++ Build > Settings > MCU GCC Linker > Libraries: add `m`.
   Optimisation: MCU GCC Compiler > Optimization > `-O2`.
5. Build (hammer). The console prints `text`, `data`, `bss`: flash = text + data, RAM = data + bss.
6. Run > Debug. Press Resume. After a second, Suspend. In the Live Expressions tab add
   `bench`, `bench_cpu_hz` and `bench_done` (must be 1).
7. Type the numbers into `results/phase2/stm32_bench.json`:
       {"cpu_hz": 168000000, "frontend_per_sample": ..., "features": ..., "model": ...,
        "total_per_window": ..., "flash_bytes": ..., "ram_bytes": ...}
   (the bench values are CPU cycles; B5 converts them to microseconds)
