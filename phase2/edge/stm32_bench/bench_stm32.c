/* STM32F407 Discovery timing benchmark (Bhumika writes it, Ramaa runs it in STM32CubeIDE).
 * Times in CPU cycles using the Cortex-M4 cycle counter (DWT). Results are read with the
 * debugger: Run > Debug, then watch the variable "bench" in the Live Expressions tab.
 * No wires, no serial port, nothing that can damage the board. */
#include "stm32f4xx.h"
#include "edge_bench.h"


volatile edge_bench_result_t bench;
volatile uint32_t bench_cpu_hz;
volatile uint32_t bench_done = 0;

static uint32_t clock_cycles(void) { return DWT->CYCCNT; }

void bench_stm32_run(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;     /* switch the cycle counter on */
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    bench_cpu_hz = SystemCoreClock;
    edge_bench_result_t r;
    edge_bench_run(clock_cycles, 20, &r);
    bench = r;
    bench_done = 1;
}
