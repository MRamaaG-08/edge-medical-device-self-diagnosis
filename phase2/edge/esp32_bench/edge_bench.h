/* Portable timing benchmark. The same code runs on the laptop, the ESP32 and the STM32;
 * only the clock function you pass in is different. */
#ifndef EDGE_BENCH_H
#define EDGE_BENCH_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef uint32_t (*edge_clock_fn)(void);   /* returns a counter: microseconds or CPU cycles */

typedef struct {
    uint32_t reps;
    float frontend_per_sample;   /* counter ticks per raw sample (filter + calibration) */
    float features;              /* ticks per window: the 13 features                   */
    float model;                 /* ticks per window: the forest                         */
    float total_per_window;      /* features + model + 250 front-end samples             */
    int   last_class;            /* sanity check: class of the test window               */
} edge_bench_result_t;

void edge_bench_run(edge_clock_fn now, uint32_t reps, edge_bench_result_t *r);
#ifdef __cplusplus
}
#endif
#endif
