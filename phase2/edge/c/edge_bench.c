#include <math.h>
#include "edge_bench.h"
#include "edge_config.h"
#include "edge_frontend.h"
#include "edge_features.h"
#include "edge_model.h"

/* A made-up but pulse-like test signal: 72 bpm on top of a steady light level. */
static float test_sample(long i)
{
    double t = (double)i / EDGE_FS;
    double ph = fmod(t * 1.2, 1.0);
    double pulse = exp(-pow((ph - 0.22) / 0.055, 2) / 2) + 0.38 * exp(-pow((ph - 0.48) / 0.075, 2) / 2);
    return (float)(50000.0 * (1.0 + 0.01 * pulse));
}

void edge_bench_run(edge_clock_fn now, uint32_t reps, edge_bench_result_t *r)
{
    static edge_window_t win;
    static float feat[EDGE_NFEAT], proba[5];
    long i = 0;
    edge_frontend_reset();
    edge_features_init();
    while (!edge_frontend_push(test_sample(i), &win)) i++;     /* warm-up + calibration */

    uint32_t t0 = now();
    for (uint32_t k = 0; k < reps * EDGE_SHIFT; ++k) edge_frontend_push(test_sample(++i), &win);
    uint32_t t1 = now();
    for (uint32_t k = 0; k < reps; ++k) edge_features(win.window, feat);
    uint32_t t2 = now();
    for (uint32_t k = 0; k < reps; ++k) r->last_class = edge_model_predict(feat, proba);
    uint32_t t3 = now();

    r->reps = reps;
    r->frontend_per_sample = (float)(t1 - t0) / (reps * EDGE_SHIFT);
    r->features = (float)(t2 - t1) / reps;
    r->model = (float)(t3 - t2) / reps;
    r->total_per_window = r->features + r->model + r->frontend_per_sample * EDGE_SHIFT;
}
