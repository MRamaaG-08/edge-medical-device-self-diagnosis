/* Parity test on your laptop: does the C code give the same numbers as Python?
 * Built and run by:  python -m phase2.edge.b3_parity
 * (which calls: gcc -O2 -I../c parity.c ../c/*.c -lm -o parity  and runs it from the repo root) */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "edge_config.h"
#include "edge_frontend.h"
#include "edge_features.h"
#include "edge_model.h"

#define MAXW 64

static int read_rows(const char *path, double *buf, int cols, int maxrows)
{
    FILE *f = fopen(path, "r");
    if (!f) { printf("cannot open %s (run b2_export_c first)\n", path); exit(1); }
    int i = 0;
    while (i < cols * maxrows && fscanf(f, "%lf%*[,\n\r ]", &buf[i]) == 1) i++;
    fclose(f);
    return i / cols;
}

static double RAW[20000], PW[MAXW * EDGE_WIN], PF[MAXW * EDGE_NFEAT], PP[MAXW * 5];

int main(void)
{
    const char *names[EDGE_NFEAT] = {"mean", "variance", "rms", "kurtosis", "zero_crossing_rate",
        "peak_to_peak", "skewness", "mean_abs_deviation", "dominant_freq", "spectral_entropy",
        "ppg_band_power", "power_ratio", "spectral_centroid"};
    int nraw = read_rows("tests/golden_edge/raw.csv", RAW, 1, 20000);
    int nw = read_rows("tests/golden_edge/windows.csv", PW, EDGE_WIN, MAXW);
    read_rows("tests/golden_edge/features.csv", PF, EDGE_NFEAT, MAXW);
    read_rows("tests/golden_edge/proba.csv", PP, 5, MAXW);

    /* 1. front end: raw samples -> calibrated windows */
    edge_frontend_reset();
    edge_window_t win;
    int got = 0;
    double worst_w = 0;
    for (int i = 0; i < nraw && got < nw; ++i) {
        if (edge_frontend_push((float)RAW[i], &win)) {
            for (int k = 0; k < EDGE_WIN; ++k) {
                double e = fabs(win.window[k] - PW[got * EDGE_WIN + k]);
                if (e > worst_w) worst_w = e;
            }
            got++;
        }
    }
    printf("front end : %d of %d windows made, worst sample difference %.2e  %s\n",
           got, nw, worst_w, (got == nw && worst_w < 1e-3) ? "PASS" : "CHECK");

    /* 2. features, from Python's windows so each stage is tested on its own */
    edge_features_init();
    double worst[EDGE_NFEAT] = {0};
    int agree = 0;
    for (int w = 0; w < nw; ++w) {
        float x[EDGE_WIN], f[EDGE_NFEAT], fpy[EDGE_NFEAT], p[5];
        for (int k = 0; k < EDGE_WIN; ++k) x[k] = (float)PW[w * EDGE_WIN + k];
        edge_features(x, f);
        for (int j = 0; j < EDGE_NFEAT; ++j) {
            double ref = PF[w * EDGE_NFEAT + j];
            double rel = fabs(f[j] - ref) / (fabs(ref) > 1e-3 ? fabs(ref) : 1e-3);
            if (rel > worst[j]) worst[j] = rel;
            fpy[j] = (float)ref;
        }
        /* 3. forest, fed Python's features so only the forest is being tested */
        int c = edge_model_predict(fpy, p);
        int best = 0;
        double pdiff = 0;
        for (int k = 0; k < 5; ++k) {
            if (PP[w * 5 + k] > PP[w * 5 + best]) best = k;
            pdiff = fmax(pdiff, fabs(p[k] - PP[w * 5 + k]));
        }
        if (c == best && pdiff < 1e-4) agree++;
    }
    int ok = (got == nw && worst_w < 1e-3);
    printf("\nfeature              worst relative error\n");
    for (int j = 0; j < EDGE_NFEAT; ++j) {
        printf("%-20s %.2e  %s\n", names[j], worst[j], worst[j] < 1e-3 ? "PASS" : "CHECK");
        if (worst[j] >= 1e-3) ok = 0;
    }
    printf("\nforest: same class and probabilities on %d of %d windows  %s\n", agree, nw,
           agree == nw ? "PASS" : "CHECK");
    printf("%s\n", (ok && agree == nw) ? "PARITY PASSED" : "PARITY NEEDS ATTENTION");
    return (ok && agree == nw) ? 0 : 1;
}
