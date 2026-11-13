#include <math.h>
#include "edge_features.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double HANN[EDGE_WIN], COS_T[EDGE_WIN], SIN_T[EDGE_WIN], WIN_POWER;
static int READY = 0;

void edge_features_init(void)
{
    WIN_POWER = 0;
    for (int i = 0; i < EDGE_WIN; ++i) {
        HANN[i] = 0.5 - 0.5 * cos(2.0 * M_PI * i / (EDGE_WIN - 1));   /* numpy np.hanning */
        COS_T[i] = cos(2.0 * M_PI * i / EDGE_WIN);
        SIN_T[i] = sin(2.0 * M_PI * i / EDGE_WIN);
        WIN_POWER += HANN[i] * HANN[i];
    }
    READY = 1;
}

void edge_features(const float *w, float *out)
{
    static double psd[EDGE_NBINS];
    const int n = EDGE_WIN;
    if (!READY) edge_features_init();

    /* ---- time domain (8) */
    double mean = 0, sumsq = 0, vmin = w[0], vmax = w[0];
    for (int i = 0; i < n; ++i) {
        double x = w[i];
        mean += x; sumsq += x * x;
        if (x < vmin) vmin = x;
        if (x > vmax) vmax = x;
    }
    mean /= n;
    double m2 = 0, m3 = 0, m4 = 0, mad = 0;
    int zc = 0, prev = 0;
    for (int i = 0; i < n; ++i) {
        double d = (double)w[i] - mean;
        m2 += d * d; m3 += d * d * d; m4 += d * d * d * d; mad += fabs(d);
        int neg = signbit(d) ? 1 : 0;
        if (i > 0 && neg != prev) zc++;
        prev = neg;
    }
    m2 /= n; m3 /= n; m4 /= n; mad /= n;
    double kurt = 0, skw = 0;
    if (m2 > 0) {   /* scipy kurtosis(fisher=True, bias=False) and skew(bias=False) */
        kurt = 1.0 / (n - 2) / (n - 3) * ((n * (double)n - 1.0) * m4 / (m2 * m2) - 3.0 * (n - 1.0) * (n - 1.0));
        skw = sqrt((n - 1.0) * n) / (n - 2.0) * (m3 / pow(m2, 1.5));
    }
    out[0] = (float)mean;
    out[1] = (float)m2;                         /* variance (population) */
    out[2] = (float)sqrt(sumsq / n);            /* rms */
    out[3] = (float)kurt;
    out[4] = (float)((double)zc / (n - 1));     /* zero-crossing rate */
    out[5] = (float)(vmax - vmin);              /* peak to peak */
    out[6] = (float)skw;
    out[7] = (float)mad;                        /* mean absolute deviation */

    /* ---- frequency domain (5): Hann window, one-sided DFT, same scaling as numpy */
    const double fscale = 1.0 / (n * (1.0 / EDGE_FS));      /* numpy rfftfreq step */
    double total = 0;
    for (int k = 0; k < EDGE_NBINS; ++k) {
        double re = 0, im = 0;
        int idx = 0;
        for (int t = 0; t < n; ++t) {
            double v = w[t] * HANN[t];
            re += v * COS_T[idx];
            im -= v * SIN_T[idx];
            idx += k; if (idx >= n) idx -= n;
        }
        psd[k] = (re * re + im * im) / (WIN_POWER * EDGE_FS);
        total += psd[k];
    }
    total += 1e-12;
    double best = -1, band = 0, broad = 0, cent = 0, ent = 0;
    int besti = 0;
    for (int k = 0; k < EDGE_NBINS; ++k) {
        double fr = k * fscale;
        double p = psd[k] / total;
        if (p < 1e-12) p = 1e-12;
        ent -= p * log2(p);
        cent += fr * psd[k];
        if (fr >= 0.5 && fr <= 4.0) band += psd[k];
        if (fr >= 0.0 && fr <= 8.0) broad += psd[k];
        if (psd[k] > best) { best = psd[k]; besti = k; }
    }
    out[8]  = (float)(besti * fscale);                         /* dominant frequency */
    out[9]  = (float)(ent / log2((double)EDGE_NBINS));         /* spectral entropy   */
    out[10] = (float)band;                                     /* ppg band power     */
    out[11] = (float)(band / (broad + 1e-12));                 /* power ratio        */
    out[12] = (float)(cent / total);                           /* spectral centroid  */
}
