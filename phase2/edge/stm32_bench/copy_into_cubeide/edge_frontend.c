#include <math.h>
#include "edge_frontend.h"
#include "edge_filter.h"

static double z[EDGE_NSECTIONS][2];
static long   seen;
static double csum, csq;          /* calibration sums */
static long   ccount;
static double mu, sigma;
static int    calibrated;
static double fbuf[EDGE_WIN];     /* filtered, uncalibrated, ring buffer */
static float  rbuf[EDGE_WIN];     /* raw, ring buffer                    */
static int    head;               /* next write position                 */
static long   npost;              /* samples since calibration finished  */
static int    since;

void edge_frontend_reset(void)
{
    for (int s = 0; s < EDGE_NSECTIONS; ++s) z[s][0] = z[s][1] = 0.0;
    seen = 0; csum = csq = 0.0; ccount = 0; mu = 0.0; sigma = 1.0;
    calibrated = 0; head = 0; npost = 0; since = 0;
}

/* One sample through the cascade of biquads (direct form II transposed, like scipy sosfilt). */
static double filter_sample(double x)
{
    for (int s = 0; s < EDGE_NSECTIONS; ++s) {
        const double *c = &EDGE_SOS[6 * s];
        double y = c[0] * x + z[s][0];
        z[s][0] = c[1] * x - c[4] * y + z[s][1];
        z[s][1] = c[2] * x - c[5] * y;
        x = y;
    }
    return x;
}

int edge_frontend_state(void) { return seen <= EDGE_WARMUP ? 0 : (calibrated ? 2 : 1); }
double edge_frontend_mu(void) { return mu; }
double edge_frontend_sigma(void) { return sigma; }

int edge_frontend_push(float raw, edge_window_t *out)
{
    double v = filter_sample((double)raw);
    seen++;
    if (seen <= EDGE_WARMUP) return 0;
    if (!calibrated) {
        csum += v; csq += v * v; ccount++;
        if (ccount == EDGE_CALIB) {
            mu = csum / ccount;
            double var = csq / ccount - mu * mu;
            sigma = var > 0 ? sqrt(var) : 1.0;
            calibrated = 1;
        }
        return 0;
    }
    fbuf[head] = v; rbuf[head] = raw;
    head = (head + 1) % EDGE_WIN;
    npost++; since++;
    if (npost < EDGE_WIN || since < EDGE_SHIFT) return 0;
    since = 0;

    double dc = 0, fm = 0, fs2 = 0, wm = 0, ws2 = 0;
    for (int i = 0; i < EDGE_WIN; ++i) {
        int k = (head + i) % EDGE_WIN;            /* oldest first */
        double f = fbuf[k];
        float w = (float)((f - mu) / sigma);
        out->window[i] = w;
        dc += rbuf[k]; fm += f; fs2 += f * f; wm += w; ws2 += (double)w * w;
    }
    dc /= EDGE_WIN; fm /= EDGE_WIN; wm /= EDGE_WIN;
    double fstd = sqrt(fmax(fs2 / EDGE_WIN - fm * fm, 0.0));
    out->dc_level = (float)dc;
    out->perfusion_index = (float)(100.0 * fstd / (fabs(dc) + 1e-9));
    out->signal_amplitude = (float)sqrt(fmax(ws2 / EDGE_WIN - wm * wm, 0.0));
    return 1;
}
