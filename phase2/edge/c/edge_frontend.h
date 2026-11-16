/* Streaming front end: C twin of phase2/frontend.py (StreamingFrontEnd). */
#ifndef EDGE_FRONTEND_H
#define EDGE_FRONTEND_H
#include "edge_config.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float window[EDGE_WIN];      /* calibrated samples, oldest first           */
    float dc_level;              /* average raw light level over the window    */
    float perfusion_index;       /* 100 x std(filtered) / dc                   */
    float signal_amplitude;      /* std of the calibrated window (about 1)     */
} edge_window_t;

void edge_frontend_reset(void);
/* Push one raw sample. Returns 1 when a new window is ready in *out. */
int  edge_frontend_push(float raw, edge_window_t *out);
/* 0 = warming up, 1 = calibrating, 2 = running */
int  edge_frontend_state(void);
double edge_frontend_mu(void);
double edge_frontend_sigma(void);

#ifdef __cplusplus
}
#endif
#endif
