/* The 13 Phase 1 features in C. Order frozen, identical to selfdiag/features.py. */
#ifndef EDGE_FEATURES_H
#define EDGE_FEATURES_H
#include "edge_config.h"
#ifdef __cplusplus
extern "C" {
#endif
void edge_features_init(void);
void edge_features(const float *w, float *out);   /* w: EDGE_WIN samples, out: 13 */
#ifdef __cplusplus
}
#endif
#endif
