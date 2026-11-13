#include "edge_model.h"
#include "edge_rf.h"

const char *EDGE_CLASS_NAMES[5] = {"CLEAN", "NOISY", "DRIFTED", "ATTENUATED", "SPIKED"};

int edge_model_predict(const float *f, float *proba)
{
    for (int c = 0; c < EDGE_RF_NCLASS; ++c) proba[c] = 0.0f;
    for (int t = 0; t < EDGE_RF_NTREES; ++t) {
        int node = EDGE_RF_ROOT[t];
        while (EDGE_RF_FEAT[node] != 255)            /* scikit-learn: go LEFT when value <= threshold */
            node = (f[EDGE_RF_FEAT[node]] <= EDGE_RF_THR[node]) ? EDGE_RF_LEFT[node] : EDGE_RF_RIGHT[node];
        const float *v = &EDGE_RF_LEAFV[EDGE_RF_LEAF[node] * EDGE_RF_NCLASS];
        for (int c = 0; c < EDGE_RF_NCLASS; ++c) proba[c] += v[c];
    }
    int best = 0;
    for (int c = 0; c < EDGE_RF_NCLASS; ++c) {
        proba[c] /= (float)EDGE_RF_NTREES;
        if (proba[c] > proba[best]) best = c;
    }
    return best;
}
