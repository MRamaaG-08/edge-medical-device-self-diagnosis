/* The compressed Random Forest: 13 features in, 5 class probabilities out. */
#ifndef EDGE_MODEL_H
#define EDGE_MODEL_H
#ifdef __cplusplus
extern "C" {
#endif
/* Returns the winning class index (0 CLEAN, 1 NOISY, 2 DRIFTED, 3 ATTENUATED, 4 SPIKED). */
int edge_model_predict(const float *features, float *proba);
extern const char *EDGE_CLASS_NAMES[5];
#ifdef __cplusplus
}
#endif
#endif
