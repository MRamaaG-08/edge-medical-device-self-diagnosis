/* Shared sizes. Must match phase2/contracts.py (Spec A of the Phase 1 paper). */
#ifndef EDGE_CONFIG_H
#define EDGE_CONFIG_H
#define EDGE_FS        100      /* samples per second (MAX30102 native rate)   */
#define EDGE_WIN       500      /* 5 s window                                    */
#define EDGE_SHIFT     250      /* a new window every 2.5 s                      */
#define EDGE_NBINS     251      /* EDGE_WIN/2 + 1 frequency bins                 */
#define EDGE_NFEAT     13       /* frozen order, see selfdiag/features.py        */
#define EDGE_WARMUP    500      /* 5 s of filter settling, thrown away           */
#define EDGE_CALIB     12000    /* 120 s clean baseline for the fixed scale      */
#endif
