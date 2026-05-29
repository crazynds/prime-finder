#pragma once
#include <stdint.h>

/* MPI tags */
#define TAG_REGISTER   1
#define TAG_GPU_TASK   2
#define TAG_GPU_RESULT 3
#define TAG_CPU_TASK   4
#define TAG_CPU_RESULT 5
#define TAG_TERMINATE  6

/* Worker types */
#define WORKER_GPU 0
#define WORKER_CPU 1

/* Max survivors per (a,b,c) — 100x100 pairs */
#define MAX_SURVIVORS 10000

typedef struct {
    int  type;       /* WORKER_GPU or WORKER_CPU */
    int  gpu_index;  /* device index, -1 if CPU */
    char hostname[64];
} msg_register_t;

typedef struct {
    long long a, b, c;
} gpu_task_t;

typedef struct {
    long long a, b, c;
    int n_survivors;
    uint8_t ks[MAX_SURVIVORS];  /* k values 1-100 */
    uint8_t js[MAX_SURVIVORS];  /* j values 1-100 */
} gpu_result_t;

/* One batch: all survivors for a given (a,b,c) sent to one CPU worker */
typedef struct {
    long long a, b, c;
    int n_pairs;
    uint8_t ks[MAX_SURVIVORS];
    uint8_t js[MAX_SURVIVORS];
} cpu_task_t;

typedef struct {
    long long a, b, c;
    int n_pairs;
    int n_probable_prime;     /* N probable prime (result >= 1) */
    int n_both_prime;         /* N AND rev(N) probable prime (result == 2) */
    uint8_t ks[MAX_SURVIVORS];
    uint8_t js[MAX_SURVIVORS];
    /*
     * results[i]:
     *   0 = N composite
     *   1 = N probable prime, rev(N) composite
     *   2 = N AND rev(N) both probable prime  ← target
     *   3 = N definitely prime (small N)
     */
    uint8_t results[MAX_SURVIVORS];
} cpu_result_t;
