#pragma once
#include <stdint.h>
#include "sieve_table.h"  /* for KJ_WORDS */

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
    uint32_t sieve_bits[KJ_WORDS];  /* pre-applied Phase 0 bitmask from master */
} gpu_task_t;

typedef struct {
    long long a, b, c;
    int n_survivors;
    uint8_t ks[MAX_SURVIVORS];  /* k values 1-100 */
    uint8_t js[MAX_SURVIVORS];  /* j values 1-100 */
} gpu_result_t;

/* One single (a,b,c,k,j) pair for Phase 2 Miller-Rabin */
typedef struct {
    long long a, b, c;
    uint8_t k, j;
} cpu_task_t;

typedef struct {
    long long a, b, c;
    uint8_t k, j;
    /*
     * result:
     *   0 = N composite
     *   1 = N probable prime, rev(N) composite
     *   2 = N AND rev(N) both probable prime  ← target
     *   3 = N definitely prime (small N)
     */
    uint8_t result;
} cpu_result_t;
