#pragma once
#include "prime_list.h"
#include "sieve.h"   /* KJ_WORDS */
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double upload;    /* H→D memcpy (primes + bits)  */
    double kernel;    /* kernel launch + sync          */
    double download;  /* D→H memcpy (bits)            */
} gpu_timing_t;

/*
 * Phase 1: GPU trial division.
 *
 * Runs the CUDA kernel over all primes in pl. For each prime p, eliminates
 * (k,j) pairs where p | N(a,b,c,k,j). Results are OR'd into sieve_bits
 * (which may already have phase-0 bits set).
 *
 * If timing != NULL, individual step durations are written there.
 * Returns 0 on success.
 */
int trial_div_gpu(const prime_list_t *pl,
                  long long a, long long b, long long c,
                  uint32_t *sieve_bits /* KJ_WORDS uint32_t, in/out */,
                  gpu_timing_t *timing /* optional, may be NULL */);

int cuda_get_device_count(void);
int cuda_set_device(int index);  /* returns 0 on success */

#ifdef __cplusplus
}
#endif
