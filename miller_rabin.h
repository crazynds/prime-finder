#pragma once

typedef struct {
    double build_n;    /* build_N construction          */
    double mr_n;       /* mpz_probab_prime_p for N      */
    double build_revn; /* build_revN (0 if N composite) */
    double mr_revn;    /* mpz_probab_prime_p for rev(N) */
} mr_timing_t;

/*
 * Phase 2: Miller-Rabin primality test using GMP.
 *
 * N = 10^a - k*10^b - j*2^c - 1
 *
 * Tests both N and rev(N) (decimal digit reversal).
 *
 * If timing != NULL, individual step durations are written there.
 * Returns:
 *   0 = N composite (rev not tested)
 *   1 = N probable prime, rev(N) composite
 *   2 = N probable prime, rev(N) probable prime  ← the one we want
 *   3 = N definitely prime (small N), rev not tested
 */
int miller_rabin_test(long long a, long long b, long long c, int k, int j,
                      mr_timing_t *timing /* optional, may be NULL */);
