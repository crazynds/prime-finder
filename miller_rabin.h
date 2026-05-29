#pragma once

/*
 * Phase 2: Miller-Rabin primality test using GMP.
 *
 * N = 10^a - k*10^b - j*2^c - 1
 *
 * Tests both N and rev(N) (decimal digit reversal).
 *
 * Returns:
 *   0 = N composite (rev not tested)
 *   1 = N probable prime, rev(N) composite
 *   2 = N probable prime, rev(N) probable prime  ← the one we want
 *   3 = N definitely prime (small N), rev not tested
 */
int miller_rabin_test(long long a, long long b, long long c, int k, int j);
