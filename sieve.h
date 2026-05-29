#pragma once
#include "sieve_table.h"
#include <stdint.h>

/*
 * Phase 0 runtime: apply the precomputed sieve table to (a,b,c).
 *
 * Returns a 10000-bit bitmask (KJ_WORDS uint32_t) where bit (k-1)*100+(j-1)
 * set means (k,j) is provably composite for some small prime.
 *
 * out_bits must point to KJ_WORDS uint32_t (zeroed by caller if accumulating).
 */
void sieve_apply(const sieve_table_t *st,
                 long long a, long long b, long long c,
                 uint32_t *out_bits);

/* Returns 1 if ALL 10000 pairs are eliminated (no need to call GPU) */
int sieve_all_eliminated(const uint32_t *bits);
