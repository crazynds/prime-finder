#pragma once
#include "prime_list.h"
#include "sieve_table.h"  /* KJ_WORDS */
#include <stdint.h>

/*
 * CPU fallback for Phase 1 trial division.
 * Same logic as the CUDA kernel but runs single-threaded on CPU.
 * Used by CPU workers when no Phase 2 tasks are available.
 */
void trial_div_cpu(const prime_list_t *pl,
                   long long a, long long b, long long c,
                   uint32_t *sieve_bits /* KJ_WORDS uint32, in/out */);
