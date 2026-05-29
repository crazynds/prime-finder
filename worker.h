#pragma once
#include "prime_list.h"
#include "sieve_table.h"

/* Entry point for all worker ranks (rank != 0). */
/* cpu_primes used for CPU fallback Phase 1 (same file as gpu_primes) */
void worker_run(const prime_list_t *gpu_primes,
                const prime_list_t *cpu_p1_primes,
                const sieve_table_t *sieve_table,
                int is_gpu_worker,
                int gpu_index);
