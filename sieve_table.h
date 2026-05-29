#pragma once
#include "prime_list.h"
#include <stdint.h>
#include <stddef.h>

/*
 * Periodic sieve table for phase 0.
 *
 * N = 10^a - k*10^b - j*10^c - 1
 *
 * N mod p depends only on (a % L10, b % L10, c % L10) where L10 = ord_p(10).
 * We precompute a 10000-bit bitmask for each (ra, rb, rc) triple.
 */

#define BITS_PER_WORD   32
#define KJ_BITS         10000       /* 100 * 100 */
#define KJ_WORDS        ((KJ_BITS + BITS_PER_WORD - 1) / BITS_PER_WORD)  /* 313 */

/* Skip prime if L10^3 > this (table too large) */
#define MAX_TABLE_ENTRIES  (1 << 18)   /* ~256k entries per prime */

typedef struct {
    uint32_t  p;
    int       L10;
    uint64_t  bitmask_offset;  /* byte offset into sieve_table_t.flat_data */
} sieve_entry_t;

typedef struct {
    sieve_entry_t *entries;   /* array of n entries */
    int            n;
    uint32_t      *flat_data; /* contiguous block holding all bitmasks */
    size_t         flat_size; /* bytes in flat_data */
    int            shared;    /* 1 = flat_data is MPI shared mem, don't free */
} sieve_table_t;

/* Build from scratch. verbose=1 prints progress bar to stderr. */
int  sieve_table_build(sieve_table_t *st, int verbose, const prime_list_t *pl);

/* Reconstruct from a pre-filled flat buffer (used by other workers) */
int  sieve_table_from_flat(sieve_table_t *st, void *flat, size_t flat_size,
                           const prime_list_t *pl);

/* Total bytes needed for flat_data — call before allocating shared memory */
size_t sieve_table_flat_size(const prime_list_t *pl);

void sieve_table_free(sieve_table_t *st);
