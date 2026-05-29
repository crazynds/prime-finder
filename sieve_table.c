#include "sieve_table.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---------- math helpers ---------- */

static uint64_t powmod64(uint64_t base, uint64_t exp, uint32_t mod)
{
    uint64_t r = 1; base %= mod;
    while (exp > 0) {
        if (exp & 1) r = r * base % mod;
        base = base * base % mod; exp >>= 1;
    }
    return r;
}
static uint64_t modinv(uint64_t a, uint32_t p) { return powmod64(a, p-2, p); }

static int mult_order(uint64_t base, uint32_t p)
{
    base %= p;
    if (base == 0) return 0;
    uint32_t phi = p - 1;
    uint32_t factors[20]; int nf = 0, tmp = phi;
    for (uint32_t f = 2; (uint64_t)f*f <= (uint32_t)tmp; f++)
        if (tmp % f == 0) { factors[nf++] = f; while (tmp % f == 0) tmp /= f; }
    if (tmp > 1) factors[nf++] = tmp;
    uint64_t ord = phi;
    for (int i = 0; i < nf; i++)
        while (ord % factors[i] == 0) {
            uint64_t c = ord / factors[i];
            if (powmod64(base, c, p) == 1) ord = c; else break;
        }
    return (int)ord;
}

/* ---------- rev lookup tables ---------- */

static uint8_t g_rev_val[101];
static uint8_t g_rev_ndig[101];
static int g_rev_tables_built = 0;

static void build_rev_tables_cpu(void)
{
    if (g_rev_tables_built) return;
    for (int x = 1; x <= 100; x++) {
        int nd = (x < 10) ? 1 : (x < 100) ? 2 : 3;
        int r = 0, t = x;
        for (int i = 0; i < nd; i++) { r = r*10 + t%10; t /= 10; }
        g_rev_val[x]  = (uint8_t)r;
        g_rev_ndig[x] = (uint8_t)nd;
    }
    g_rev_tables_built = 1;
}

/* ---------- flat layout helpers ---------- */

/*
 * Flat memory layout:
 *   [int n_entries]
 *   [sieve_entry_t entries[n]]
 *   [uint32_t bitmask data...]   — bitmask_offset in each entry is byte offset
 *                                  from start of flat_data (the uint32_t block)
 *
 * sieve_table_t.flat_data points to the bitmask data block.
 * entries[i].bitmask_offset / sizeof(uint32_t) gives the word offset.
 */

/* Compute total bitmask bytes needed for one prime with given L10 */
static size_t bitmask_bytes(int L10)
{
    return (size_t)L10 * L10 * L10 * KJ_WORDS * sizeof(uint32_t);
}

size_t sieve_table_flat_size(const prime_list_t *pl)
{
    build_rev_tables_cpu();
    size_t total = 0;
    for (uint32_t i = 0; i < pl->count; i++) {
        uint32_t p = pl->primes[i];
        if (p < 3) continue;
        int L10 = mult_order(10, p);
        if (L10 == 0) continue;
        long long entries = (long long)L10 * L10 * L10;
        if (entries > MAX_TABLE_ENTRIES) continue;
        total += bitmask_bytes(L10);
    }
    return total;
}

/* ---------- core fill routine: fill one prime's bitmask into buf ---------- */

static void fill_prime_bitmask(uint32_t *buf, uint32_t p, int L10)
{
    size_t nb = bitmask_bytes(L10);
    memset(buf, 0, nb);

    uint64_t *pow10     = malloc((size_t)L10 * sizeof(uint64_t));
    uint64_t *inv_pow10 = malloc((size_t)L10 * sizeof(uint64_t));
    if (!pow10 || !inv_pow10) { free(pow10); free(inv_pow10); return; }

    pow10[0] = 1;
    for (int r = 1; r < L10; r++) pow10[r] = pow10[r-1] * 10 % p;
    for (int r = 0; r < L10; r++) inv_pow10[r] = (pow10[r] != 0) ? modinv(pow10[r], p) : 0;

    for (int ra = 0; ra < L10; ra++) {
    for (int rb = 0; rb < L10; rb++) {
    for (int rc = 0; rc < L10; rc++) {

        long long cell = ((long long)ra * L10 + rb) * L10 + rc;
        uint32_t *bits = buf + cell * KJ_WORDS;

        /* Normal sieve */
        uint64_t inv_rc = inv_pow10[rc];
        if (pow10[rc] != 0) {
            for (int k = 1; k <= 100; k++) {
                long long num = ((long long)pow10[ra]
                                 - (long long)k * pow10[rb] % p
                                 - 1 + 2*(long long)p) % p;
                uint64_t j0 = (uint64_t)num * inv_rc % p;
                if (j0 == 0) j0 = p;
                for (uint64_t j = j0; j <= 100; j += p) {
                    int bit = (k-1)*100 + ((int)j-1);
                    bits[bit/BITS_PER_WORD] |= 1u << (bit%BITS_PER_WORD);
                }
            }
        }

        /* Rev sieve */
        uint64_t inv_ra = inv_pow10[ra];
        uint64_t inv_rc_rev[4];
        uint64_t rc_rev3 = 0;
        for (int lj = 1; lj <= 3; lj++) {
            int idx = (rc + lj) % L10;
            inv_rc_rev[lj] = inv_ra * pow10[idx] % p;
            if (lj == 3) rc_rev3 = pow10[ra] * inv_pow10[idx] % p;
        }

        for (int k = 1; k <= 100; k++) {
            int lk = g_rev_ndig[k], rk = g_rev_val[k];
            uint64_t rb_rev = pow10[ra] * inv_pow10[(rb + lk) % L10] % p;
            uint64_t base_k = ((long long)pow10[ra]
                                - (long long)rk * rb_rev % p
                                - 1 + 2*(long long)p) % p;

            if (inv_rc_rev[1] != 0) {
                uint64_t j0 = base_k * inv_rc_rev[1] % p;
                if (j0 == 0) j0 = p;
                for (uint64_t j = j0; j <= 9; j += p) {
                    int bit = (k-1)*100 + ((int)j-1);
                    bits[bit/BITS_PER_WORD] |= 1u << (bit%BITS_PER_WORD);
                }
            }
            if (inv_rc_rev[2] != 0) {
                uint64_t trj = base_k * inv_rc_rev[2] % p;
                if (trj == 0) trj = p;
                for (uint64_t rj = trj; rj <= 99; rj += p) {
                    int j = -1;
                    if (rj >= 1 && rj <= 9) j = (int)rj * 10;
                    else if (rj >= 10 && rj <= 99 && rj % 10 != 0)
                        j = (int)(rj%10)*10 + (int)(rj/10);
                    if (j >= 10 && j <= 99) {
                        int bit = (k-1)*100 + (j-1);
                        bits[bit/BITS_PER_WORD] |= 1u << (bit%BITS_PER_WORD);
                    }
                }
            }
            if (rc_rev3 != 0 && rc_rev3 == base_k) {
                int bit = (k-1)*100 + 99;
                bits[bit/BITS_PER_WORD] |= 1u << (bit%BITS_PER_WORD);
            }
        }

    }}} /* ra, rb, rc */

    free(inv_pow10);
    free(pow10);
}

/* ---------- public API ---------- */

int sieve_table_build(sieve_table_t *st, int verbose, const prime_list_t *pl)
{
    build_rev_tables_cpu();
    memset(st, 0, sizeof(*st));

    /* Count eligible primes first */
    int n = 0;
    for (uint32_t i = 0; i < pl->count; i++) {
        uint32_t p = pl->primes[i];
        if (p < 3) continue;
        int L10 = mult_order(10, p);
        if (L10 == 0 || (long long)L10*L10*L10 > MAX_TABLE_ENTRIES) continue;
        n++;
    }

    st->entries = malloc((size_t)n * sizeof(sieve_entry_t));
    if (!st->entries) return -1;

    /* Allocate flat bitmask block */
    st->flat_size = sieve_table_flat_size(pl);
    st->flat_data = (st->flat_size > 0) ? malloc(st->flat_size) : NULL;
    if (st->flat_size > 0 && !st->flat_data) { free(st->entries); return -1; }
    st->shared = 0;

    size_t offset_bytes = 0;
    int ei = 0;
    int last_10pct = -1;
    for (uint32_t i = 0; i < pl->count; i++) {
        /* Print progress every 10% (only if verbose) */
        if (verbose) {
            int pct10 = (n > 0) ? (ei * 10 / n) : 10;
            if (pct10 != last_10pct) {
                char bar[31];
                int filled = pct10 * 3;  /* 30 chars total, 3 per 10% */
                memset(bar, '#', filled);
                memset(bar + filled, ' ', 30 - filled);
                bar[30] = '\0';
                fprintf(stderr, "  sieve [%s] %3d%%\n", bar, pct10 * 10);
                fflush(stderr);
                last_10pct = pct10;
            }
        }

        uint32_t p = pl->primes[i];
        if (p < 3) continue;
        int L10 = mult_order(10, p);
        if (L10 == 0 || (long long)L10*L10*L10 > MAX_TABLE_ENTRIES) continue;

        uint32_t *buf = (uint32_t *)((char *)st->flat_data + offset_bytes);
        fill_prime_bitmask(buf, p, L10);

        st->entries[ei].p              = p;
        st->entries[ei].L10            = L10;
        st->entries[ei].bitmask_offset = offset_bytes;
        ei++;
        offset_bytes += bitmask_bytes(L10);
    }
    st->n = ei;
    if (verbose)
        fprintf(stderr, "  sieve [##############################] 100%%\n");
    return 0;
}

/*
 * Reconstruct sieve_table_t from a pre-filled flat buffer.
 * The flat buffer must have been produced by sieve_table_build() on the same pl.
 * Does NOT free flat — caller owns it (or it's MPI shared memory).
 */
int sieve_table_from_flat(sieve_table_t *st, void *flat, size_t flat_size,
                           const prime_list_t *pl)
{
    build_rev_tables_cpu();
    memset(st, 0, sizeof(*st));

    int n = 0;
    for (uint32_t i = 0; i < pl->count; i++) {
        uint32_t p = pl->primes[i];
        if (p < 3) continue;
        int L10 = mult_order(10, p);
        if (L10 == 0 || (long long)L10*L10*L10 > MAX_TABLE_ENTRIES) continue;
        n++;
    }

    st->entries = malloc((size_t)n * sizeof(sieve_entry_t));
    if (!st->entries) return -1;

    st->flat_data = (uint32_t *)flat;
    st->flat_size = flat_size;
    st->shared    = 1;  /* don't free flat_data */

    size_t offset_bytes = 0;
    int ei = 0;
    for (uint32_t i = 0; i < pl->count; i++) {
        uint32_t p = pl->primes[i];
        if (p < 3) continue;
        int L10 = mult_order(10, p);
        if (L10 == 0 || (long long)L10*L10*L10 > MAX_TABLE_ENTRIES) continue;

        st->entries[ei].p              = p;
        st->entries[ei].L10            = L10;
        st->entries[ei].bitmask_offset = offset_bytes;
        ei++;
        offset_bytes += bitmask_bytes(L10);
    }
    st->n = ei;
    (void)flat_size;
    return 0;
}

void sieve_table_free(sieve_table_t *st)
{
    free(st->entries);
    if (!st->shared) free(st->flat_data);
    memset(st, 0, sizeof(*st));
}
