/*
 * gen_primes <limit> <output.bin>
 *
 * Segmented sieve of Eratosthenes. Output format:
 *
 *   [uint32_t count]          — number of primes
 *   [uint32_t first_prime]    — p[0] stored as absolute value
 *   [uint8_t  gap[1..n-1]]    — gap[i] = p[i] - p[i-1], fits in uint8_t
 *                               (max prime gap below 10^8 is 220)
 *
 * ~4x smaller than storing every prime as uint32_t.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define SEG_BITS  (1 << 19)   /* bits per segment (covers 2*SEG_BITS odd numbers) */

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <limit> <output.bin>\n", argv[0]);
        return 1;
    }

    uint64_t limit = (uint64_t)atoll(argv[1]);
    if (limit < 2) { fprintf(stderr, "limit must be >= 2\n"); return 1; }

    uint64_t sqrtlim = (uint64_t)sqrt((double)limit) + 1;

    /* --- Simple sieve up to sqrt(limit) --- */
    uint8_t *is_composite = calloc(sqrtlim + 1, 1);
    if (!is_composite) { perror("calloc"); return 1; }
    for (uint64_t i = 2; i * i <= sqrtlim; i++)
        if (!is_composite[i])
            for (uint64_t j = i*i; j <= sqrtlim; j += i)
                is_composite[j] = 1;

    uint32_t n_small = 0;
    for (uint64_t i = 2; i <= sqrtlim; i++)
        if (!is_composite[i]) n_small++;

    uint32_t *small_p = malloc(n_small * sizeof(uint32_t));
    if (!small_p) { perror("malloc"); return 1; }
    uint32_t si = 0;
    for (uint64_t i = 2; i <= sqrtlim; i++)
        if (!is_composite[i]) small_p[si++] = (uint32_t)i;
    free(is_composite);

    /* --- Output file --- */
    FILE *fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 1; }

    /* Reserve space for count (filled at end) */
    uint32_t total = 0;
    fwrite(&total, sizeof(uint32_t), 1, fout);

    /* Output buffer */
    uint8_t  *outbuf = malloc(SEG_BITS * 2 + 4);
    uint8_t  *seg    = malloc(SEG_BITS / 8 + 1);
    if (!outbuf || !seg) { perror("malloc"); return 1; }

    uint32_t prev_prime = 0;

    /* --- Segmented sieve: cover [low, high] in steps of SEG_BITS*2 odd numbers --- */
    for (uint64_t low = 3; low <= limit; low += SEG_BITS * 2) {
        uint64_t high = low + SEG_BITS * 2 - 1;
        if (high > limit) high = limit;

        uint64_t seg_size = (high - low) / 2 + 1;  /* number of odd numbers in [low,high] */
        memset(seg, 0, (seg_size + 7) / 8);

        /* Mark composites: for each small prime p, find first odd multiple in [low,high] */
        for (uint32_t i = 0; i < n_small; i++) {
            uint64_t p = small_p[i];
            if (p == 2) continue;
            /* First odd multiple of p >= low */
            uint64_t start = ((low + p - 1) / p) * p;
            if (start == p) start += p;           /* skip p itself */
            if (start % 2 == 0) start += p;       /* must be odd */
            if (start > high) continue;

            for (uint64_t j = start; j <= high; j += 2*p) {
                uint64_t idx = (j - low) / 2;
                seg[idx / 8] |= 1u << (idx % 8);
            }
        }

        /* Emit primes from this segment */
        uint32_t buf_len = 0;
        for (uint64_t idx = 0; idx < seg_size; idx++) {
            if (seg[idx / 8] & (1u << (idx % 8))) continue;
            uint32_t prime = (uint32_t)(low + idx * 2);
            if (prime > limit) break;

            if (total == 0) {
                /* First prime: store absolute as uint32_t */
                memcpy(outbuf + buf_len, &prime, 4);
                buf_len += 4;
            } else {
                uint32_t gap = prime - prev_prime;
                if (gap > 255) {
                    /* Extremely rare for p < 4*10^18; encode as 0x00 + uint16_t */
                    outbuf[buf_len++] = 0;
                    outbuf[buf_len++] = (uint8_t)(gap & 0xFF);
                    outbuf[buf_len++] = (uint8_t)(gap >> 8);
                } else {
                    outbuf[buf_len++] = (uint8_t)gap;
                }
            }
            prev_prime = prime;
            total++;
        }
        if (buf_len > 0) fwrite(outbuf, 1, buf_len, fout);
    }

    /* Handle 2 separately — prepend it by rewriting is complex; instead: */
    /* We'll add 2 by rewriting the file start. Actually let's just not include
     * 2 since our formula N = 10^a - k*10^b - j*10^c - 1 is always odd.
     * For trial division of odd N, only odd primes matter. */

    /* Fill count */
    rewind(fout);
    fwrite(&total, sizeof(uint32_t), 1, fout);
    fclose(fout);

    free(seg); free(outbuf); free(small_p);
    fprintf(stderr, "wrote %u primes up to %llu → %s\n",
            total, (unsigned long long)limit, argv[2]);
    return 0;
}
