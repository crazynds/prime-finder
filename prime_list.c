#include "prime_list.h"
#include <stdio.h>
#include <stdlib.h>

/*
 * File format:
 *   [uint32_t count]
 *   [uint32_t first_prime]
 *   [uint8_t gap...]   gap[i] = p[i] - p[i-1]
 *                      gap == 0x00 followed by uint16_t for gaps > 255 (rare)
 */
int prime_list_load(prime_list_t *pl, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }

    uint32_t count;
    if (fread(&count, sizeof(uint32_t), 1, f) != 1) { fclose(f); return -1; }

    pl->count  = count;
    pl->primes = malloc((size_t)count * sizeof(uint32_t));
    if (!pl->primes) { fclose(f); return -1; }

    if (count == 0) { fclose(f); return 0; }

    /* First prime stored as absolute uint32_t */
    if (fread(&pl->primes[0], sizeof(uint32_t), 1, f) != 1) goto fail;

    /* Remaining primes decoded from gaps */
    for (uint32_t i = 1; i < count; i++) {
        uint8_t gap8;
        if (fread(&gap8, 1, 1, f) != 1) goto fail;

        uint32_t gap;
        if (gap8 == 0) {
            /* Extended gap: next 2 bytes are little-endian uint16_t */
            uint8_t lo, hi;
            if (fread(&lo, 1, 1, f) != 1) goto fail;
            if (fread(&hi, 1, 1, f) != 1) goto fail;
            gap = (uint32_t)lo | ((uint32_t)hi << 8);
        } else {
            gap = gap8;
        }
        pl->primes[i] = pl->primes[i-1] + gap;
    }

    fclose(f);
    return 0;

fail:
    free(pl->primes);
    pl->primes = NULL;
    fclose(f);
    return -1;
}

void prime_list_free(prime_list_t *pl)
{
    free(pl->primes);
    pl->primes = NULL;
    pl->count  = 0;
}
