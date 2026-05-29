#include "sieve.h"
#include <string.h>

void sieve_apply(const sieve_table_t *st,
                 long long a, long long b, long long c,
                 uint32_t *out_bits)
{
    for (int i = 0; i < st->n; i++) {
        const sieve_entry_t *e = &st->entries[i];

        int ra = (int)(a % e->L10);
        int rb = (int)(b % e->L10);
        int rc = (int)(c % e->L10);  /* all three use ord_p(10) */

        long long cell = ((long long)ra * e->L10 + rb) * e->L10 + rc;
        const uint32_t *base = (const uint32_t *)((const char *)st->flat_data + e->bitmask_offset);
        const uint32_t *bits = base + cell * KJ_WORDS;

        for (int w = 0; w < KJ_WORDS; w++)
            out_bits[w] |= bits[w];
    }
}

int sieve_all_eliminated(const uint32_t *bits)
{
    for (int w = 0; w < KJ_WORDS - 1; w++)
        if (bits[w] != 0xFFFFFFFFu) return 0;
    /* Last word: 10000 % 32 = 16 valid bits */
    if ((bits[KJ_WORDS-1] & 0xFFFFu) != 0xFFFFu) return 0;
    return 1;
}
