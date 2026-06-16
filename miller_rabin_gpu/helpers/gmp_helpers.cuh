#pragma once
// helpers/gmp_helpers.cuh — conversões host entre limbs de 16 bits (little-endian) e mpz_t.
// Compartilhado por batch_mod_ctx.cu e pelos arquivos de redução (reductions/*.cu).

#include <cstdint>
#include <gmp.h>

// limbs little-endian de 16 bits → mpz_t.
static inline void limbs_to_mpz(mpz_t out, const uint64_t *lims, int n)
{
    mpz_set_ui(out, 0);
    for (int i = n - 1; i >= 0; i--)
    {
        mpz_mul_2exp(out, out, 16);
        mpz_add_ui(out, out, (unsigned long)lims[i]);
    }
}

// mpz_t → n limbs little-endian de 16 bits (trunca acima de n).
static inline void mpz_to_limbs(uint64_t *out, int n, const mpz_t x)
{
    mpz_t tmp;
    mpz_init_set(tmp, x);
    for (int i = 0; i < n; i++)
    {
        out[i] = mpz_get_ui(tmp) & 0xFFFF;
        mpz_tdiv_q_2exp(tmp, tmp, 16);
    }
    mpz_clear(tmp);
}
