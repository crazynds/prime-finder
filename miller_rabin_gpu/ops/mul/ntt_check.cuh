#pragma once
// ops/mul/ntt_check.cuh — checagens de spec compartilhadas pelos backends NTT.
// Incluir DEPOIS do header do backend (precisa de LIMB_MASK).

#include <stdexcept>
#include <string>
#include <cstdint>

// Garantia de precisão da NTT inteira sobre base 2^LIMB_BITS.
//
// A convolução produz coeficientes = Σ A[i]·B[k-i]. O número de termos somados é,
// no pior caso deste sistema, ≤ padded/2 (operandos ocupam ≤ n_limbs ≤ padded/2
// limbs, pois padded ≥ 2·n_limbs). Cada termo ≤ (2^LIMB_BITS − 1)². Se o maior
// coeficiente possível atingir o primo da NTT, há wraparound silencioso → produto
// errado. Lançamos um erro claro em vez de produzir lixo.
inline void check_ntt_precision(int padded, unsigned long long p_val)
{
    const __uint128_t max_terms = (__uint128_t)(padded / 2);
    const __uint128_t lm = (__uint128_t)LIMB_MASK; // 2^LIMB_BITS − 1
    const __uint128_t max_coeff = max_terms * lm * lm;
    if (max_coeff >= (__uint128_t)p_val)
        throw std::runtime_error(
            "[ntt] precisão insuficiente: (padded/2)*(2^LIMB_BITS-1)^2 (~"
            "termos=" + std::to_string(padded / 2) + ", LIMB_BITS=" + std::to_string(LIMB_BITS) +
            ") >= primo da NTT (" + std::to_string(p_val) +
            "). Reduza LIMB_BITS ou o tamanho do operando.");
}
