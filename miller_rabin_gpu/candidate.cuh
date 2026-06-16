#pragma once
// candidate.cuh — Representa um número candidato a primo para o teste de Miller-Rabin.
//
// NumberCandidate: candidato genérico — aceita qualquer mpz_t.
// SparseCandidate: formato específico 10^a - k*10^b - j*10^c - 1, com revN.

#include <vector>
#include <gmp.h>
#include "ops/mul/multiplier.cuh"

// ── Utilitários GMP ───────────────────────────────────────────────────────────

static inline void mpz_to_limbs_vec(uint64_t* out, int n, const mpz_t x)
{
    mpz_t tmp; mpz_init_set(tmp, x);
    for (int i = 0; i < n; i++) {
        out[i] = mpz_get_ui(tmp) & LIMB_MASK;
        mpz_tdiv_q_2exp(tmp, tmp, LIMB_BITS);
    }
    mpz_clear(tmp);
}

static inline int ndig(int x) { return (x < 10) ? 1 : (x < 100) ? 2 : 3; }
static inline int revd(int x, int l) { int r = 0; for (int i = 0; i < l; i++) { r = r*10 + (x%10); x /= 10; } return r; }

// ── Candidato genérico ────────────────────────────────────────────────────────
//
// Representa N com sua decomposição N-1 = 2^s * d (d ímpar).
// Constrói a partir de qualquer mpz_t via build_from_mpz().

struct NumberCandidate {
    int s = 0;  // N-1 = 2^s * d

    std::vector<uint64_t> N_lims;
    std::vector<uint64_t> Nm1_lims;
    std::vector<uint64_t> d_lims;   // d tal que N-1 = 2^s * d, d ímpar

    bool is_s1() const { return s == 1; }

    void build_from_mpz(const mpz_t N, int n_limbs)
    {
        N_lims.assign(n_limbs, 0);
        Nm1_lims.assign(n_limbs, 0);
        d_lims.assign(n_limbs, 0);

        mpz_t Nm1, d; mpz_inits(Nm1, d, nullptr);
        mpz_sub_ui(Nm1, N, 1);

        // Encontra s: maior potência de 2 que divide N-1
        s = 0;
        mpz_set(d, Nm1);
        while (mpz_even_p(d)) {
            mpz_tdiv_q_2exp(d, d, 1);
            s++;
        }

        mpz_to_limbs_vec(N_lims.data(),   n_limbs, N);
        mpz_to_limbs_vec(Nm1_lims.data(), n_limbs, Nm1);
        mpz_to_limbs_vec(d_lims.data(),   n_limbs, d);

        mpz_clears(Nm1, d, nullptr);
    }
};

// ── Candidato esparso ─────────────────────────────────────────────────────────
//
// Formato: N = 10^a - k*10^b - j*10^c - 1
// Também computa revN (dígitos revertidos) para teste duplo.

struct SparseCandidate {
    long long a, b, c;
    int k, j;

    NumberCandidate N_cand;
    NumberCandidate revN_cand;

    void build(int n_limbs)
    {
        mpz_t N, revN, tmp;
        mpz_inits(N, revN, tmp, nullptr);

        // N = 10^a - k*10^b - j*10^c - 1
        mpz_ui_pow_ui(N, 10, (unsigned long)a);
        mpz_ui_pow_ui(tmp, 10, (unsigned long)b); mpz_mul_ui(tmp, tmp, (unsigned long)k);
        mpz_sub(N, N, tmp);
        mpz_ui_pow_ui(tmp, 10, (unsigned long)c); mpz_mul_ui(tmp, tmp, (unsigned long)j);
        mpz_sub(N, N, tmp);
        mpz_sub_ui(N, N, 1);

        // revN: dígitos revertidos de k e j nos expoentes correspondentes
        int lk = ndig(k), lj = ndig(j);
        long long eb = a - b - lk, ec = a - c - lj;
        mpz_ui_pow_ui(revN, 10, (unsigned long)a);
        mpz_ui_pow_ui(tmp, 10, (unsigned long)eb); mpz_mul_ui(tmp, tmp, (unsigned long)revd(k, lk));
        mpz_sub(revN, revN, tmp);
        mpz_ui_pow_ui(tmp, 10, (unsigned long)ec); mpz_mul_ui(tmp, tmp, (unsigned long)revd(j, lj));
        mpz_sub(revN, revN, tmp);
        mpz_sub_ui(revN, revN, 1);

        N_cand.build_from_mpz(N, n_limbs);
        revN_cand.build_from_mpz(revN, n_limbs);

        mpz_clears(N, revN, tmp, nullptr);
    }
};
