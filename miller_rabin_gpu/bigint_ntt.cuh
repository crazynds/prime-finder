#pragma once
// bigint_ntt.cuh — NTT bigint multiply, 16-bit limbs, primo unico, n_batch polys.
//
// Layout de dados: buf[batch_i * padded + coeff_j]
// Uma chamada GPU_NTT_Inplace processa todos os n_batch polinomios de uma vez.

#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "gpuntt/ntt_merge/ntt.cuh"

using namespace gpuntt;

// ── Tamanho dos limbs ─────────────────────────────────────────────────────────
// #define LIMB_BITS 32
#ifndef LIMB_BITS
#  define LIMB_BITS 16
#endif
#define LIMB_MASK ((1ULL << LIMB_BITS) - 1ULL)

inline int limbs_for_digits(int decimal_digits)
{
    return (int)((decimal_digits * 3.32193 + LIMB_BITS - 1) / LIMB_BITS) + 4;
}

inline int next_pow2_ntt(int n)
{
    int p = 1; while (p < n) p <<= 1; return p;
}

static constexpr int CARRY_PASSES_MUL = 4;
static constexpr int CARRY_PASSES_ADD = 2;

struct BigIntNTTBatch {
    int n_limbs, padded, logn, n_batch;

    Data64          p_val;
    Ninverse64      n_inv;
    Modulus<Data64> modulus;

    Root64* d_fwd_table = nullptr;
    Root64* d_inv_table = nullptr;

    // d_buf_A e d_buf_B são contíguos: d_buf_AB[0..n_batch*padded-1] = A,
    // d_buf_AB[n_batch*padded..2*n_batch*padded-1] = B.
    // Isso permite chamar GPU_NTT_Inplace(d_buf_A, 2*n_batch) para transformar
    // A e B em um único lançamento de kernel.
    Data64* d_buf_AB     = nullptr;  // alocação única [2 * n_batch * padded]
    Data64* d_buf_A      = nullptr;  // aponta para d_buf_AB
    Data64* d_buf_B      = nullptr;  // aponta para d_buf_AB + n_batch * padded
    Data64* d_tile_carry = nullptr;  // [n_batch * n_tiles] carry inter-tile

    explicit BigIntNTTBatch(int n_limbs_, int n_batch_);
    ~BigIntNTTBatch();

    // Carrega d_src [n_batch * n_src] em d_buf_A com zero-pad ate padded, depois NTT
    void ntt_A(const Data64* d_src, int n_src, cudaStream_t s = 0);
    // Idem para d_buf_B
    void ntt_B(const Data64* d_src, int n_src, cudaStream_t s = 0);
    // Carrega d_srcA em buf_A e d_srcB em buf_B, depois NTT em batch (2*n_batch de uma vez)
    void ntt_AB(const Data64* d_srcA, const Data64* d_srcB, int n_src, cudaStream_t s = 0);

    // d_buf_A = d_buf_A * d_buf_B (pointwise), INTT -> d_buf_A
    void pmul_and_intt(cudaStream_t s = 0);
    // d_buf_A = d_buf_A^2 (pointwise), INTT -> d_buf_A
    void psq_and_intt(cudaStream_t s = 0);
    // d_buf_A = d_buf_A * d_ext (externo, ja em dominio NTT [n_batch*padded]), INTT -> d_buf_A
    void pmul_ext_and_intt(const Data64* d_ext, cudaStream_t s = 0);
    // Apenas NTT forward em d_buf_A (ja preenchido externamente com zero-pad)
    void fwd_A(cudaStream_t s = 0);

    // Copia d_buf_A -> d_out [n_batch * n_out] e normaliza carries
    void carry_to_limbs(Data64* d_out, int n_out, int n_passes = CARRY_PASSES_MUL,
                        cudaStream_t s = 0);
    // d_a += d_b (ambos [n_batch * n]), depois normaliza carries
    void add_and_carry(Data64* d_a, const Data64* d_b, int n, int n_passes,
                       cudaStream_t s = 0);
    // d_dst += d_buf_A (bruto, stride=padded) e normaliza carries em uma passagem.
    // Equivale a carry_to_limbs(tmp) + add_and_carry(d_dst, tmp), sem buffer intermediário.
    void add_raw_buf_and_carry(Data64* d_dst, int n_dst, int n_passes,
                               cudaStream_t s = 0);

    BigIntNTTBatch(const BigIntNTTBatch&) = delete;
    BigIntNTTBatch& operator=(const BigIntNTTBatch&) = delete;

private:
    ntt_configuration<Data64> make_cfg(type t, cudaStream_t s);
};
