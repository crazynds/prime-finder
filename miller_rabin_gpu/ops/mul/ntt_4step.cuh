#pragma once
// ops/mul/ntt_4step.cuh — Backend de multiplicação big-int via NTT "4-step" (radix)
// da GPU-NTT (Alisah-Ozcan/GPU-NTT). Mesma superfície pública que BigIntNTTBatch
// (ver ops/mul/multiplier.cuh): as reduções e o orquestrador NÃO sabem qual backend
// está ativo. Selecionado por MUL_ALG == MUL_NTT_4STEP (params.cmake).
//
// Diferenças vs. o backend "merge":
//   • Transformada out-of-place com 2 transposes (ping-pong entre d_buf_* e d_scratch).
//   • 3 tabelas de raízes (n1, n2, W) para forward e 3 para inverse.
//   • Restrição da lib: logn ∈ [12, 24].
//
// O domínio transformado (após o transpose final) é consistente entre forward e
// inverse e o pointwise é elementwise, então pmul/psq/pmul_ext valem como no merge.

#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "gpuntt/ntt_merge/ntt.cuh"   // aliases Data64/Root64/Ninverse64/Modulus
#include "gpuntt/common/nttparameters.cuh"
#include "gpuntt/ntt_4step/ntt_4step.cuh"

using namespace gpuntt;

#ifndef LIMB_BITS
#define LIMB_BITS 16
#endif
#ifndef LIMB_MASK
#define LIMB_MASK ((1ULL << LIMB_BITS) - 1ULL)
#endif

#ifndef NTT_HELPERS_DEFINED
#define NTT_HELPERS_DEFINED
inline int limbs_for_digits(int decimal_digits)
{
    return (int)((decimal_digits * 3.32193 + LIMB_BITS - 1) / LIMB_BITS) + 4;
}
inline int next_pow2_ntt(int n)
{
    int p = 1;
    while (p < n)
        p <<= 1;
    return p;
}
#endif

struct Ntt4StepBatch
{
    int n_limbs, padded, logn, n_batch;
    int n1, n2; // padded = n1 * n2

    Data64 p_val;
    Ninverse64 n_inv;
    Modulus<Data64> modulus;

    // Tabelas de raízes (device). Forward e inverse separadas.
    Root64 *d_n1_fwd = nullptr; // [n1>>1]
    Root64 *d_n2_fwd = nullptr; // [n2>>1]
    Root64 *d_W_fwd = nullptr;  // [padded]
    Root64 *d_n1_inv = nullptr;
    Root64 *d_n2_inv = nullptr;
    Root64 *d_W_inv = nullptr;

    // modulus / ninverse no device (variante RNS de GPU_4STEP_NTT, mod_count=1).
    Modulus<Data64> *d_modulus = nullptr;
    Ninverse64 *d_ninverse = nullptr;

    // Buffers de trabalho — mesma convenção do merge: A e B contíguos.
    Data64 *d_buf_AB = nullptr; // [2 * n_batch * padded]
    Data64 *d_buf_A = nullptr;  // = d_buf_AB
    Data64 *d_buf_B = nullptr;  // = d_buf_AB + n_batch*padded
    Data64 *d_scratch = nullptr; // [2 * n_batch * padded] — ping-pong dos transposes
    Data64 *d_tile_carry = nullptr; // [n_batch * n_tiles]

    explicit Ntt4StepBatch(int n_limbs_, int n_batch_);
    ~Ntt4StepBatch();

    // ── Forward transforms ────────────────────────────────────────────────────
    void ntt_A(const Data64 *d_src, int n_src, cudaStream_t s = 0);
    void ntt_B(const Data64 *d_src, int n_src, cudaStream_t s = 0);
    void ntt_AB(const Data64 *d_srcA, const Data64 *d_srcB, int n_src, cudaStream_t s = 0);
    void fwd_A(cudaStream_t s = 0); // d_buf_A já preenchido (zero-pad) → forward in-place

    // ── Pointwise ───────────────────────────────────────────────────────────────
    void pmul(cudaStream_t s = 0);
    void psq(cudaStream_t s = 0);
    void pmul_ext(const Data64 *d_ext, cudaStream_t s = 0);

    // ── Inverse ───────────────────────────────────────────────────────────────
    void intt_A(cudaStream_t s = 0);

    // ── Compostos ───────────────────────────────────────────────────────────────
    void pmul_and_intt(cudaStream_t s = 0);
    void psq_and_intt(cudaStream_t s = 0);
    void pmul_ext_and_intt(const Data64 *d_ext, cudaStream_t s = 0);

    // ── Schoolbook (MUL_SCHOOLBOOK) ────────────────────────────────────
    void schoolbook_mul(const Data64 *d_A, const Data64 *d_B, int n_src, cudaStream_t s = 0);
    void schoolbook_sq(const Data64 *d_A, int n_src, cudaStream_t s = 0);

    // ── Carry / soma (definidos em ops/carry/carry_norm.cu, agnósticos) ─────────
    void carry_to_limbs(Data64 *d_out, int n_out, cudaStream_t s = 0);
    void add_and_carry(Data64 *d_a, const Data64 *d_b, int n, int n_passes, cudaStream_t s = 0);
    void vadd_raw_buf(Data64 *d_dst, int n_dst, cudaStream_t s = 0);
    void carry_after_vadd(Data64 *d_dst, int n_dst, cudaStream_t s = 0);
    void add_raw_buf_and_carry(Data64 *d_dst, int n_dst, cudaStream_t s = 0);

    Ntt4StepBatch(const Ntt4StepBatch &) = delete;
    Ntt4StepBatch &operator=(const Ntt4StepBatch &) = delete;

private:
    // Transformada 4-step completa (T · NTT · T, 3 ops), forward ou inverse.
    // `src` contém a entrada; o resultado é escrito em `dst` (src é usado como
    // scratch do ping-pong). src e dst devem ser buffers distintos. `fwd` escolhe
    // as tabelas/tipo (FORWARD vs INVERSE).
    void transform(Data64 *src, Data64 *dst, bool fwd, int batch, cudaStream_t s);
};

void carry_stats_print_and_reset();
