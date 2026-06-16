// ops/shift/shift.cuh — Deslocamento/extração de limbs (base 2^LIMB_BITS), batched.
// Operações agnósticas ao algoritmo de redução; lançadores host em shift.cu.
#pragma once

#include "ops/mul/ntt_merge.cuh" // Data64, LIMB_BITS
#include <cuda_runtime.h>

namespace ops
{
    // dst[cand*n_out + j] = src[cand*n_src + j + offset] (0 fora do range).
    // Shift à direita por `offset` limbs, igual para todos os candidatos.
    void shift_right(Data64 *dst, const Data64 *src, int offset,
                     int n_out, int n_src, int n_batch, int thr, cudaStream_t s);

    // Idem, mas offset = bark[cand] + delta (por-candidato).
    void shift_right_var(Data64 *dst, const Data64 *src, const int *bark, int delta,
                         int n_out, int n_src, int n_batch, int thr, cudaStream_t s);

    // dst[cand*padded + j] = (j < n_low) ? src[cand*n_sum + j] : 0 — extrai limbs baixos.
    void extract_low(Data64 *dst, const Data64 *src, int n_low, int padded,
                     int n_sum, int n_batch, int thr, cudaStream_t s);
}
