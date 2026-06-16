// ops/sub/sub.cuh — Subtração big-int tileada (prefix-scan de borrow G/P/K), batched.
//
// Subtractor genérico usado por TODAS as reduções (Barrett finalize e Montgomery
// cond_sub). Suporta strides separados para a/b/out, largura de b por-candidato
// (bk) e modo incondicional (uncond) ou condicional (só subtrai se a >= b).
//
// Os buffers de tile (tile_cmp/tile_bstate) devem ter n_tiles = ceil(W/MR_SUB_TILE)
// inteiros por candidato. O borrow_in é resolvido DENTRO do apply (fundido).
#pragma once

#include "ops/mul/multiplier.cuh" // Data64, LIMB_BITS (via backend selecionado)
#include <cuda_runtime.h>

namespace ops
{
    // Fase 1: por tile, compara a vs b e grava cmp + estado de borrow.
    void sub_phase1(const Data64 *a, int sa, const Data64 *b, int sb,
                    const int *bk, int W, int *tile_cmp, int *tile_bstate,
                    int n_batch, cudaStream_t s);

    // Fase 2 (fundida com resolve): out = a − b com borrow tileado correto.
    // uncond != 0 ⇒ sempre subtrai; senão só quando a >= b (caso contrário no-op).
    void sub_apply(Data64 *out, int so, const Data64 *a, int sa, const Data64 *b, int sb,
                   const int *bk, int W, const int *tile_cmp, const int *tile_bstate,
                   int uncond, int n_batch, cudaStream_t s);

    // out[cand*out_limbs + j] = (j < W) ? r[cand*W + j] : 0 — copia limbs baixos.
    void copy_low(Data64 *out, const Data64 *r, int out_limbs, int W,
                  int n_batch, int thr, cudaStream_t s);

    // nº de tiles para uma largura W (= grid.x das fases; dimensiona os buffers).
    int sub_n_tiles(int W);
}
