// ops/mul/multiplier.cuh — Seleção COMPILE-TIME do backend de multiplicação big-int.
//
// As reduções (Barrett/Montgomery) e o orquestrador programam contra o tipo
// `Multiplier` e NUNCA contêm #if por backend. Trocar de algoritmo = trocar
// MUL_ALG em params.cmake e recompilar.
//
// Contrato que todo backend deve expor (mesma superfície do BigIntNTTBatch):
//   ints: n_limbs, padded, n_sum-equivalente (via padded), n_batch
//   Data64* d_buf_A;                       // slot de trabalho no domínio transformado
//   ntt_A / ntt_B / ntt_AB(src,len)        // forward transform de operando(s) variável(is)
//   fwd_A()                                // forward de d_buf_A já preenchido
//   pmul() / psq() / pmul_ext(pre)         // pointwise (A*B, A², A*precomputado)
//   intt_A()                               // inverse transform
//   carry_to_limbs(out, out_len)           // domínio → limbs normalizados (com carry)
//   vadd_raw_buf / carry_after_vadd / add_raw_buf_and_carry  // soma raw + carry (REDC)
//   pmul_and_intt / psq_and_intt / pmul_ext_and_intt         // compostos
//   schoolbook_mul / schoolbook_sq         // convolução direta (n pequeno)
//
// Backends que não suportam alguma operação devem oferecer o equivalente ou
// abortar de forma explícita.
#pragma once

#include "config.h" // MUL_ALG + identificadores MUL_* (via constants.h)

#ifndef MUL_ALG
#error "MUL_ALG não definido (params.cmake → config.h). Use MUL_SCHOOLBOOK | MUL_NTT_MERGE | MUL_NTT_4STEP."
#endif

// Seleção da classe de multiplicação. SCHOOLBOOK e NTT_MERGE compartilham a classe
// "merge" (BigIntNTTBatch) — a redução modular sempre usa NTT, e a classe merge
// provê tanto o NTT quanto o schoolbook_*. Só NTT_4STEP troca a classe.
#if MUL_ALG == MUL_NTT_4STEP
#include "ops/mul/ntt_4step.cuh"   // classe Ntt4StepBatch (mesma API, algoritmo radix)
using Multiplier = Ntt4StepBatch;
#elif MUL_ALG == MUL_NTT_MERGE || MUL_ALG == MUL_SCHOOLBOOK
#include "ops/mul/ntt_merge.cuh"   // classe BigIntNTTBatch (backend merge da GPU-NTT)
using Multiplier = BigIntNTTBatch;
#else
#error "MUL_ALG inválido. Use MUL_SCHOOLBOOK | MUL_NTT_MERGE | MUL_NTT_4STEP."
#endif
