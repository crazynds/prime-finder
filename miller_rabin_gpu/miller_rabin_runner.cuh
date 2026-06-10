#pragma once
// miller_rabin_runner.cuh — Execução do teste de Miller-Rabin em GPU para um batch.
//
// gpu_miller_rabin_s1: otimizado para N-1 = 2*d (s=1, N ≡ 3 mod 4).
// gpu_miller_rabin:    versão geral para N-1 = 2^s * d, qualquer s >= 1.

#include "montgomery.cuh"
#include "config.cuh"
#include <vector>
#define MR_WINDOW_SIZE (1 << MR_WINDOW_BITS)

static constexpr int WINDOW_BITS = MR_WINDOW_BITS;
static constexpr int WINDOW_SIZE = MR_WINDOW_SIZE;

inline const std::vector<uint32_t> DEFAULT_WITNESSES = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53
};

// Seleciona table[w] para cada candidato dado uma janela de WINDOW_BITS bits.
// Declarado aqui para poder ser referenciado em correctness_tests.cuh.
__global__ void select_window_kernel(
        Data64* __restrict__,
        const Data64* __restrict__,
        const Data64* __restrict__,
        int, int, int, int);

// Para números onde N-1 = 2*d (s=1).
// exp_all: d = (N-1)/2, flat [n_total * n_limbs].
std::vector<bool> gpu_miller_rabin_s1(
        BatchMontCtx& mont,
        const std::vector<uint64_t>& exp_all,
        const std::vector<uint64_t>& Nm1_all,
        int n_total,
        const std::vector<uint32_t>& witnesses,
        const char* label,
        bool show_report   = false,
        bool show_progress = false);

// Versão geral: N-1 = 2^s * d.
// exp_all: d (ímpar), s: número de fatores de 2 em N-1.
std::vector<bool> gpu_miller_rabin(
        BatchMontCtx& mont,
        const std::vector<uint64_t>& exp_all,
        const std::vector<uint64_t>& Nm1_all,
        int s,
        int n_total,
        const std::vector<uint32_t>& witnesses,
        const char* label,
        bool show_report   = false,
        bool show_progress = false);
