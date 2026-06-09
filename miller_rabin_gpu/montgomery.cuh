#pragma once
// montgomery.cuh — Montgomery batched, um MontCtx para todos os candidatos.
//
// Layout: todos os arrays [n_batch * stride], batch_i = candidato.
// NTT chamada uma unica vez com batch=n_batch.

#include "bigint_ntt.cuh"
#include "config.cuh"
#include <vector>
#include <gmp.h>
#include <cuda_runtime.h>

struct BatchMontCtx {
    int n_limbs, n_batch, padded, n_sum;
    BigIntNTTBatch ntt;

    // Per-candidate data, [n_batch * n_limbs]
    Data64* d_N      = nullptr;
    Data64* d_Nprime = nullptr;

    // Pre-computado NTT(N) e NTT(N'), [n_batch * padded] — so leitura no hot path
    Data64* d_ntt_N      = nullptr;
    Data64* d_ntt_Nprime = nullptr;

    // Valores de referencia em forma Montgomery — para check sem GMP
    // [n_batch * n_limbs] cada
    Data64* d_one_mont  = nullptr;   // to_mont(1)   por candidato
    Data64* d_Nm1_mont  = nullptr;   // to_mont(N-1) por candidato

    // Buffers de trabalho
    Data64* d_T  = nullptr;   // [n_batch * n_sum]
    Data64* d_m  = nullptr;   // [n_batch * padded]  (NTT workspace de m)

    // Buffers para cond_sub tileado [n_batch * n_cs_tiles]
    int n_cs_tiles = 0;
    int* d_cs_tile_cmp    = nullptr;  // cmp por tile: 1, -1, 0
    int* d_cs_tile_bstate = nullptr;  // estado G/P/K de borrow por tile
    int* d_cs_tile_bin    = nullptr;  // borrow_in resolvido por tile

    // Acumuladores de tempo para profiling (ms, via CUDA events)
    struct PerfInner {
        float ntt_fwd_ms  = 0;  // NTT forward (ntt_A, ntt_B)
        float ntt_inv_ms  = 0;  // NTT inverse (psq, pmul, pmul_ext + INTT)
        float carry_ms    = 0;  // carry_to_limbs + add_and_carry
        float cond_sub_ms = 0;  // cond_sub_batch (3 kernels)
        float other_ms    = 0;  // extract_low, shift_right, etc.
        void print() const {
            float total = ntt_fwd_ms + ntt_inv_ms + carry_ms + cond_sub_ms + other_ms;
            auto pct = [&](float v) { return total > 0 ? v * 100.0f / total : 0.0f; };
            printf("  ├─ %-18s %8.2fs  %5.1f%%\n", "ntt_forward",  ntt_fwd_ms /1000.0f, pct(ntt_fwd_ms));
            printf("  ├─ %-18s %8.2fs  %5.1f%%\n", "ntt_inverse",  ntt_inv_ms /1000.0f, pct(ntt_inv_ms));
            printf("  ├─ %-18s %8.2fs  %5.1f%%\n", "carry_norm",   carry_ms   /1000.0f, pct(carry_ms));
            printf("  ├─ %-18s %8.2fs  %5.1f%%\n", "cond_sub",     cond_sub_ms/1000.0f, pct(cond_sub_ms));
            printf("  ├─ %-18s %8.2fs  %5.1f%%\n", "other",        other_ms   /1000.0f, pct(other_ms));
            printf("  └─ %-18s %8.2fs\n",           "TOTAL interno",total      /1000.0f);
        }
    } perf;

    // Ring de eventos para profiling sem sync no hot path.
    // TSTART/TSTOP gravam eventos sem bloquear; perf_flush() sincroniza UMA vez
    // no final de cada mont_mul_batch/mont_sq_batch e acumula todos os tempos.
    //
    // Tamanho: mont_mul_batch gera no máximo 11 seções TSTART/TSTOP (reduce_batch
    // tem 7 + 4 extras), então 12 eventos são suficientes para cobrir um call.
    static constexpr int PERF_RING = MR_PERF_RING;
    cudaEvent_t ev_ring[MR_PERF_RING + 1] = {};   // PERF_RING seções → PERF_RING+1 marcos
    float*      acc_ring[MR_PERF_RING]    = {};   // ponteiro para o acumulador de cada seção
    int         ring_cur               = 0;    // número de seções abertas

    int device_id = 0;  // GPU utilizada

    // Construtor a partir de limbs pré-computados.
    // device_id: índice da GPU (0 por padrão; use cudaGetDeviceCount para listar).
    // N_all: vetor flat [n_batch * n_limbs], little-endian 16-bit limbs.
    explicit BatchMontCtx(const std::vector<uint64_t>& N_all, int n_limbs_, int n_batch_,
                          int device_id_ = 0);

    // Construtor de conveniência: aceita os números diretamente como mpz_t.
    // Calcula n_limbs automaticamente a partir do maior número do vetor.
    explicit BatchMontCtx(const std::vector<mpz_t*>& numbers, int device_id_ = 0);
    ~BatchMontCtx();

    // x_all (host, n_batch * n_limbs) -> forma Montgomery (host)
    void to_mont_batch(const std::vector<uint64_t>& x_all,
                       std::vector<uint64_t>& out_all) const;

    // d_x (GPU, forma Montgomery) -> valores normais (host)
    void from_mont_batch(const Data64* d_x, std::vector<uint64_t>& out_all) const;

    // Verifica resultados no GPU: para cada candidato, r_mont == 1_mont ou (N-1)_mont?
    // d_passed[t] = 1 se passou, 0 se composto. n_total elementos.
    void check_passed(const Data64* d_r_mont, uint8_t* d_passed, cudaStream_t s = 0) const;

    // d_out = mont_mul(d_A, d_B) para todos os n_batch candidatos
    void mont_mul_batch(const Data64* d_A, const Data64* d_B, Data64* d_out,
                        cudaStream_t s = 0);
    // d_out = mont_sq(d_A) para todos os n_batch candidatos
    void mont_sq_batch(const Data64* d_A, Data64* d_out, cudaStream_t s = 0);

    BatchMontCtx(const BatchMontCtx&) = delete;
    BatchMontCtx& operator=(const BatchMontCtx&) = delete;

private:
    void reduce_batch(Data64* d_out, cudaStream_t s);
    void cond_sub_batch(Data64* d_x, cudaStream_t s);
    // Sincroniza o último evento do ring e acumula todos os tempos pendentes.
    // Chamado UMA VEZ no final de cada mont_mul_batch / mont_sq_batch.
    void perf_flush(cudaStream_t s);
};
