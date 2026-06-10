#pragma once
// montgomery.cuh — Montgomery batched, um MontCtx para todos os candidatos.
//
// Layout: todos os arrays [n_batch * stride], batch_i = candidato.
// NTT chamada uma unica vez com batch=n_batch.

#include "config.cuh"
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

    // Acumuladores de tempo por seção de kernel (ms, via CUDA events).
    // Cada campo corresponde a exatamente um par TSTART/TSTOP.
    struct PerfInner {
        // mul: load_padded(A)+NTT(A) + load_padded(B)+NTT(B)  |  sq: load_padded(A)+NTT(A)
        float ntt_input_ms     = 0;
        // mul: pmul_batch + INTT  |  sq: psq_batch + INTT
        float ntt_product_ms   = 0;
        // carry_intra + carry_inter sobre T (resultado do produto/quadrado)
        float carry_product_ms = 0;
        // reduce: extract_low(T) + fwd_A — prepara T_low para multiplicar por N'
        float red_ntt_tlow_ms  = 0;
        // reduce: pmul_ext(N') + INTT — m = T_low * N' no domínio NTT
        float red_pmul_np_ms   = 0;
        // reduce: carry_intra + carry_inter sobre m
        float red_carry_m_ms   = 0;
        // reduce: load_padded(m) + NTT(m) — transforma m para multiplicar por N
        float red_ntt_m_ms     = 0;
        // reduce: pmul_ext(N) + INTT — mN = m * N no domínio NTT
        float red_pmul_n_ms    = 0;
        // reduce: vadd_from_raw + carry_intra + carry_inter — T += mN, normaliza
        float red_add_carry_ms = 0;
        // reduce: shift_right(T, n_limbs) — resultado final = T[n_limbs..]
        float red_shift_ms     = 0;
        // cs_phase1 + cs_resolve + cs_apply — subtração condicional mod N
        float cond_sub_ms      = 0;

        void print() const {
            float total = ntt_input_ms + ntt_product_ms + carry_product_ms
                        + red_ntt_tlow_ms + red_pmul_np_ms + red_carry_m_ms
                        + red_ntt_m_ms + red_pmul_n_ms + red_add_carry_ms
                        + red_shift_ms + cond_sub_ms;
            auto pct = [&](float v) { return total > 0 ? v * 100.0f / total : 0.0f; };
            auto row = [&](const char* name, float ms) {
                printf("  ├─ %-22s %8.3fs  %5.1f%%\n", name, ms/1000.0f, pct(ms));
            };
            row("ntt_input",        ntt_input_ms);
            row("ntt_product",      ntt_product_ms);
            row("carry_product",    carry_product_ms);
            row("red:ntt_Tlow",     red_ntt_tlow_ms);
            row("red:pmul_Np+INTT", red_pmul_np_ms);
            row("red:carry_m",      red_carry_m_ms);
            row("red:ntt_m",        red_ntt_m_ms);
            row("red:pmul_N+INTT",  red_pmul_n_ms);
            row("red:add_carry",    red_add_carry_ms);
            row("red:shift_right",  red_shift_ms);
            row("cond_sub",         cond_sub_ms);
            printf("  └─ %-22s %8.3fs\n", "TOTAL", total/1000.0f);
        }
    } perf;

    // Ring de eventos para profiling sem sync no hot path.
    // TSTART/TSTOP gravam eventos sem bloquear; perf_flush() sincroniza UMA vez
    // no final de cada mont_mul_batch/mont_sq_batch e acumula todos os tempos.
    // 12 cobre as 11 seções máximas de mont_mul_batch (reduce tem 7 + 4 extras).
    static constexpr int PERF_RING = 32;
    cudaEvent_t ev_ring[PERF_RING + 1] = {};
    float*      acc_ring[PERF_RING]    = {};
    int         ring_cur               = 0;

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
