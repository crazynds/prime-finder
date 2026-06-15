#pragma once
// montgomery.cuh — Montgomery batched, um MontCtx para todos os candidatos.
//
// Layout: todos os arrays [n_batch * stride], batch_i = candidato.
// NTT chamada uma unica vez com batch=n_batch.

#include "config.h"
#include "bigint_ntt.cuh"
#include "time_format.h"
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include <gmp.h>
#include <cuda_runtime.h>

// Validação do algoritmo de redução escolhido (params.cmake → MOD_REDUCTION_ALG).
#if MOD_REDUCTION_ALG == MOD_RED_BURNIKEL_ZIEGLER
#error "MOD_RED_BURNIKEL_ZIEGLER ainda nao implementado. Use MOD_RED_MONTGOMERY ou MOD_RED_BARRETT em params.cmake."
#elif MOD_REDUCTION_ALG != MOD_RED_MONTGOMERY && MOD_REDUCTION_ALG != MOD_RED_BARRETT
#error "MOD_REDUCTION_ALG invalido. Valores: MOD_RED_MONTGOMERY | MOD_RED_BARRETT | MOD_RED_BURNIKEL_ZIEGLER."
#endif

// Headroom de limbs do contexto NTT. Barrett multiplica operandos de até
// (n_limbs+1) limbs (A1·μ), exigindo padded >= 2(k+1)-1; +1 limb garante isso
// mesmo quando 2k já é potência de dois. Montgomery usa operandos de k limbs.
#if MOD_REDUCTION_ALG == MOD_RED_BARRETT
#define MOD_NTT_EXTRA 1
#else
#define MOD_NTT_EXTRA 0
#endif

// Conversão entre inteiro normal e a "forma de trabalho" do backend de redução
// (definidas em reduce_montgomery.cu / reduce_barrett.cu). Montgomery: x·R^{±1} mod N;
// Barrett: resíduo plano x mod N. res e x são mpz_t; N o módulo; n_limbs a largura.
void mod_residue_forward(mpz_t res, const mpz_t x, const mpz_t N, int n_limbs);
void mod_residue_backward(mpz_t res, const mpz_t x, const mpz_t N, int n_limbs);

struct BatchModCtx
{
    int n_limbs, n_batch, padded, n_sum;
    BigIntNTTBatch ntt;

    // Per-candidate data, [n_batch * n_limbs]
    Data64 *d_N = nullptr;
    Data64 *d_Nprime = nullptr;

    // Pre-computado NTT(N) e NTT(N'), [n_batch * padded] — so leitura no hot path
    Data64 *d_ntt_N = nullptr;
    Data64 *d_ntt_Nprime = nullptr; // só MOD_RED_MONTGOMERY

#if MOD_REDUCTION_ALG == MOD_RED_BARRETT
    // Parâmetro de Barrett POR-CANDIDATO: bar_k[i] = nº de limbs "tight" de N_i — o
    // índice do limb mais significativo não-nulo + 1 (b^{k-1} <= N_i < b^{k}).
    // limbs_for_digits() aloca +4 limbs de folga e os candidatos esparsos diferem
    // entre si em alguns limbs do topo, então bar_k varia por candidato.
    // A largura dos buffers é uniforme: bar_W1 = max_i bar_k[i] + 1.
    int bar_W1 = 0;            // max(bar_k) + 1 (largura de A1, μ e q̂)
    int *d_bar_k = nullptr;    // [n_batch] bar_k por candidato (device)
    // μ_i = floor(b^{2·bar_k_i}/N_i) por candidato, já transformado (NTT(μ)).
    Data64 *d_ntt_mu = nullptr; // [n_batch * padded]
    // Scratch da redução de Barrett (tempos de vida disjuntos ⇒ reaproveitados):
    //   d_bar_w1   [n_batch * bar_W1] — A1 = T>>(k-1), depois q̂, depois resíduo r.
    //   d_bar_prod [n_batch * n_sum]  — produto intermediário: A1·μ, depois q̂·N.
    Data64 *d_bar_w1 = nullptr;
    Data64 *d_bar_prod = nullptr;
#endif

    // Valores de referencia na forma de trabalho — para check sem GMP.
    // Montgomery: to_mont(·). Barrett: resíduo plano (1 e N-1). [n_batch*n_limbs]
    Data64 *d_one_res = nullptr; // forma de trabalho de 1   por candidato
    Data64 *d_Nm1_res = nullptr; // forma de trabalho de N-1 por candidato

    // Buffers de trabalho
    Data64 *d_T = nullptr; // [n_batch * n_sum]
    Data64 *d_m = nullptr; // [n_batch * padded]  (NTT workspace de m)

    // Buffers para cond_sub tileado [n_batch * n_cs_tiles]
    int n_cs_tiles = 0;
    int *d_cs_tile_cmp = nullptr;    // cmp por tile: 1, -1, 0
    int *d_cs_tile_bstate = nullptr; // estado G/P/K de borrow por tile
    int *d_cs_tile_bin = nullptr;    // borrow_in resolvido por tile

    // Acumuladores de tempo por seção de kernel (ms, via CUDA events).
    // Cada campo corresponde a exatamente um par TSTART/TSTOP.
    // Um estágio cronometrado: tempo acumulado (ms) e nº de chamadas.
    struct Stage
    {
        float ms = 0;
        long long calls = 0;
        Stage operator+(const Stage &o) const { return {ms + o.ms, calls + o.calls}; }
    };

    struct PerfInner
    {
        // NTT: mul: load_padded(A+B)+NTT(A+B) | sq: load_padded(A)+NTT(A)
        // Schoolbook: schoolbook_mul / schoolbook_sq
        Stage ntt_input;
        // NTT: mul: pmul_batch | sq: psq_batch
        Stage pmul;
        // NTT: INTT após pmul/psq
        Stage intt_product;
        // carry_intra + carry_inter sobre T (resultado do produto/quadrado)
        Stage carry_product;
        // reduce: extract_low(T) + fwd_A — prepara T_low para multiplicar por N'
        Stage red_ntt_tlow;
        // reduce: pmul_ext(N') — multiplicação pontual T_low * N'
        Stage red_pmul_np;
        // reduce: INTT após pmul(N')
        Stage red_intt_np;
        // reduce: carry_intra + carry_inter sobre m
        Stage red_carry_m;
        // reduce: load_padded(m) + NTT(m) — transforma m para multiplicar por N
        Stage red_ntt_m;
        // reduce: pmul_ext(N) — multiplicação pontual m * N
        Stage red_pmul_n;
        // reduce: INTT após pmul(N)
        Stage red_intt_n;
        // reduce: vadd_from_raw — T += mN (soma antes do carry)
        Stage red_vadd;
        // reduce: carry_intra + carry_inter sobre T após vadd
        Stage red_carry_add;
        // reduce: shift_right(T, n_limbs) — resultado final = T[n_limbs..]
        Stage red_shift;
        // cs_phase1 + cs_resolve + cs_apply — subtração condicional mod N
        Stage cond_sub;

#if MOD_REDUCTION_ALG == MOD_RED_BARRETT
        // Etapas exclusivas do Barrett (reduce_barrett.cu).
        Stage bar_shift;   // shift_right_var_batch (A1 e q̂), acumulado
        Stage bar_sub;     // r = T − qn (subtração tileada incondicional)
        Stage bar_condsub; // r −= N, até 2 subtrações condicionais
        Stage bar_copy;    // copia r[0..n_limbs) → out
#endif

        // Soma campo a campo — usado para o resumo combinado (mul + sq).
        PerfInner operator+(const PerfInner &o) const
        {
            PerfInner r;
            r.ntt_input = ntt_input + o.ntt_input;
            r.pmul = pmul + o.pmul;
            r.intt_product = intt_product + o.intt_product;
            r.carry_product = carry_product + o.carry_product;
            r.red_ntt_tlow = red_ntt_tlow + o.red_ntt_tlow;
            r.red_pmul_np = red_pmul_np + o.red_pmul_np;
            r.red_intt_np = red_intt_np + o.red_intt_np;
            r.red_carry_m = red_carry_m + o.red_carry_m;
            r.red_ntt_m = red_ntt_m + o.red_ntt_m;
            r.red_pmul_n = red_pmul_n + o.red_pmul_n;
            r.red_intt_n = red_intt_n + o.red_intt_n;
            r.red_vadd = red_vadd + o.red_vadd;
            r.red_carry_add = red_carry_add + o.red_carry_add;
            r.red_shift = red_shift + o.red_shift;
            r.cond_sub = cond_sub + o.cond_sub;
#if MOD_REDUCTION_ALG == MOD_RED_BARRETT
            r.bar_shift = bar_shift + o.bar_shift;
            r.bar_sub = bar_sub + o.bar_sub;
            r.bar_condsub = bar_condsub + o.bar_condsub;
            r.bar_copy = bar_copy + o.bar_copy;
#endif
            return r;
        }

    } perf_mul, perf_sq;

    // Aponta para o acumulador do contexto atual (modmul_batch vs modsq_batch).
    PerfInner *perf_cur = &perf_mul;

    // Fase de host fornecida pelo chamador (ex.: setup, tabela, memcpy). Entra na
    // árvore como folha sintética sob o grupo "setup / host".
    struct HostPhase
    {
        const char *name;
        float ms;
        std::string note; // anotação opcional (ex.: "(17.5 GB/s)")
    };

    // Imprime UMA árvore: root TOTAL → {mont_mul, mont_sq, setup/host, others}.
    // app_total_ms = tempo total da aplicação; o nó "others (overhead)" recebe a
    // diferença para os tempos medidos (kernel launch, loop, gaps). host = fases de
    // host (setup/tabela/memcpy/...) agrupadas sob "setup / host".
    // No fim, a visão cross-cutting por tipo de kernel (acumulada mul + sq).
    void print_perf(double app_total_ms = 0.0,
                    const std::vector<HostPhase> &host = {}) const; // ver mod_perf.cu

    // Liga/desliga coleta de tempos. Quando false, TSTART/TSTOP e perf_flush
    // viram no-op — zero overhead de cudaEventRecord no hot path.
    bool perf_enabled = false;

    // Ring de eventos para profiling sem sync no hot path.
    // TSTART/TSTOP gravam eventos sem bloquear; perf_flush() sincroniza UMA vez
    // no final de cada modmul_batch/modsq_batch e acumula todos os tempos.
    static constexpr int PERF_RING = 32;
    cudaEvent_t ev_ring[PERF_RING + 1] = {};
    float *acc_ring[PERF_RING] = {};
    int ring_cur = 0;

    int device_id = 0; // GPU utilizada

    // Construtor a partir de limbs pré-computados.
    // device_id: índice da GPU (0 por padrão; use cudaGetDeviceCount para listar).
    // N_all: vetor flat [n_batch * n_limbs], little-endian 16-bit limbs.
    explicit BatchModCtx(const std::vector<uint64_t> &N_all, int n_limbs_, int n_batch_,
                          int device_id_ = 0);

    // Construtor de conveniência: aceita os números diretamente como mpz_t.
    // Calcula n_limbs automaticamente a partir do maior número do vetor.
    explicit BatchModCtx(const std::vector<mpz_t *> &numbers, int device_id_ = 0);
    ~BatchModCtx();

    // x_all (host, n_batch * n_limbs) -> forma Montgomery (host)
    void to_residue_batch(const std::vector<uint64_t> &x_all,
                       std::vector<uint64_t> &out_all) const;

    // d_x (GPU, forma Montgomery) -> valores normais (host)
    void from_residue_batch(const Data64 *d_x, std::vector<uint64_t> &out_all) const;

    // Verifica resultados no GPU: para cada candidato, r_mont == 1_mont ou (N-1)_mont?
    // d_passed[t] = 1 se passou, 0 se composto. n_total elementos.
    void check_passed(const Data64 *d_r_mont, uint8_t *d_passed, cudaStream_t s = 0) const;

    // d_out = mont_mul(d_A, d_B) para todos os n_batch candidatos
    void modmul_batch(const Data64 *d_A, const Data64 *d_B, Data64 *d_out,
                        cudaStream_t s = 0);
    // d_out = mont_sq(d_A) para todos os n_batch candidatos
    void modsq_batch(const Data64 *d_A, Data64 *d_out, cudaStream_t s = 0);

    // Apenas NTT(A)*NTT(B) + INTT — sem REDC. Mede custo puro da multiplicação.
    void mul_no_redc_batch(const Data64 *d_A, const Data64 *d_B, Data64 *d_out,
                           cudaStream_t s = 0);
    // Apenas NTT(A)^2 + INTT — sem REDC.
    void sq_no_redc_batch(const Data64 *d_A, Data64 *d_out, cudaStream_t s = 0);

    BatchModCtx(const BatchModCtx &) = delete;
    BatchModCtx &operator=(const BatchModCtx &) = delete;

private:
    // Pré-computa e aloca as estruturas específicas do backend de redução
    // (N'/NTT(N') p/ Montgomery; bar_k/μ/NTT(μ) p/ Barrett; tiles de cond_sub).
    // Definido em reduce_montgomery.cu / reduce_barrett.cu. Chamado no construtor.
    void precompute_reduction(const std::vector<uint64_t> &N_all);
    // Libera o que precompute_reduction alocou. Chamado no destrutor.
    void free_reduction();
    // Reduz d_T (produto em [n_batch*n_sum]) → d_out na forma de trabalho.
    // Implementado em reduce_montgomery.cu (REDC) ou reduce_barrett.cu (Barrett).
    void reduce_batch(Data64 *d_out, cudaStream_t s);
    // Subtração condicional mod N (só Montgomery; Barrett finaliza no próprio kernel).
    void cond_sub_batch(Data64 *d_x, cudaStream_t s);
    // Sincroniza o último evento do ring e acumula todos os tempos pendentes.
    // Chamado UMA VEZ no final de cada modmul_batch / modsq_batch.
    void perf_flush(cudaStream_t s);
};
