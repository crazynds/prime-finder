#pragma once
// batch_mod_ctx.cuh — contexto de aritmética modular batched (Montgomery / Barrett).
//
// Layout: todos os arrays [n_batch * stride], batch_i = candidato.
// NTT chamada uma unica vez com batch=n_batch.

#include "config.h"
#include "ops/mul/multiplier.cuh"
#include "helpers/time_format.h"
#include "perf/perf_node.cuh"
#include "perf/perf_timer.cuh"
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include <memory>
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
// (definidas em reductions/montgomery.cu / reductions/barrett.cu). Montgomery: x·R^{±1} mod N;
// Barrett: resíduo plano x mod N. res e x são mpz_t; N o módulo; n_limbs a largura.
void mod_residue_forward(mpz_t res, const mpz_t x, const mpz_t N, int n_limbs);
void mod_residue_backward(mpz_t res, const mpz_t x, const mpz_t N, int n_limbs);

// ── Índices da árvore de perfil ───────────────────────────────────────────────
// Usados por batch_mod_ctx.cu e pelos arquivos de redução para navegar perf_cur.
// perf_cur é a raiz do contexto (mul ou sq); seus filhos:
//   child(PERF_PROD) = nó "produto"   (4 folhas: NTT, pmul, INTT, carry)
//   child(PERF_RED)  = nó "reducao"   (estrutura interna varia por algoritmo)
//   child(PERF_FIN)  = nó "finalize"  (Barrett: 3 folhas; Montgomery: folha cond_sub)
enum PerfCtxIdx  { PERF_PROD = 0, PERF_RED = 1, PERF_FIN = 2 };
// Filhos de PERF_PROD:
enum PerfProdIdx { PERF_PROD_NTT = 0, PERF_PROD_PMUL = 1, PERF_PROD_INTT = 2, PERF_PROD_CARRY = 3 };

struct BatchModCtx
{
    int n_limbs, n_batch, padded, n_sum;
    Multiplier ntt; // backend de multiplicação (compile-time: MUL_ALG)

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
    Data64 *d_m = nullptr; // [n_batch * padded]  (NTT workspace de m, só Montgomery)

    // Buffers do subtractor tileado (ops/sub) [n_batch * n_cs_tiles]
    int n_cs_tiles = 0;
    int *d_cs_tile_cmp = nullptr;    // cmp por tile: 1, -1, 0
    int *d_cs_tile_bstate = nullptr; // estado G/P/K de borrow por tile

    // ── Perfil dinâmico (árvore de PerfNode) ─────────────────────────────────
    // A árvore é montada no construtor via build_perf_nodes(). Cada sub-grupo
    // (produto, redução, finalize) é um ramo com filhos criados por branch().
    // Os filhos são acessados por índice (child(int)) — sem struct de ponteiros fixos.
    // O report simplesmente caminha a árvore; nenhum campo precisa ser hardcoded.
    PerfNode perf_root{"TOTAL"};
    PerfTimer timer;

    // Raízes dos contextos mul e sq (ramos de perf_root).
    // perf_cur aponta para mul ou sq durante cada chamada pública.
    PerfNode *perf_mul = nullptr;
    PerfNode *perf_sq  = nullptr;
    PerfNode *perf_cur = nullptr;

    // Fase de host fornecida pelo chamador (ex.: setup, tabela, memcpy). Entra na
    // árvore como folha sintética sob o grupo "setup / host".
    struct HostPhase
    {
        const char *name;
        float ms;
        std::string note; // anotação opcional (ex.: "(17.5 GB/s)")
    };

    // Caminha o grafo perf_root e imprime. app_total_ms preenche "others (overhead)";
    // host = fases de host agrupadas sob "setup / host". Ver helpers/mod_perf.cu.
    void print_perf(double app_total_ms = 0.0,
                    const std::vector<HostPhase> &host = {});

    // Liga/desliga coleta de tempos. Quando false, TSTART/TSTOP viram no-op.
    bool perf_enabled = false;

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
    // Pré-computa e aloca as estruturas específicas do backend de redução.
    void precompute_reduction(const std::vector<uint64_t> &N_all);
    // Libera o que precompute_reduction alocou.
    void free_reduction();
    // Reduz d_T (produto em [n_batch*n_sum]) → d_out na forma de trabalho.
    void reduce_batch(Data64 *d_out, cudaStream_t s);
    // Subtração condicional mod N (só Montgomery; Barrett finaliza no próprio kernel).
    void cond_sub_batch(Data64 *d_x, cudaStream_t s);
    // Sincroniza o último evento do ring e acumula todos os tempos pendentes.
    void perf_flush(cudaStream_t s);
    // Monta a subárvore de uma via (mul/sq) sob perf_root e retorna o ramo raiz.
    PerfNode *build_perf_nodes(const char *ctx_name);
};
