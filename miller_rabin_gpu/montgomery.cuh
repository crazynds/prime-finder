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

struct BatchMontCtx
{
    int n_limbs, n_batch, padded, n_sum;
    BigIntNTTBatch ntt;

    // Per-candidate data, [n_batch * n_limbs]
    Data64 *d_N = nullptr;
    Data64 *d_Nprime = nullptr;

    // Pre-computado NTT(N) e NTT(N'), [n_batch * padded] — so leitura no hot path
    Data64 *d_ntt_N = nullptr;
    Data64 *d_ntt_Nprime = nullptr;

    // Valores de referencia em forma Montgomery — para check sem GMP
    // [n_batch * n_limbs] cada
    Data64 *d_one_mont = nullptr; // to_mont(1)   por candidato
    Data64 *d_Nm1_mont = nullptr; // to_mont(N-1) por candidato

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
            return r;
        }

    } perf_mul, perf_sq;

    // Aponta para o acumulador do contexto atual (mont_mul_batch vs mont_sq_batch).
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
                    const std::vector<HostPhase> &host = {}) const
    {
        // Nó da árvore: folha com Stage (st), folha sintética (extra >= 0) ou grupo.
        struct Node
        {
            std::string name;
            const Stage *st = nullptr;
            float extra = -1.0f;
            std::string note;
            std::vector<Node> kids;
        };
        auto L = [](const char *n, const Stage *s)
        { Node x; x.name = n; x.st = s; return x; };
        auto G = [](const char *n, std::vector<Node> k)
        { Node x; x.name = n; x.kids = std::move(k); return x; };
        auto F = [](const char *n, float ms, std::string note = "")
        { Node x; x.name = n; x.extra = ms; x.note = std::move(note); return x; };

        // Subárvore de um contexto (mont_mul ou mont_sq) com os passos do algoritmo.
        auto ctx = [&](const char *name, const PerfInner &p) -> Node
        {
#if MONT_MUL_ALG == MONT_MUL_ALG_SCHOOLBOOK
            Node produto = G("multiplicacao (produto)", {L("schoolbook_mul/sq", &p.ntt_input), L("carry_product", &p.carry_product)});
#else
            Node produto = G("multiplicacao (produto)", {L("ntt_input", &p.ntt_input), L("pmul/psq", &p.pmul), L("intt_product", &p.intt_product), L("carry_product", &p.carry_product)});
#endif
            Node reducao = G("reducao Montgomery", {
                                                       G("Multiplicacao", {L("ntt_Tlow", &p.red_ntt_tlow), L("pmul_Np", &p.red_pmul_np), L("intt_Np", &p.red_intt_np), L("carry_m", &p.red_carry_m), L("ntt_m", &p.red_ntt_m), L("pmul_N", &p.red_pmul_n), L("intt_N", &p.red_intt_n)}),
                                                       G("Soma", {L("vadd", &p.red_vadd), L("carry_add", &p.red_carry_add)}),
                                                       L("shift_right", &p.red_shift),
                                                   });
            return G(name, {produto, reducao, L("cond_sub", &p.cond_sub)});
        };

        Node root = G("TOTAL", {ctx("mont_mul", perf_mul), ctx("mont_sq", perf_sq)});
        // Grupo das fases de host (setup, tabela, memcpy, ...).
        if (!host.empty())
        {
            std::vector<Node> hk;
            for (auto &h : host)
                hk.push_back(F(h.name, h.ms, h.note));
            root.kids.push_back(G("setup / host", std::move(hk)));
        }

        std::function<float(const Node &)> nodeMs = [&](const Node &n) -> float
        {
            if (n.st)
                return n.st->ms;
            if (n.extra >= 0.0f)
                return n.extra;
            float t = 0;
            for (auto &k : n.kids)
                t += nodeMs(k);
            return t;
        };

        // "others (overhead)" = total da aplicação - tudo que foi medido.
        if (app_total_ms > 0.0)
        {
            float measured = 0;
            for (auto &k : root.kids)
                measured += nodeMs(k);
            float others = (float)app_total_ms - measured;
            if (others > 0.0f)
                root.kids.push_back(F("others (overhead)", others));
        }

        float total = nodeMs(root);
        auto pct = [&](float v)
        { return total > 0 ? v * 100.0f / total : 0.0f; };
        auto avg = [](const Stage &s)
        { return s.calls > 0 ? (double)s.ms / (double)s.calls : 0.0; };

        // Largura VISÍVEL de um label: conta code points UTF-8 (cada ├─│ é 1 coluna
        // mas ocupa 3 bytes), não bytes — senão %-Ns desalinha nos níveis profundos.
        auto padlbl = [](const std::string &s, int w) -> std::string
        {
            int cols = 0;
            for (unsigned char c : s)
                if ((c & 0xC0) != 0x80) // ignora bytes de continuação UTF-8
                    cols++;
            std::string r = s;
            for (int i = cols; i < w; i++)
                r += ' ';
            return r;
        };
        constexpr int LBL_W = 36;

        // Impressão recursiva. Ordena filhos por tempo (== tempo/call, calls iguais).
        std::function<void(const Node &, const std::string &, bool)> rec =
            [&](const Node &n, const std::string &prefix, bool last)
        {
            std::string label = padlbl(prefix + (last ? "└─ " : "├─ ") + n.name, LBL_W);
            float ms = nodeMs(n);
            const char *note = n.note.empty() ? "" : n.note.c_str();
            if (n.st)
                printf("  %s %12s  %5.1f%%  %12s/call %s\n",
                       label.c_str(), fmt_time_ms(ms).c_str(), pct(ms),
                       fmt_time_ms(avg(*n.st)).c_str(), note);
            else if (n.extra >= 0.0f)
                printf("  %s %12s  %5.1f%% %s\n",
                       label.c_str(), fmt_time_ms(ms).c_str(), pct(ms), note);
            else
            {
                printf("  %s %12s  %5.1f%%\n", label.c_str(), fmt_time_ms(ms).c_str(), pct(ms));
                std::vector<const Node *> ks;
                for (auto &k : n.kids)
                    ks.push_back(&k);
                std::sort(ks.begin(), ks.end(), [&](const Node *a, const Node *b)
                          { return nodeMs(*a) > nodeMs(*b); });
                std::string cp = prefix + (last ? "   " : "│  ");
                for (size_t i = 0; i < ks.size(); i++)
                    rec(*ks[i], cp, i + 1 == ks.size());
            }
        };

        // Root impresso sem conector; filhos (mont_mul/mont_sq) ordenados por tempo.
        printf("  %s %12s  %5.1f%%\n", padlbl(root.name, LBL_W).c_str(), fmt_time_ms(total).c_str(), 100.0);
        std::vector<const Node *> ctxs;
        for (auto &k : root.kids)
            ctxs.push_back(&k);
        std::sort(ctxs.begin(), ctxs.end(), [&](const Node *a, const Node *b)
                  { return nodeMs(*a) > nodeMs(*b); });
        for (size_t i = 0; i < ctxs.size(); i++)
            rec(*ctxs[i], "", i + 1 == ctxs.size());

        // ── Visão cross-cutting por tipo de kernel (acumulada mul + sq) ──
        PerfInner all = perf_mul + perf_sq;
        auto sumc = [&](std::initializer_list<const Stage *> ss)
        {
            float t = 0;
            for (auto *s : ss)
                t += s->ms;
            return t;
        };
#if MONT_MUL_ALG == MONT_MUL_ALG_SCHOOLBOOK
        float ntt_t = sumc({&all.red_ntt_tlow, &all.red_intt_np, &all.red_ntt_m, &all.red_intt_n});
        float pw_t = sumc({&all.red_pmul_np, &all.red_pmul_n});
        float prod_t = all.ntt_input.ms; // convolução schoolbook
#else
        float ntt_t = sumc({&all.ntt_input, &all.intt_product, &all.red_ntt_tlow, &all.red_intt_np, &all.red_ntt_m, &all.red_intt_n});
        float pw_t = sumc({&all.pmul, &all.red_pmul_np, &all.red_pmul_n});
        float prod_t = 0;
#endif
        float carry_t = sumc({&all.carry_product, &all.red_carry_m, &all.red_carry_add});
        float add_t = all.red_vadd.ms;
        float shift_t = all.red_shift.ms;
        float cs_t = all.cond_sub.ms;
        auto crow = [&](const char *name, float ms)
        {
            if (ms > 0)
                printf("     %-20s %12s  %5.1f%%\n", name, fmt_time_ms(ms).c_str(), pct(ms));
        };
        printf("\n  por tipo de kernel (acumulado):\n");
        crow("NTT/INTT", ntt_t);
        crow("pointwise (pmul)", pw_t);
        crow("carry", carry_t);
        crow("soma (vadd)", add_t);
        crow("shift", shift_t);
        crow("cond_sub", cs_t);
        crow("produto direto", prod_t);
    }

    // Liga/desliga coleta de tempos. Quando false, TSTART/TSTOP e perf_flush
    // viram no-op — zero overhead de cudaEventRecord no hot path.
    bool perf_enabled = false;

    // Ring de eventos para profiling sem sync no hot path.
    // TSTART/TSTOP gravam eventos sem bloquear; perf_flush() sincroniza UMA vez
    // no final de cada mont_mul_batch/mont_sq_batch e acumula todos os tempos.
    static constexpr int PERF_RING = 32;
    cudaEvent_t ev_ring[PERF_RING + 1] = {};
    float *acc_ring[PERF_RING] = {};
    int ring_cur = 0;

    int device_id = 0; // GPU utilizada

    // Construtor a partir de limbs pré-computados.
    // device_id: índice da GPU (0 por padrão; use cudaGetDeviceCount para listar).
    // N_all: vetor flat [n_batch * n_limbs], little-endian 16-bit limbs.
    explicit BatchMontCtx(const std::vector<uint64_t> &N_all, int n_limbs_, int n_batch_,
                          int device_id_ = 0);

    // Construtor de conveniência: aceita os números diretamente como mpz_t.
    // Calcula n_limbs automaticamente a partir do maior número do vetor.
    explicit BatchMontCtx(const std::vector<mpz_t *> &numbers, int device_id_ = 0);
    ~BatchMontCtx();

    // x_all (host, n_batch * n_limbs) -> forma Montgomery (host)
    void to_mont_batch(const std::vector<uint64_t> &x_all,
                       std::vector<uint64_t> &out_all) const;

    // d_x (GPU, forma Montgomery) -> valores normais (host)
    void from_mont_batch(const Data64 *d_x, std::vector<uint64_t> &out_all) const;

    // Verifica resultados no GPU: para cada candidato, r_mont == 1_mont ou (N-1)_mont?
    // d_passed[t] = 1 se passou, 0 se composto. n_total elementos.
    void check_passed(const Data64 *d_r_mont, uint8_t *d_passed, cudaStream_t s = 0) const;

    // d_out = mont_mul(d_A, d_B) para todos os n_batch candidatos
    void mont_mul_batch(const Data64 *d_A, const Data64 *d_B, Data64 *d_out,
                        cudaStream_t s = 0);
    // d_out = mont_sq(d_A) para todos os n_batch candidatos
    void mont_sq_batch(const Data64 *d_A, Data64 *d_out, cudaStream_t s = 0);

    // Apenas NTT(A)*NTT(B) + INTT — sem REDC. Mede custo puro da multiplicação.
    void mul_no_redc_batch(const Data64 *d_A, const Data64 *d_B, Data64 *d_out,
                           cudaStream_t s = 0);
    // Apenas NTT(A)^2 + INTT — sem REDC.
    void sq_no_redc_batch(const Data64 *d_A, Data64 *d_out, cudaStream_t s = 0);

    BatchMontCtx(const BatchMontCtx &) = delete;
    BatchMontCtx &operator=(const BatchMontCtx &) = delete;

private:
    void reduce_batch(Data64 *d_out, cudaStream_t s);
    void cond_sub_batch(Data64 *d_x, cudaStream_t s);
    // Sincroniza o último evento do ring e acumula todos os tempos pendentes.
    // Chamado UMA VEZ no final de cada mont_mul_batch / mont_sq_batch.
    void perf_flush(cudaStream_t s);
};
