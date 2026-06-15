// mod_perf.cu — Impressão da árvore de perfil de tempo do BatchModCtx.
// Separado de batch_mod_ctx.cu por ser puramente de relatório (sem lógica de kernel).

#include "batch_mod_ctx.cuh"
#include "time_format.h"
#include <cstdio>
#include <algorithm>
#include <functional>
#include <vector>
#include <string>

// Imprime UMA árvore: root TOTAL → {mont_mul, mont_sq, setup/host, others}.
// app_total_ms = tempo total da aplicação; o nó "others (overhead)" recebe a
// diferença para os tempos medidos (kernel launch, loop, gaps). host = fases de
// host (setup/tabela/memcpy/...) agrupadas sob "setup / host".
// No fim, a visão cross-cutting por tipo de kernel (acumulada mul + sq).
void BatchModCtx::print_perf(double app_total_ms,
                             const std::vector<HostPhase> &host) const
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

    // Subárvore de um contexto (mul ou sq) com os passos do algoritmo. Os campos
    // de PerfInner são reaproveitados pelos dois redutores, mas com semântica
    // diferente — os rótulos abaixo seguem MOD_REDUCTION_ALG para não enganar.
    auto ctx = [&](const char *name, const PerfInner &p) -> Node
    {
#if MONT_MUL_ALG == MONT_MUL_ALG_SCHOOLBOOK
        Node produto = G("multiplicacao (produto)", {L("schoolbook_mul/sq", &p.ntt_input), L("carry_product", &p.carry_product)});
#else
        Node produto = G("multiplicacao (produto)", {L("ntt_input", &p.ntt_input), L("pmul/psq", &p.pmul), L("intt_product", &p.intt_product), L("carry_product", &p.carry_product)});
#endif
#if MOD_REDUCTION_ALG == MOD_RED_BARRETT
        // Barrett: red_* reaproveitados → q2 = A1·μ e qn = q̂·N; cond_sub = finalize.
        Node reducao = G("reducao Barrett", {
                                                L("shift (A1,q)", &p.bar_shift),
                                                G("q2 = A1.mu", {L("ntt(A1)", &p.red_ntt_tlow), L("pmul(mu)", &p.red_pmul_np), L("intt(q2)", &p.red_intt_np), L("carry(q2)", &p.red_carry_m)}),
                                                G("qn = q.N", {L("ntt(q)", &p.red_ntt_m), L("pmul(N)", &p.red_pmul_n), L("intt(qn)", &p.red_intt_n), L("carry(qn)", &p.red_carry_add)}),
                                            });
        Node finalize = G("barrett_finalize", {L("sub (T-qn)", &p.bar_sub), L("cond_sub N (2x)", &p.bar_condsub), L("copy_out", &p.bar_copy)});
        return G(name, {produto, reducao, finalize});
#else
        Node reducao = G("reducao Montgomery", {
                                                   G("Multiplicacao", {L("ntt_Tlow", &p.red_ntt_tlow), L("pmul_Np", &p.red_pmul_np), L("intt_Np", &p.red_intt_np), L("carry_m", &p.red_carry_m), L("ntt_m", &p.red_ntt_m), L("pmul_N", &p.red_pmul_n), L("intt_N", &p.red_intt_n)}),
                                                   G("Soma", {L("vadd", &p.red_vadd), L("carry_add", &p.red_carry_add)}),
                                                   L("shift_right", &p.red_shift),
                                               });
        return G(name, {produto, reducao, L("cond_sub", &p.cond_sub)});
#endif
    };

#if MOD_REDUCTION_ALG == MOD_RED_BARRETT
    Node root = G("TOTAL", {ctx("mul", perf_mul), ctx("sq", perf_sq)});
#else
    Node root = G("TOTAL", {ctx("mont_mul", perf_mul), ctx("mont_sq", perf_sq)});
#endif
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
#if MOD_REDUCTION_ALG == MOD_RED_BARRETT
    float add_t = all.bar_sub.ms;
    float shift_t = all.bar_condsub.ms;
    float cs_t = all.bar_copy.ms;
#else
    float add_t = all.red_vadd.ms;
    float shift_t = all.red_shift.ms;
    float cs_t = all.cond_sub.ms;
#endif
    auto crow = [&](const char *name, float ms)
    {
        if (ms > 0)
            printf("     %-20s %12s  %5.1f%%\n", name, fmt_time_ms(ms).c_str(), pct(ms));
    };
    printf("\n  por tipo de kernel (acumulado):\n");
    crow("NTT/INTT", ntt_t);
    crow("pointwise (pmul)", pw_t);
    crow("carry", carry_t);
#if MOD_REDUCTION_ALG == MOD_RED_BARRETT
    crow("shift (var)", all.bar_shift.ms);
    crow("finalize: sub (T-qn)", add_t);
    crow("finalize: cond_sub N", shift_t);
    crow("finalize: copy_out", cs_t);
#else
    crow("soma (vadd)", add_t);
    crow("shift", shift_t);
    crow("cond_sub", cs_t);
#endif
    crow("produto direto", prod_t);
}
