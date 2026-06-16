// perf/perf_node.cuh — Árvore de perfil montada dinamicamente em runtime.
//
// Cada grupo/operação chama branch() para criar filhos com nomes pré-declarados,
// acessados depois por índice (child(int)). Sem campos fixos por algoritmo.
// O report simplesmente caminha a árvore — não depende de nenhuma struct externa.
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <initializer_list>
#include <cstdio>
#include "helpers/time_format.h"

class PerfNode
{
public:
    std::string name;
    double ms = 0.0;       // tempo próprio (folhas cronometradas)
    long long calls = 0;   // nº de chamadas cronometradas
    std::string note;      // anotação opcional (ex.: banda GB/s)
    std::vector<std::unique_ptr<PerfNode>> children;

    explicit PerfNode(std::string n) : name(std::move(n)) {}

    // Cria e anexa um filho (opcionalmente com sub-filhos pré-declarados).
    // O ponteiro retornado é estável (armazenado em unique_ptr).
    PerfNode *branch(const std::string &n,
                     std::initializer_list<std::string> kids = {})
    {
        children.push_back(std::make_unique<PerfNode>(n));
        PerfNode *p = children.back().get();
        for (auto &k : kids)
            p->children.push_back(std::make_unique<PerfNode>(k));
        return p;
    }

    // Acesso por índice (estável após construção da árvore).
    PerfNode *child(int i) { return children[(size_t)i].get(); }
    const PerfNode *child(int i) const { return children[(size_t)i].get(); }

    // Tempo exibido: próprio se folha; soma dos filhos se grupo.
    double total_ms() const
    {
        if (children.empty())
            return ms;
        double t = 0.0;
        for (auto &c : children)
            t += c->total_ms();
        return t;
    }

    bool is_leaf() const { return children.empty(); }
};

// ── Impressão da árvore ───────────────────────────────────────────────────────
namespace perf_detail
{
    // Largura visível (code points UTF-8) para alinhar rótulos com ├─│.
    inline std::string padlbl(const std::string &s, int w)
    {
        int cols = 0;
        for (unsigned char c : s)
            if ((c & 0xC0) != 0x80)
                cols++;
        std::string r = s;
        for (int i = cols; i < w; i++)
            r += ' ';
        return r;
    }

    inline void rec(const PerfNode *n, const std::string &prefix, bool last,
                    double root_ms, int lbl_w)
    {
        std::string label = padlbl(prefix + (last ? "└─ " : "├─ ") + n->name, lbl_w);
        double ms = n->total_ms();
        double pct = root_ms > 0 ? ms * 100.0 / root_ms : 0.0;
        const char *note = n->note.empty() ? "" : n->note.c_str();
        if (n->is_leaf())
            printf("  %s %12s  %5.1f%%  %12s/call %s\n",
                   label.c_str(), fmt_time_ms((float)ms).c_str(), pct,
                   fmt_time_ms((float)(n->calls > 0 ? ms / n->calls : 0.0)).c_str(), note);
        else
            printf("  %s %12s  %5.1f%% %s\n",
                   label.c_str(), fmt_time_ms((float)ms).c_str(), pct, note);

        // Ordena filhos por tempo decrescente.
        std::vector<const PerfNode *> ks;
        for (auto &k : n->children)
            ks.push_back(k.get());
        for (size_t i = 0; i + 1 < ks.size(); i++)
            for (size_t j = 0; j + 1 < ks.size() - i; j++)
                if (ks[j]->total_ms() < ks[j + 1]->total_ms())
                    std::swap(ks[j], ks[j + 1]);
        std::string cp = prefix + (last ? "   " : "│  ");
        for (size_t i = 0; i < ks.size(); i++)
            rec(ks[i], cp, i + 1 == ks.size(), root_ms, lbl_w);
    }
} // namespace perf_detail

// Imprime a árvore inteira a partir de `root` (root = 100%).
inline void print_perf_tree(const PerfNode &root, int lbl_w = 36)
{
    double root_ms = root.total_ms();
    printf("  %s %12s  %5.1f%%\n",
           perf_detail::padlbl(root.name, lbl_w).c_str(),
           fmt_time_ms((float)root_ms).c_str(), 100.0);
    std::vector<const PerfNode *> ks;
    for (auto &k : root.children)
        ks.push_back(k.get());
    for (size_t i = 0; i + 1 < ks.size(); i++)
        for (size_t j = 0; j + 1 < ks.size() - i; j++)
            if (ks[j]->total_ms() < ks[j + 1]->total_ms())
                std::swap(ks[j], ks[j + 1]);
    for (size_t i = 0; i < ks.size(); i++)
        perf_detail::rec(ks[i], "", i + 1 == ks.size(), root_ms, lbl_w);
}
