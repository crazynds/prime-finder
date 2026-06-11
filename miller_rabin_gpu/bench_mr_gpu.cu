// bench_mr_gpu.cu
// Uso: ./bench_mr_gpu [--test] <candidatos.txt>
// Formato: uma linha por candidato: a b c k j
//
// Fase 1: testa N. Fase 2: testa revN dos sobreviventes.
// --test: executa testes de corretude (GMP) antes do benchmark.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <string>
#include <cuda_runtime.h>

#include "candidate.cuh"
#include "miller_rabin_runner.cuh"
#include "correctness_tests.cuh"
#include "bench_ops.cuh"

using hrc = std::chrono::high_resolution_clock;

static constexpr int BATCH_SIZE = MR_BATCH_SIZE;

// Monta arrays flat [bsz * n_limbs] para um batch de NumberCandidates
static void pack_batch(
        const std::vector<NumberCandidate*>& cands,
        int n_limbs,
        std::vector<uint64_t>& N_out,
        std::vector<uint64_t>& Nm1_out,
        std::vector<uint64_t>& d_out)
{
    int bsz = (int)cands.size();
    N_out.assign((size_t)bsz * n_limbs, 0);
    Nm1_out.assign((size_t)bsz * n_limbs, 0);
    d_out.assign((size_t)bsz * n_limbs, 0);
    for (int i = 0; i < bsz; i++) {
        std::copy(cands[i]->N_lims.begin(),   cands[i]->N_lims.end(),   N_out.begin()   + i*n_limbs);
        std::copy(cands[i]->Nm1_lims.begin(), cands[i]->Nm1_lims.end(), Nm1_out.begin() + i*n_limbs);
        std::copy(cands[i]->d_lims.begin(),   cands[i]->d_lims.end(),   d_out.begin()   + i*n_limbs);
    }
}

// Executa os witnesses sobre um batch de NumberCandidates, escolhendo a versão
// otimizada (s=1) ou geral conforme o s do primeiro candidato do batch.
static std::vector<bool> test_batch(
        const std::vector<NumberCandidate*>& cands,
        int n_limbs,
        const char* label,
        bool show_report,
        bool show_progress)
{
    std::vector<uint64_t> N_batch, Nm1_batch, d_batch;
    pack_batch(cands, n_limbs, N_batch, Nm1_batch, d_batch);

    int bsz = (int)cands.size();
    int s   = cands[0]->s;

    BatchMontCtx mont(N_batch, n_limbs, bsz);
    if (s == 1)
        return gpu_miller_rabin_s1(mont, d_batch, Nm1_batch, bsz, DEFAULT_WITNESSES, label, show_report, show_progress);
    else
        return gpu_miller_rabin(mont, d_batch, Nm1_batch, s, bsz, DEFAULT_WITNESSES, label, show_report, show_progress);
}

int main(int argc, char* argv[])
{
    bool run_tests      = false;
    bool show_report    = false;
    bool show_progress  = false;
    bool run_bench      = false;
    bool run_bench_long = false;
    bool show_config    = false;
    const char* input_file = nullptr;

    for (int i = 1; i < argc; i++) {
        if      (std::string(argv[i]) == "--test")           run_tests      = true;
        else if (std::string(argv[i]) == "--report")         show_report    = true;
        else if (std::string(argv[i]) == "--progress")       show_progress  = true;
        else if (std::string(argv[i]) == "--bench-ops")      run_bench      = true;
        else if (std::string(argv[i]) == "--bench-ops-long") run_bench_long = true;
        else if (std::string(argv[i]) == "--config")         show_config    = true;
        else if (!input_file)                                input_file     = argv[i];
    }

    if (show_config) {
#if   CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE
        const char* carry_alg = "SINGLE_TILE";
#elif CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
        const char* carry_alg = "MULTI_TILE";
#else
        const char* carry_alg = "SEQUENTIAL";
#endif
#if   MONT_MUL_ALG == MONT_MUL_ALG_NTT
        const char* mul_alg = "NTT";
#else
        const char* mul_alg = "SCHOOLBOOK";
#endif
        printf("╔══════════════════════════════════════════════════╗\n");
        printf("║  Configuração de build                           ║\n");
        printf("╚══════════════════════════════════════════════════╝\n");
        printf("  window_bits       %d\n",   MR_WINDOW_BITS);
        printf("  batch_size        %d\n",   MR_BATCH_SIZE);
        printf("  mont_mul_alg      %s\n",   mul_alg);
        printf("  carry_norm_alg    %s\n",   carry_alg);
        printf("  carry_tile        %d\n",   MR_CARRY_TILE);
        printf("  thr_load          %d\n",   MR_THR_LOAD);
        printf("  thr_pmul          %d\n",   MR_THR_PMUL);
        printf("  thr_reduce        %d\n",   MR_THR_REDUCE);
        printf("  thr_select_win    %d\n",   MR_THR_SELECT_WIN);
        printf("  thr_check         %d\n",   MR_THR_CHECK);
        printf("  cs_tile           %d\n",   MR_CS_TILE);
        printf("  progress_interval %d ms\n", MR_PROGRESS_INTERVAL_MS);
#ifdef MR_ADVANCED_MONITOR
        printf("  advanced_monitor  ON\n");
#else
        printf("  advanced_monitor  OFF\n");
#endif
        printf("\n");
    }

    if (run_bench || run_bench_long) {
        run_bench_ops(run_bench_long);
        return 0;
    }

    if (!input_file) {
        fprintf(stderr, "Uso: %s [--test] [--report] [--progress] [--config] [--bench-ops] [--bench-ops-long] <candidatos.txt>\n", argv[0]);
        return 1;
    }

    std::vector<SparseCandidate> cands;
    {
        std::ifstream fin(input_file);
        if (!fin) { fprintf(stderr, "Erro ao abrir %s\n", input_file); return 1; }
        long long a, b, c; int k, j;
        while (fin >> a >> b >> c >> k >> j)
            cands.push_back({a, b, c, k, j});
    }
    if (cands.empty()) { fprintf(stderr, "Nenhum candidato.\n"); return 1; }

    int n_cands = (int)cands.size();
    int n_limbs = limbs_for_digits((int)cands[0].a + 4);

    printf("Candidatos: %d  batch_size: %d  witnesses: %d\n",
           n_cands, BATCH_SIZE, (int)DEFAULT_WITNESSES.size());
    printf("Limbs 16-bit: %d  NTT padded: %d\n\n", n_limbs, next_pow2_ntt(2*n_limbs));

    printf("Construindo N/revN (GMP)...\n"); fflush(stdout);
    for (auto& c : cands) c.build(n_limbs);

    if (run_tests) {
        int bsz_test = std::min(n_cands, BATCH_SIZE);
        std::vector<NumberCandidate*> test_cands;
        for (int i = 0; i < bsz_test; i++) test_cands.push_back(&cands[i].N_cand);
        std::vector<uint64_t> N_test, Nm1_test, d_test;
        pack_batch(test_cands, n_limbs, N_test, Nm1_test, d_test);
        BatchMontCtx mont_test(N_test, n_limbs, bsz_test);
        run_correctness_tests(mont_test, N_test);
        run_known_prime_tests();
        run_general_s_prime_tests();
        run_s1_nextprime_tests();
    }

    auto t_global = hrc::now();
    std::vector<int> primes_idx;
    int n_batches = (n_cands + BATCH_SIZE - 1) / BATCH_SIZE;

    for (int b = 0; b < n_batches; b++) {
        int bstart = b * BATCH_SIZE;
        int bend   = std::min(bstart + BATCH_SIZE, n_cands);
        int bsz    = bend - bstart;

        printf("\n=== Batch %d/%d  (candidatos %d..%d) ===\n",
               b+1, n_batches, bstart, bend-1);

        // Fase 1: testa N
        std::vector<NumberCandidate*> N_cands;
        for (int i = 0; i < bsz; i++) N_cands.push_back(&cands[bstart+i].N_cand);

        printf("--- Fase 1: testando N (%d candidatos) ---\n", bsz); fflush(stdout);
        auto t0 = hrc::now();
        auto alive_N = test_batch(N_cands, n_limbs, "N", show_report, show_progress);
        printf("Fase 1: %.2fs\n",
               std::chrono::duration_cast<std::chrono::milliseconds>(hrc::now()-t0).count()/1000.0);

        std::vector<int> survivors;
        for (int i = 0; i < bsz; i++) if (alive_N[i]) survivors.push_back(i);
        if (survivors.empty()) { printf("Nenhum sobreviveu ao teste de N.\n"); continue; }
        printf("%d/%d sobreviveram ao teste de N.\n", (int)survivors.size(), bsz);

        // Fase 2: testa revN dos sobreviventes
        std::vector<NumberCandidate*> revN_cands;
        for (int si : survivors) revN_cands.push_back(&cands[bstart+si].revN_cand);

        printf("--- Fase 2: testando revN (%d candidatos) ---\n", (int)survivors.size()); fflush(stdout);
        t0 = hrc::now();
        auto alive_revN = test_batch(revN_cands, n_limbs, "revN", show_report, show_progress);
        printf("Fase 2: %.2fs\n",
               std::chrono::duration_cast<std::chrono::milliseconds>(hrc::now()-t0).count()/1000.0);

        for (int si = 0; si < (int)survivors.size(); si++)
            if (alive_revN[si]) primes_idx.push_back(bstart + survivors[si]);
    }

    double total = std::chrono::duration_cast<std::chrono::milliseconds>(
                       hrc::now()-t_global).count()/1000.0;

    printf("\n=== Resultado ===\n");
    if (primes_idx.empty()) {
        printf("  Nenhum candidato sobreviveu.\n");
    } else {
        for (int idx : primes_idx) {
            auto& c = cands[idx];
            printf("  PRIMO: 10^%lld - %d*10^%lld - %d*10^%lld - 1  (e revN)\n",
                   c.a, c.k, c.b, c.j, c.c);
        }
    }
    printf("Tempo total: %.2fs\n", total);
    return 0;
}
