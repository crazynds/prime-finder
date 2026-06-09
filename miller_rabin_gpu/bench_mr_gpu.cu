// bench_mr_gpu.cu
// Uso: ./bench_mr_gpu <candidatos.txt>
// Formato: uma linha por candidato: a b c k j
//
// Todos os candidatos (N e revN) processados como um unico batch.
// Uma chamada NTT processa todos de uma vez. Sem streams multiplas.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>
#include <algorithm>
#include <gmp.h>
#include <cuda_runtime.h>
#include "montgomery.cuh"

#define CU(expr) \
    do { cudaError_t _e=(expr); if(_e!=cudaSuccess) \
        throw std::runtime_error(std::string("[CUDA] " #expr ": ")+cudaGetErrorString(_e)); \
    } while(0)

using hrc = std::chrono::high_resolution_clock;

// Tamanho da janela para exponenciação por janela deslizante.
// k=1 → 1 sq + 1 mul por bit (baseline)
// k=2 → 2 sq + 1 mul por 2 bits  (~25% menos operações)
// k=4 → 4 sq + 1 mul por 4 bits  (~37% menos operações)
// k=8 → 8 sq + 1 mul por 8 bits  (~44% menos operações, pré-computa 255 entradas)
static constexpr int WINDOW_BITS = 8;
static constexpr int WINDOW_SIZE = 1 << WINDOW_BITS;

static const uint32_t WITNESSES[] = {
    2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53
};
static const int N_WITNESSES = (int)(sizeof(WITNESSES)/sizeof(WITNESSES[0]));

// ── helpers GMP ───────────────────────────────────────────────────────────────

static void mpz_to_limbs_vec(uint64_t* out, int n, const mpz_t x)
{
    mpz_t tmp; mpz_init_set(tmp, x);
    for (int i = 0; i < n; i++) {
        out[i] = mpz_get_ui(tmp) & 0xFFFF;
        mpz_tdiv_q_2exp(tmp, tmp, 16);
    }
    mpz_clear(tmp);
}

static int ndig(int x) { return (x<10)?1:(x<100)?2:3; }
static int revd(int x,int l) { int r=0; for(int i=0;i<l;i++){r=r*10+(x%10);x/=10;} return r; }

// ── Candidato ─────────────────────────────────────────────────────────────────

struct Candidate {
    long long a, b, c;
    int k, j;

    // Limbs no host — pre-computados
    std::vector<uint64_t> N_lims, Nm1_lims, d_lims;
    std::vector<uint64_t> revN_lims, revNm1_lims, drev_lims;

    void build(int n_limbs)
    {
        N_lims.assign(n_limbs, 0);    Nm1_lims.assign(n_limbs, 0);   d_lims.assign(n_limbs, 0);
        revN_lims.assign(n_limbs, 0); revNm1_lims.assign(n_limbs, 0); drev_lims.assign(n_limbs, 0);

        mpz_t N, revN, tmp, Nm1, d, revNm1, drev;
        mpz_inits(N, revN, tmp, Nm1, d, revNm1, drev, nullptr);

        mpz_ui_pow_ui(N, 10, (unsigned long)a);
        mpz_ui_pow_ui(tmp, 10, (unsigned long)b); mpz_mul_ui(tmp,tmp,(unsigned long)k);
        mpz_sub(N,N,tmp);
        mpz_ui_pow_ui(tmp, 10, (unsigned long)c); mpz_mul_ui(tmp,tmp,(unsigned long)j);
        mpz_sub(N,N,tmp); mpz_sub_ui(N,N,1);

        int lk=ndig(k), lj=ndig(j);
        long long eb=a-b-lk, ec=a-c-lj;
        mpz_ui_pow_ui(revN, 10, (unsigned long)a);
        mpz_ui_pow_ui(tmp, 10, (unsigned long)eb); mpz_mul_ui(tmp,tmp,(unsigned long)revd(k,lk));
        mpz_sub(revN,revN,tmp);
        mpz_ui_pow_ui(tmp, 10, (unsigned long)ec); mpz_mul_ui(tmp,tmp,(unsigned long)revd(j,lj));
        mpz_sub(revN,revN,tmp); mpz_sub_ui(revN,revN,1);

        mpz_sub_ui(Nm1, N, 1);     mpz_tdiv_q_2exp(d,    Nm1,    1);
        mpz_sub_ui(revNm1, revN,1); mpz_tdiv_q_2exp(drev, revNm1, 1);

        mpz_to_limbs_vec(N_lims.data(),      n_limbs, N);
        mpz_to_limbs_vec(Nm1_lims.data(),    n_limbs, Nm1);
        mpz_to_limbs_vec(d_lims.data(),      n_limbs, d);
        mpz_to_limbs_vec(revN_lims.data(),   n_limbs, revN);
        mpz_to_limbs_vec(revNm1_lims.data(), n_limbs, revNm1);
        mpz_to_limbs_vec(drev_lims.data(),   n_limbs, drev);

        mpz_clears(N, revN, tmp, Nm1, d, revNm1, drev, nullptr);
    }
};

// Forward declaration para select_window_kernel (definido mais abaixo).
// Também usada em correctness_tests.cuh via #include.
__global__ static void select_window_kernel(
        Data64* __restrict__, const Data64* __restrict__, const Data64* __restrict__,
        int, int, int, int);

#include "correctness_tests.cuh"


// ── Kernel: seleciona table[w] para cada candidato dado janela de k bits ──────
// msb_pos: bit mais significativo da janela (inclusive).
// Para cada candidato extrai os k bits [msb_pos .. msb_pos-k+1] do expoente
// e copia d_table[w * n_total * n_limbs + ...] em d_out.
// Bits fora do range do expoente (< 0) são tratados como 0.

__global__ static void select_window_kernel(
        Data64* __restrict__       d_out,
        const Data64* __restrict__ d_table,   // [WINDOW_SIZE * n_total * n_limbs]
        const Data64* __restrict__ d_exp,
        int msb_pos, int k,
        int n_limbs, int n_total)
{
    int t = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_total || j >= n_limbs) return;

    // Extrai janela de k bits: bit msb_pos é o mais significativo
    int w = 0;
    for (int b = 0; b < k; b++) {
        int bp = msb_pos - b;
        if (bp >= 0) {
            int li  = bp / LIMB_BITS;
            int bit = bp % LIMB_BITS;
            if ((d_exp[(size_t)t * n_limbs + li] >> bit) & 1)
                w |= (1 << (k - 1 - b));
        }
    }

    d_out[(size_t)t * n_limbs + j] =
        d_table[(size_t)w * n_total * n_limbs + (size_t)t * n_limbs + j];
}

// ── Roda todos os witnesses no batch ─────────────────────────────────────────

static std::vector<bool> run_all_witnesses(
        BatchMontCtx& mont,
        const std::vector<uint64_t>& exp_all,
        const std::vector<uint64_t>& Nm1_all,
        int n_total,
        const char* label)
{
    int n = mont.n_limbs;
    size_t total_bytes = (size_t)n_total * n * sizeof(Data64);

    Data64 *d_r, *d_base, *d_scratch, *d_one, *d_cur_mul, *d_exp_dev;
    uint8_t* d_passed;
    CU(cudaMalloc(&d_r,       total_bytes));
    CU(cudaMalloc(&d_base,    total_bytes));
    CU(cudaMalloc(&d_scratch, total_bytes));
    CU(cudaMalloc(&d_one,     total_bytes));
    CU(cudaMalloc(&d_cur_mul, total_bytes));
    CU(cudaMalloc(&d_exp_dev, total_bytes));
    CU(cudaMalloc(&d_passed,  (size_t)n_total));
    CU(cudaMemcpy(d_exp_dev, exp_all.data(), total_bytes, cudaMemcpyHostToDevice));

    std::vector<uint8_t> passed_h(n_total);

    std::vector<uint64_t> one_all((size_t)n_total * n, 0);
    for (int t = 0; t < n_total; t++) one_all[t*n] = 1;
    std::vector<uint64_t> one_mont;
    mont.to_mont_batch(one_all, one_mont);
    CU(cudaMemcpy(d_one, one_mont.data(), total_bytes, cudaMemcpyHostToDevice));

    // Encontra o MSB real do expoente entre todos os candidatos
    int msb = n * LIMB_BITS - 1;
    while (msb > 0) {
        int li = msb/LIMB_BITS, bit = msb%LIMB_BITS;
        bool any = false;
        for (int t = 0; t < n_total && !any; t++)
            if ((exp_all[t*n + li] >> bit) & 1) any = true;
        if (any) break;
        msb--;
    }

    // Alinha o início para múltiplo de WINDOW_BITS:
    // start_win é o bit mais significativo da primeira janela
    int n_windows   = (msb / WINDOW_BITS + 1);
    int start_win   = n_windows * WINDOW_BITS - 1;  // MSB da 1a janela (pode estar acima do msb real)

    // Para cada janela: any_nonzero[wi] = true se algum candidato tem janela != 0
    std::vector<bool> any_nonzero(n_windows, false);
    for (int wi = 0; wi < n_windows; wi++) {
        int i = start_win - wi * WINDOW_BITS;  // MSB desta janela
        for (int t = 0; t < n_total && !any_nonzero[wi]; t++)
            for (int b = 0; b < WINDOW_BITS && !any_nonzero[wi]; b++) {
                int bp = i - b;
                if (bp >= 0 && bp <= msb) {
                    int li = bp/LIMB_BITS, bit = bp%LIMB_BITS;
                    if ((exp_all[t*n + li] >> bit) & 1)
                        any_nonzero[wi] = true;
                }
            }
    }

    const int thr = 256;
    dim3 grid_sel((unsigned)(n + thr-1)/thr, (unsigned)n_total);

    // ── Tabela de potências de base para janela de WINDOW_BITS bits ───────────
    // d_table[w * n_total * n_limbs] = base^w em forma Montgomery, para todo w em [0, WINDOW_SIZE)
    // Alocado fora do loop de witnesses para reutilizar o buffer
    Data64* d_table;
    CU(cudaMalloc(&d_table, (size_t)WINDOW_SIZE * total_bytes));

    // ── Acumuladores de tempo (GPU, via eventos CUDA) ─────────────────────────
    struct PerfCtrs {
        float sq_ms      = 0;   // mont_sq_batch (k squarings por janela)
        float mul_ms     = 0;   // select_window_kernel + mont_mul_batch
        float table_ms   = 0;   // pré-computação da tabela por witness
        float check_ms   = 0;   // check_passed + memcpy resultado
        float setup_ms   = 0;   // to_mont_batch + uploads por witness
        long  sq_calls   = 0;
        long  mul_calls  = 0;
    } perf;

    cudaEvent_t ev0, ev1;
    CU(cudaEventCreate(&ev0));
    CU(cudaEventCreate(&ev1));

    auto gpu_elapsed = [&]() {
        float ms = 0;
        CU(cudaEventSynchronize(ev1));
        CU(cudaEventElapsedTime(&ms, ev0, ev1));
        return ms;
    };

    // alive[t] = true enquanto nao for composto
    std::vector<bool> alive(n_total, true);

    for (int wi = 0; wi < N_WITNESSES; wi++) {
        int n_alive = 0;
        for (bool b : alive) if (b) n_alive++;
        if (n_alive == 0) break;

        printf("  [%s] Witness %-3u  vivos: %d\n", label, WITNESSES[wi], n_alive);
        fflush(stdout);

        // ── Setup do witness: base em forma Montgomery ────────────────────────
        CU(cudaEventRecord(ev0));
        std::vector<uint64_t> w_all((size_t)n_total * n, 0);
        for (int t = 0; t < n_total; t++) {
            w_all[t*n + 0] = WITNESSES[wi] & LIMB_MASK;
            w_all[t*n + 1] = (WITNESSES[wi] >> LIMB_BITS) & LIMB_MASK;
        }
        std::vector<uint64_t> base_mont;
        mont.to_mont_batch(w_all, base_mont);
        CU(cudaMemcpy(d_base, base_mont.data(), total_bytes, cudaMemcpyHostToDevice));
        CU(cudaMemcpy(d_r,    one_mont.data(),  total_bytes, cudaMemcpyHostToDevice));
        CU(cudaEventRecord(ev1));
        perf.setup_ms += gpu_elapsed();

        // ── Pré-computa tabela: table[0]=1, table[1]=base, ..., table[2^k-1]=base^(2^k-1) ──
        CU(cudaEventRecord(ev0));
        CU(cudaMemcpy(d_table,
                      d_one, total_bytes, cudaMemcpyDeviceToDevice));  // table[0] = 1
        CU(cudaMemcpy(d_table + (size_t)1 * n_total * n,
                      d_base, total_bytes, cudaMemcpyDeviceToDevice)); // table[1] = base
        for (int w = 2; w < WINDOW_SIZE; w++) {
            // table[w] = table[w-1] * base
            mont.mont_mul_batch(
                d_table + (size_t)(w-1) * n_total * n,
                d_base,
                d_table + (size_t)w     * n_total * n);
        }
        CU(cudaEventRecord(ev1));
        perf.table_ms += gpu_elapsed();

        auto t_start      = hrc::now();
        auto t_last_print = t_start;

        for (int win = 0; win < n_windows; win++) {
            int i = start_win - win * WINDOW_BITS;  // MSB desta janela

            // ── k squarings: r = r^(2^k) ─────────────────────────────────────
            // perf_flush() dentro de mont_sq_batch sincroniza a GPU uma vez por
            // call, então CPU wall clock é preciso aqui.
            auto t_sq0 = hrc::now();
            for (int sq = 0; sq < WINDOW_BITS; sq++) {
                mont.mont_sq_batch(d_r, d_scratch);
                std::swap(d_r, d_scratch);
            }
            perf.sq_ms    += std::chrono::duration<float, std::milli>(hrc::now() - t_sq0).count();
            perf.sq_calls += WINDOW_BITS;

            // ── Multiplicação por table[w] (skip se janela toda zero) ─────────
            if (any_nonzero[win]) {
                auto t_mul0 = hrc::now();
                select_window_kernel<<<grid_sel, thr>>>(
                    d_cur_mul, d_table, d_exp_dev, i, WINDOW_BITS, n, n_total);
                mont.mont_mul_batch(d_r, d_cur_mul, d_scratch);
                std::swap(d_r, d_scratch);
                perf.mul_ms += std::chrono::duration<float, std::milli>(hrc::now() - t_mul0).count();
                perf.mul_calls++;
            }

            // Progresso via CPU clock
            auto now = hrc::now();
            auto since_print = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   now - t_last_print).count();
            if (since_print >= 2000 || win == n_windows - 1) {
                t_last_print = now;
                int done_wins = win + 1;
                int pct  = done_wins * 100 / n_windows;
                auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
                               now - t_start).count();
                int done_bits  = done_wins * WINDOW_BITS;
                int total_bits = n_windows * WINDOW_BITS;
                double ms_per_iter = done_bits > 0 ? (double)ms / done_bits : 0;
                printf("\r    bit %d/%d  %3d%%  %.1fs  %.2fms/iter   ",
                       done_bits, total_bits, pct, ms/1000.0, ms_per_iter);
                fflush(stdout);
            }
        }
        printf("\n");

        // ── Check resultado ───────────────────────────────────────────────────
        CU(cudaEventRecord(ev0));
        mont.check_passed(d_r, d_passed);
        CU(cudaMemcpy(passed_h.data(), d_passed, n_total, cudaMemcpyDeviceToHost));
        CU(cudaEventRecord(ev1));
        perf.check_ms += gpu_elapsed();

        for (int t = 0; t < n_total; t++) {
            if (!alive[t]) continue;
            if (!passed_h[t]) {
                alive[t] = false;
                printf("    [%s] entrada %d COMPOSTA (witness %u)\n",
                       label, t, WITNESSES[wi]);
            }
        }
    }

    CU(cudaEventDestroy(ev0));
    CU(cudaEventDestroy(ev1));
    cudaFree(d_table);

    // ── Relatório de performance ──────────────────────────────────────────────
    float window_ms = perf.sq_ms + perf.mul_ms;
    float total_ms  = window_ms + perf.check_ms + perf.setup_ms + perf.table_ms;
    auto pct = [&](float v) { return total_ms > 0 ? v * 100.0f / total_ms : 0.0f; };

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Perfil de tempo — WINDOW_BITS=%-2d                           ║\n", WINDOW_BITS);
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    printf("  window loop (sq + mul)  %8.2fs  %5.1f%%\n",
           window_ms/1000.0, pct(window_ms));
    printf("  ├─ squarings            %8.2fs  %5.1f%%"
           "  (%ld sq,  %.3fms/sq)\n",
           perf.sq_ms/1000.0, pct(perf.sq_ms),
           perf.sq_calls,
           perf.sq_calls > 0 ? perf.sq_ms / perf.sq_calls : 0.0f);
    printf("  └─ mul + select_win     %8.2fs  %5.1f%%"
           "  (%ld jan, %.3fms/jan)\n",
           perf.mul_ms/1000.0, pct(perf.mul_ms),
           perf.mul_calls,
           perf.mul_calls > 0 ? perf.mul_ms / perf.mul_calls : 0.0f);
    printf("\n");
    printf("  pré-cômputo tabela      %8.2fs  %5.1f%%\n",
           perf.table_ms/1000.0, pct(perf.table_ms));
    printf("  check_passed + memcpy   %8.2fs  %5.1f%%\n",
           perf.check_ms/1000.0, pct(perf.check_ms));
    printf("  setup por witness       %8.2fs  %5.1f%%\n",
           perf.setup_ms/1000.0, pct(perf.setup_ms));
    printf("  ────────────────────────────────────────────\n");
    printf("  TOTAL                   %8.2fs\n\n", total_ms/1000.0);

    printf("  Breakdown interno (mont_sq + mont_mul acumulado):\n");
    mont.perf.print();

    cudaFree(d_r); cudaFree(d_base); cudaFree(d_scratch);
    cudaFree(d_one); cudaFree(d_cur_mul); cudaFree(d_exp_dev);
    cudaFree(d_passed);
    return alive;
}

// ── main ──────────────────────────────────────────────────────────────────────

static constexpr int BATCH_SIZE = 128;

int main(int argc, char* argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Uso: %s <candidatos.txt>\n", argv[0]);
        return 1;
    }

    std::vector<Candidate> cands;
    {
        std::ifstream fin(argv[1]);
        if (!fin) { fprintf(stderr, "Erro ao abrir %s\n", argv[1]); return 1; }
        long long a, b, c; int k, j;
        while (fin >> a >> b >> c >> k >> j) {
            Candidate cd; cd.a=a; cd.b=b; cd.c=c; cd.k=k; cd.j=j;
            cands.push_back(cd);
        }
    }
    if (cands.empty()) { fprintf(stderr, "Nenhum candidato.\n"); return 1; }

    int n_cands = (int)cands.size();
    int n_limbs = limbs_for_digits((int)cands[0].a + 4);

    printf("Candidatos: %d  batch_size: %d  witnesses: %d\n",
           n_cands, BATCH_SIZE, N_WITNESSES);
    printf("Limbs 16-bit: %d  NTT padded: %d\n\n", n_limbs, next_pow2_ntt(2*n_limbs));

    printf("Construindo N/revN (GMP)...\n"); fflush(stdout);
    for (auto& c : cands) c.build(n_limbs);

    // ── Testes de corretude (apenas com -DTEST_MODE) ──────────────────────────
#ifdef TEST_MODE
    {
        // Usa o primeiro batch para testar
        int bsz_test = std::min(n_cands, BATCH_SIZE);
        std::vector<uint64_t> N_test((size_t)bsz_test * n_limbs, 0);
        for (int i = 0; i < bsz_test; i++)
            std::copy(cands[i].N_lims.begin(), cands[i].N_lims.end(),
                      N_test.begin() + i*n_limbs);
        BatchMontCtx mont_test(N_test, n_limbs, bsz_test);
        run_correctness_tests(mont_test, N_test);
    }
#endif

    auto t_global = hrc::now();
    std::vector<int> primes_idx;  // indices globais dos candidatos confirmados primos

    int n_batches = (n_cands + BATCH_SIZE - 1) / BATCH_SIZE;

    for (int b = 0; b < n_batches; b++) {
        int bstart = b * BATCH_SIZE;
        int bend   = std::min(bstart + BATCH_SIZE, n_cands);
        int bsz    = bend - bstart;

        printf("\n=== Batch %d/%d  (candidatos %d..%d) ===\n",
               b+1, n_batches, bstart, bend-1);

        // ── Fase 1: testa N para os bsz candidatos ────────────────────────────
        std::vector<uint64_t> N_batch  ((size_t)bsz * n_limbs, 0);
        std::vector<uint64_t> Nm1_batch((size_t)bsz * n_limbs, 0);
        std::vector<uint64_t> exp_batch((size_t)bsz * n_limbs, 0);

        for (int i = 0; i < bsz; i++) {
            auto& c = cands[bstart + i];
            std::copy(c.N_lims.begin(),   c.N_lims.end(),   N_batch.begin()   + i*n_limbs);
            std::copy(c.Nm1_lims.begin(), c.Nm1_lims.end(), Nm1_batch.begin() + i*n_limbs);
            std::copy(c.d_lims.begin(),   c.d_lims.end(),   exp_batch.begin() + i*n_limbs);
        }

        printf("--- Fase 1: testando N (%d candidatos) ---\n", bsz); fflush(stdout);
        auto t0 = hrc::now();
        BatchMontCtx mont_N(N_batch, n_limbs, bsz);
        auto alive_N = run_all_witnesses(mont_N, exp_batch, Nm1_batch, bsz, "N");
        printf("Fase 1: %.2fs\n",
               std::chrono::duration_cast<std::chrono::milliseconds>(
                   hrc::now()-t0).count()/1000.0);

        // Coleta sobreviventes de N
        std::vector<int> survivors;
        for (int i = 0; i < bsz; i++)
            if (alive_N[i]) survivors.push_back(i);

        if (survivors.empty()) {
            printf("Nenhum sobreviveu ao teste de N — pulando revN.\n");
            continue;
        }

        printf("%d/%d sobreviveram ao teste de N.\n", (int)survivors.size(), bsz);

        // ── Fase 2: testa revN apenas dos sobreviventes ───────────────────────
        int n_surv = (int)survivors.size();
        std::vector<uint64_t> revN_batch  ((size_t)n_surv * n_limbs, 0);
        std::vector<uint64_t> revNm1_batch((size_t)n_surv * n_limbs, 0);
        std::vector<uint64_t> drev_batch  ((size_t)n_surv * n_limbs, 0);

        for (int si = 0; si < n_surv; si++) {
            auto& c = cands[bstart + survivors[si]];
            std::copy(c.revN_lims.begin(),   c.revN_lims.end(),   revN_batch.begin()   + si*n_limbs);
            std::copy(c.revNm1_lims.begin(), c.revNm1_lims.end(), revNm1_batch.begin() + si*n_limbs);
            std::copy(c.drev_lims.begin(),   c.drev_lims.end(),   drev_batch.begin()   + si*n_limbs);
        }

        printf("--- Fase 2: testando revN (%d candidatos) ---\n", n_surv); fflush(stdout);
        t0 = hrc::now();
        BatchMontCtx mont_revN(revN_batch, n_limbs, n_surv);
        auto alive_revN = run_all_witnesses(mont_revN, drev_batch, revNm1_batch, n_surv, "revN");
        printf("Fase 2: %.2fs\n",
               std::chrono::duration_cast<std::chrono::milliseconds>(
                   hrc::now()-t0).count()/1000.0);

        for (int si = 0; si < n_surv; si++)
            if (alive_revN[si])
                primes_idx.push_back(bstart + survivors[si]);
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
