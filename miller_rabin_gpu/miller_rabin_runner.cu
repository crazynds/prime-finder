// miller_rabin_runner.cu
#include "miller_rabin_runner.cuh"
#include "bigint_ntt.cuh"
#include "time_format.h"
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <string>

#define CU(expr) \
    do { cudaError_t _e=(expr); if(_e!=cudaSuccess) \
        throw std::runtime_error(std::string("[CUDA] " #expr ": ")+cudaGetErrorString(_e)); \
    } while(0)

using hrc = std::chrono::high_resolution_clock;

// ── Kernel: seleciona table[w] para cada candidato dado janela de WINDOW_BITS bits ──
// msb_pos: bit mais significativo da janela (inclusive).
// Extrai k bits [msb_pos .. msb_pos-k+1] do expoente e copia d_table[w * ...].

__global__ void select_window_kernel(
        Data64* __restrict__       d_out,
        const Data64* __restrict__ d_table,
        const Data64* __restrict__ d_exp,
        int msb_pos, int k,
        int n_limbs, int n_total)
{
    int t = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_total || j >= n_limbs) return;

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

// ── Kernel: checa se d_r == d_ref para cada candidato em d_alive ─────────────
// Se iguais e alive[t]==1, seta alive[t]=2 (passou por N-1).

__global__ static void check_equals_kernel(
        const Data64* __restrict__ d_r,
        const Data64* __restrict__ d_ref,
        uint8_t* __restrict__      d_alive,
        int n_limbs, int n_total)
{
    int t = blockIdx.x;
    if (t >= n_total || d_alive[t] != 1) return;

    __shared__ int match;
    if (threadIdx.x == 0) match = 1;
    __syncthreads();

    const Data64* rv = d_r   + (size_t)t * n_limbs;
    const Data64* ref = d_ref + (size_t)t * n_limbs;
    for (int j = (int)threadIdx.x; j < n_limbs; j += (int)blockDim.x)
        if (rv[j] != ref[j]) atomicAnd(&match, 0);
    __syncthreads();

    if (threadIdx.x == 0 && match)
        d_alive[t] = 2;  // passou (r == ref)
}

// ── Estrutura de contadores de performance ────────────────────────────────────

struct PerfCtrs {
    float sq_ms      = 0, mul_ms    = 0;
    float table_ms   = 0, check_ms  = 0;
    float setup_ms   = 0;   // CPU: to_mont + preparação
    float memcpy_ms  = 0;   // apenas cudaMemcpy do setup
    size_t memcpy_bytes = 0;
    long  sq_calls = 0, mul_calls = 0;
};

// ── Loop de exponenciação por janela deslizante ───────────────────────────────
// Computa a^exp para todos os candidatos e retorna d_r (resultado em Montgomery).
// Aloca e desaloca d_table internamente.

static void window_exp_loop(
        BatchMontCtx& mont,
        const std::vector<uint64_t>& exp_all,
        Data64*& d_r,
        Data64* d_one_mont_h,
        Data64* d_base,
        Data64*& d_scratch,
        Data64* d_cur_mul,
        Data64* d_exp_dev,
        int n_total,
        PerfCtrs& perf,
        uint32_t witness,
        bool show_progress)
{
    int n = mont.n_limbs;
    size_t total_bytes = (size_t)n_total * n * sizeof(Data64);

    // Pré-computa tabela: table[0]=1, table[1]=base, ..., table[WINDOW_SIZE-1]=base^(WINDOW_SIZE-1)
    Data64* d_table;
    CU(cudaMalloc(&d_table, (size_t)WINDOW_SIZE * total_bytes));

    cudaEvent_t ev0, ev1;
    CU(cudaEventCreate(&ev0)); CU(cudaEventCreate(&ev1));
    auto elapsed_ms = [&]() { float ms=0; CU(cudaEventSynchronize(ev1)); CU(cudaEventElapsedTime(&ms,ev0,ev1)); return ms; };

    CU(cudaEventRecord(ev0));
    CU(cudaMemcpy(d_table, d_one_mont_h, total_bytes, cudaMemcpyDeviceToDevice));
    CU(cudaMemcpy(d_table + (size_t)1 * n_total * n, d_base, total_bytes, cudaMemcpyDeviceToDevice));
    for (int w = 2; w < WINDOW_SIZE; w++)
        mont.mont_mul_batch(d_table + (size_t)(w-1)*n_total*n, d_base, d_table + (size_t)w*n_total*n);
    CU(cudaEventRecord(ev1));
    perf.table_ms += elapsed_ms();

    // Encontra MSB real entre todos os candidatos
    int msb = n * LIMB_BITS - 1;
    while (msb > 0) {
        int li = msb/LIMB_BITS, bit = msb%LIMB_BITS;
        bool any = false;
        for (int t = 0; t < n_total && !any; t++)
            if ((exp_all[t*n + li] >> bit) & 1) any = true;
        if (any) break;
        msb--;
    }

    int n_windows = msb / WINDOW_BITS + 1;
    int start_win = n_windows * WINDOW_BITS - 1;

    // Precomputa quais janelas têm algum bit != 0 para evitar muls desnecessárias
    std::vector<bool> any_nonzero(n_windows, false);
    for (int wi = 0; wi < n_windows; wi++) {
        int i = start_win - wi * WINDOW_BITS;
        for (int t = 0; t < n_total && !any_nonzero[wi]; t++)
            for (int b = 0; b < WINDOW_BITS && !any_nonzero[wi]; b++) {
                int bp = i - b;
                if (bp >= 0 && bp <= msb) {
                    if ((exp_all[t*n + bp/LIMB_BITS] >> (bp%LIMB_BITS)) & 1)
                        any_nonzero[wi] = true;
                }
            }
    }

    const int thr = MR_THR_SELECT_WIN;
    dim3 grid_sel((unsigned)(n + thr-1)/thr, (unsigned)n_total);

    auto t_start      = hrc::now();
    auto t_last_print = t_start;

    for (int win = 0; win < n_windows; win++) {
        int i = start_win - win * WINDOW_BITS;

        auto t_sq0 = hrc::now();
        for (int sq = 0; sq < WINDOW_BITS; sq++) {
            mont.mont_sq_batch(d_r, d_scratch);
            std::swap(d_r, d_scratch);
        }
        perf.sq_ms    += std::chrono::duration<float,std::milli>(hrc::now()-t_sq0).count();
        perf.sq_calls += WINDOW_BITS;

        if (any_nonzero[win]) {
            auto t_mul0 = hrc::now();
            select_window_kernel<<<grid_sel, thr>>>(d_cur_mul, d_table, d_exp_dev, i, WINDOW_BITS, n, n_total);
            mont.mont_mul_batch(d_r, d_cur_mul, d_scratch);
            std::swap(d_r, d_scratch);
            perf.mul_ms += std::chrono::duration<float,std::milli>(hrc::now()-t_mul0).count();
            perf.mul_calls++;
        }

        if (show_progress) {
            auto now = hrc::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now-t_last_print).count() >= MR_PROGRESS_INTERVAL_MS
                || win == n_windows-1)
            {
                t_last_print = now;
                int done_bits  = (win+1) * WINDOW_BITS;
                int total_bits = n_windows * WINDOW_BITS;
                double ms = std::chrono::duration_cast<std::chrono::milliseconds>(now-t_start).count();
                printf("\r    bit %d/%d  %3d%%  %s  %s/iter   ",
                       done_bits, total_bits, done_bits*100/total_bits,
                       fmt_time_ms(ms).c_str(),
                       fmt_time_ms(done_bits>0 ? ms/done_bits : 0.0).c_str());
                fflush(stdout);
            }
        }
    }
    if (show_progress) printf("\n");

    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    cudaFree(d_table);
}

// ── Imprime relatório de performance ─────────────────────────────────────────

static void print_perf(const PerfCtrs& perf, BatchMontCtx& mont)
{
    float window_ms = perf.sq_ms + perf.mul_ms;
    float total_ms  = window_ms + perf.check_ms + perf.setup_ms + perf.memcpy_ms + perf.table_ms;
    auto pct = [&](float v) { return total_ms > 0 ? v*100.0f/total_ms : 0.0f; };

    double memcpy_gb   = perf.memcpy_bytes / 1e9;
    double memcpy_gbps = perf.memcpy_ms > 0 ? memcpy_gb / (perf.memcpy_ms / 1000.0) : 0.0;

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Perfil de tempo — WINDOW_BITS=%-2d                           ║\n", WINDOW_BITS);
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("  window loop (sq + mul)  %12s  %5.1f%%\n", fmt_time_ms(window_ms).c_str(), pct(window_ms));
    printf("  ├─ squarings            %12s  %5.1f%%  (%ld sq,  %s/sq)\n",
           fmt_time_ms(perf.sq_ms).c_str(), pct(perf.sq_ms), perf.sq_calls,
           fmt_time_ms(perf.sq_calls > 0 ? perf.sq_ms/perf.sq_calls : 0.0).c_str());
    printf("  └─ mul + select_win     %12s  %5.1f%%  (%ld jan, %s/jan)\n",
           fmt_time_ms(perf.mul_ms).c_str(), pct(perf.mul_ms), perf.mul_calls,
           fmt_time_ms(perf.mul_calls > 0 ? perf.mul_ms/perf.mul_calls : 0.0).c_str());
    printf("\n  Breakdown da aplicacao (arvore unificada):\n");

    // Fases de host (entram na árvore sob "setup / host"). Só o memcpy carrega
    // anotação de banda (GB/s).
    char gbps[32];
    snprintf(gbps, sizeof(gbps), "(%.2f GB/s)", memcpy_gbps);
    std::vector<BatchMontCtx::HostPhase> host = {
        {"pre-computo tabela", perf.table_ms, ""},
        {"setup CPU (to_mont)", perf.setup_ms, ""},
        {"check + memcpy", perf.check_ms, ""},
        {"memcpy setup", perf.memcpy_ms, gbps},
    };
    mont.print_perf(total_ms, host);
    carry_stats_print_and_reset();
}

// ── Aloca buffers de trabalho para os witnesses ───────────────────────────────

struct WitnessBuffers {
    Data64 *d_r, *d_base, *d_scratch, *d_one, *d_cur_mul, *d_exp_dev;
    uint8_t* d_passed;
    int n_total, n;

    WitnessBuffers(BatchMontCtx& mont, const std::vector<uint64_t>& exp_all,
                   int n_total_)
        : n_total(n_total_), n(mont.n_limbs)
    {
        size_t tb = (size_t)n_total * n * sizeof(Data64);
        CU(cudaMalloc(&d_r,       tb));
        CU(cudaMalloc(&d_base,    tb));
        CU(cudaMalloc(&d_scratch, tb));
        CU(cudaMalloc(&d_one,     tb));
        CU(cudaMalloc(&d_cur_mul, tb));
        CU(cudaMalloc(&d_exp_dev, tb));
        CU(cudaMalloc(&d_passed,  (size_t)n_total));
        CU(cudaMemcpy(d_exp_dev, exp_all.data(), tb, cudaMemcpyHostToDevice));

        // 1 em forma Montgomery
        std::vector<uint64_t> one_all((size_t)n_total * n, 0);
        for (int t = 0; t < n_total; t++) one_all[t*n] = 1;
        std::vector<uint64_t> one_mont;
        mont.to_mont_batch(one_all, one_mont);
        CU(cudaMemcpy(d_one, one_mont.data(), tb, cudaMemcpyHostToDevice));
    }

    ~WitnessBuffers() {
        cudaFree(d_r);   cudaFree(d_base); cudaFree(d_scratch);
        cudaFree(d_one); cudaFree(d_cur_mul); cudaFree(d_exp_dev);
        cudaFree(d_passed);
    }
};

// ── miller_rabin_s1 ───────────────────────────────────────────────────────────
// Otimizado para s=1: computa a^d onde d=(N-1)/2 e verifica se r == ±1 (mod N).

std::vector<bool> gpu_miller_rabin_s1(
        BatchMontCtx& mont,
        const std::vector<uint64_t>& exp_all,
        const std::vector<uint64_t>& Nm1_all,
        int n_total,
        const std::vector<uint32_t>& witnesses,
        const char* label,
        bool show_report,
        bool show_progress)
{
    int n = mont.n_limbs;
    size_t total_bytes = (size_t)n_total * n * sizeof(Data64);
    WitnessBuffers buf(mont, exp_all, n_total);

    std::vector<uint8_t> passed_h(n_total);
    std::vector<bool>    alive(n_total, true);
    PerfCtrs perf;

    mont.perf_enabled = show_report;

    cudaEvent_t ev0, ev1;
    CU(cudaEventCreate(&ev0)); CU(cudaEventCreate(&ev1));
    auto elapsed_ms = [&]() { float ms=0; CU(cudaEventSynchronize(ev1)); CU(cudaEventElapsedTime(&ms,ev0,ev1)); return ms; };

    for (int wi = 0; wi < (int)witnesses.size(); wi++) {
        int n_alive = 0;
        for (bool b : alive) if (b) n_alive++;
        if (n_alive == 0) break;

        if (show_report) {
            printf("  [%s] Witness %-3u  vivos: %d\n", label, witnesses[wi], n_alive);
            fflush(stdout);
        }

        // Setup CPU: calcula base em forma Montgomery
        CU(cudaEventRecord(ev0));
        std::vector<uint64_t> w_all((size_t)n_total * n, 0);
        for (int t = 0; t < n_total; t++) {
            w_all[t*n]   = witnesses[wi] & LIMB_MASK;
            w_all[t*n+1] = (witnesses[wi] >> LIMB_BITS) & LIMB_MASK;
        }
        std::vector<uint64_t> base_mont;
        mont.to_mont_batch(w_all, base_mont);
        CU(cudaEventRecord(ev1));
        perf.setup_ms += elapsed_ms();

        // Memcpy setup: base → GPU, 1_mont → d_r
        CU(cudaEventRecord(ev0));
        CU(cudaMemcpy(buf.d_base, base_mont.data(), total_bytes, cudaMemcpyHostToDevice));
        CU(cudaMemcpy(buf.d_r,    buf.d_one,        total_bytes, cudaMemcpyDeviceToDevice));
        CU(cudaEventRecord(ev1));
        perf.memcpy_ms    += elapsed_ms();
        perf.memcpy_bytes += total_bytes * 2;  // base (H→D) + d_r init (D→D)

        // Exponenciação por janela: r = base^d
        window_exp_loop(mont, exp_all, buf.d_r, buf.d_one, buf.d_base, buf.d_scratch,
                        buf.d_cur_mul, buf.d_exp_dev, n_total, perf, witnesses[wi], show_progress);

        // Verifica r == 1 ou r == N-1
        CU(cudaEventRecord(ev0));
        mont.check_passed(buf.d_r, buf.d_passed);
        CU(cudaMemcpy(passed_h.data(), buf.d_passed, n_total, cudaMemcpyDeviceToHost));
        CU(cudaEventRecord(ev1));
        perf.check_ms += elapsed_ms();

        for (int t = 0; t < n_total; t++) {
            if (!alive[t]) continue;
            if (!passed_h[t]) {
                alive[t] = false;
                if (show_report)
                    printf("    [%s] entrada %d COMPOSTA (witness %u)\n", label, t, witnesses[wi]);
            }
        }
    }

    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    if (show_report) print_perf(perf, mont);
    return alive;
}

// ── miller_rabin ──────────────────────────────────────────────────────────────
// Versão geral para N-1 = 2^s * d.
// Após computar r = a^d, faz até s-1 squarings extras verificando N-1.

std::vector<bool> gpu_miller_rabin(
        BatchMontCtx& mont,
        const std::vector<uint64_t>& exp_all,
        const std::vector<uint64_t>& Nm1_all,
        int s,
        int n_total,
        const std::vector<uint32_t>& witnesses,
        const char* label,
        bool show_report,
        bool show_progress)
{
    // s=1 é um caso especial sem squarings extras — delega para a versão otimizada
    if (s == 1) return gpu_miller_rabin_s1(mont, exp_all, Nm1_all, n_total, witnesses, label, show_report, show_progress);

    int n = mont.n_limbs;
    size_t total_bytes = (size_t)n_total * n * sizeof(Data64);
    WitnessBuffers buf(mont, exp_all, n_total);

    mont.perf_enabled = show_report;

    // d_alive[t]: 0=desconhecido, 1=ainda vivo no round atual, 2=passou (primo provável)
    uint8_t* d_alive;
    CU(cudaMalloc(&d_alive, (size_t)n_total));

    std::vector<uint8_t> alive_h(n_total);
    std::vector<bool>    round_alive(n_total, true);
    PerfCtrs perf;

    cudaEvent_t ev0, ev1;
    CU(cudaEventCreate(&ev0)); CU(cudaEventCreate(&ev1));
    auto elapsed_ms = [&]() { float ms=0; CU(cudaEventSynchronize(ev1)); CU(cudaEventElapsedTime(&ms,ev0,ev1)); return ms; };

    for (int wi = 0; wi < (int)witnesses.size(); wi++) {
        int n_alive = 0;
        for (bool b : round_alive) if (b) n_alive++;
        if (n_alive == 0) break;

        if (show_report) {
            printf("  [%s] Witness %-3u  vivos: %d\n", label, witnesses[wi], n_alive);
            fflush(stdout);
        }

        // Setup CPU: calcula base em forma Montgomery
        CU(cudaEventRecord(ev0));
        std::vector<uint64_t> w_all((size_t)n_total * n, 0);
        for (int t = 0; t < n_total; t++) {
            w_all[t*n]   = witnesses[wi] & LIMB_MASK;
            w_all[t*n+1] = (witnesses[wi] >> LIMB_BITS) & LIMB_MASK;
        }
        std::vector<uint64_t> base_mont;
        mont.to_mont_batch(w_all, base_mont);
        for (int t = 0; t < n_total; t++) alive_h[t] = round_alive[t] ? 1 : 0;
        CU(cudaEventRecord(ev1));
        perf.setup_ms += elapsed_ms();

        // Memcpy setup: base → GPU, d_r init, alive → GPU
        CU(cudaEventRecord(ev0));
        CU(cudaMemcpy(buf.d_base, base_mont.data(), total_bytes, cudaMemcpyHostToDevice));
        CU(cudaMemcpy(buf.d_r,    buf.d_one,        total_bytes, cudaMemcpyDeviceToDevice));
        CU(cudaMemcpy(d_alive,    alive_h.data(),   n_total,     cudaMemcpyHostToDevice));
        CU(cudaEventRecord(ev1));
        perf.memcpy_ms    += elapsed_ms();
        perf.memcpy_bytes += total_bytes * 2 + (size_t)n_total;  // base + d_r + alive

        // Exponenciação por janela: r = base^d
        window_exp_loop(mont, exp_all, buf.d_r, buf.d_one, buf.d_base, buf.d_scratch,
                        buf.d_cur_mul, buf.d_exp_dev, n_total, perf, witnesses[wi], show_progress);

        // Verifica r == 1 ou r == N-1 (inicial)
        CU(cudaEventRecord(ev0));
        mont.check_passed(buf.d_r, buf.d_passed);
        CU(cudaMemcpy(alive_h.data(), buf.d_passed, n_total, cudaMemcpyDeviceToHost));
        // Marca como 2 (passou) quem obteve r==1 ou r==N-1;
        // quem ainda é candidato (round_alive mas não passou) fica com 1 para
        // os squarings extras seguintes; os já descartados ficam com 0.
        for (int t = 0; t < n_total; t++) {
            if (!round_alive[t])      alive_h[t] = 0;  // já morto
            else if (alive_h[t])      alive_h[t] = 2;  // passou no check inicial
            else                      alive_h[t] = 1;  // ainda na corrida, precisa de squarings
        }
        CU(cudaMemcpy(d_alive, alive_h.data(), n_total, cudaMemcpyHostToDevice));
        CU(cudaEventRecord(ev1));
        perf.check_ms += elapsed_ms();

        // Squarings extras: verifica r^(2^i) == N-1 para i = 1..s-1
        for (int sq = 1; sq < s; sq++) {
            mont.mont_sq_batch(buf.d_r, buf.d_scratch);
            std::swap(buf.d_r, buf.d_scratch);

            // Marca 2 quem ainda é vivo (alive==1) e r == N-1
            check_equals_kernel<<<n_total, MR_THR_CHECK>>>(buf.d_r, mont.d_Nm1_mont, d_alive, n, n_total);
        }

        CU(cudaMemcpy(alive_h.data(), d_alive, n_total, cudaMemcpyDeviceToHost));
        for (int t = 0; t < n_total; t++) {
            if (!round_alive[t]) continue;
            if (alive_h[t] != 2) {  // não passou por nenhuma verificação
                round_alive[t] = false;
                if (show_report)
                    printf("    [%s] entrada %d COMPOSTA (witness %u)\n", label, t, witnesses[wi]);
            }
        }
    }

    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    cudaFree(d_alive);
    if (show_report) print_perf(perf, mont);
    return round_alive;
}
