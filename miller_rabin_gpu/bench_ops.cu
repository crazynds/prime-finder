// bench_ops.cu — Benchmark de operações Montgomery GPU vs GMP CPU.
//
// Mede ops/segundo para mont_mul, mont_sq e mod em batch, comparando com GMP.
// Tabela: linhas = operação, colunas = tamanho do número em bits.
//
// Ativado com a flag --bench-ops no bench_mr_gpu.

#include "bench_ops.cuh"
#include "montgomery.cuh"
#include "miller_rabin_runner.cuh"
#include "config.cuh"
#include <gmp.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <stdexcept>
#include <random>

using hrc = std::chrono::high_resolution_clock;
using dsec = std::chrono::duration<double>;

// ── Parâmetros ────────────────────────────────────────────────────────────────

static constexpr double BENCH_SECS = 3.0;
static constexpr int N_BATCH = MR_BATCH_SIZE;
static constexpr int BIT_SIZES[] = {128, 1024, 4096, 16384, 65536};
static constexpr int N_SIZES = (int)(sizeof(BIT_SIZES) / sizeof(BIT_SIZES[0]));

// ── Helpers ───────────────────────────────────────────────────────────────────

static gmp_randstate_t rng_state;

static void rand_odd_mpz(mpz_t out, int n_bits, gmp_randstate_t state)
{
    mpz_urandomb(out, state, n_bits);
    mpz_setbit(out, n_bits - 1); // garante tamanho
    mpz_setbit(out, 0);          // garante ímpar (necessário para Montgomery)
}

struct BenchResult
{
    double ops_per_sec;
    long long n_ops;
    double elapsed_sec;
    bool skipped = false;
};

// Formata ops/seg de forma legível (k, M, G)
static std::string fmt_ops(double ops)
{
    if (ops < 0)
        return "  ERRO  ";
    char buf[32];
    if (ops >= 1e9)
        snprintf(buf, sizeof(buf), "%7.2f G", ops / 1e9);
    else if (ops >= 1e6)
        snprintf(buf, sizeof(buf), "%7.2f M", ops / 1e6);
    else if (ops >= 1e3)
        snprintf(buf, sizeof(buf), "%7.2f k", ops / 1e3);
    else
        snprintf(buf, sizeof(buf), "%7.2f  ", ops);
    return buf;
}

// ── Benchmark GMP ─────────────────────────────────────────────────────────────

// GMP: apenas mpz_mul, sem redução.
static BenchResult bench_gmp_mul_only(int n_bits, bool is_last)
{
    __mpz_struct A, B, C;
    mpz_init(&A); mpz_init(&B); mpz_init(&C);
    rand_odd_mpz(&A, n_bits, rng_state);
    rand_odd_mpz(&B, n_bits, rng_state);

    long long ops = 0;
    auto t0 = hrc::now();
    double elapsed = 0;
    do {
        mpz_mul(&C, &A, &B);
        ops++;
        elapsed = dsec(hrc::now() - t0).count();
    } while (elapsed < BENCH_SECS || (is_last && ops < 1));

    mpz_clear(&A); mpz_clear(&B); mpz_clear(&C);
    return {ops / elapsed, ops, elapsed};
}

// GMP: apenas mpz_mul(self), sem redução.
static BenchResult bench_gmp_sq_only(int n_bits, bool is_last)
{
    __mpz_struct A, C;
    mpz_init(&A); mpz_init(&C);
    rand_odd_mpz(&A, n_bits, rng_state);

    long long ops = 0;
    auto t0 = hrc::now();
    double elapsed = 0;
    do {
        mpz_mul(&C, &A, &A);
        ops++;
        elapsed = dsec(hrc::now() - t0).count();
    } while (elapsed < BENCH_SECS || (is_last && ops < 1));

    mpz_clear(&A); mpz_clear(&C);
    return {ops / elapsed, ops, elapsed};
}

// GMP: mpz_mul + mpz_mod — equivalente ao mont_mul_batch do GPU (mul + redução mod N).
static BenchResult bench_gmp_mul(int n_bits, bool is_last)
{
    __mpz_struct A, B, N, C;
    mpz_init(&A); mpz_init(&B); mpz_init(&N); mpz_init(&C);
    rand_odd_mpz(&A, n_bits, rng_state);
    rand_odd_mpz(&B, n_bits, rng_state);
    rand_odd_mpz(&N, n_bits, rng_state);

    long long ops = 0;
    auto t0 = hrc::now();
    double elapsed = 0;
    do {
        mpz_mul(&C, &A, &B);
        mpz_mod(&C, &C, &N);
        ops++;
        elapsed = dsec(hrc::now() - t0).count();
    } while (elapsed < BENCH_SECS || (is_last && ops < 1));

    mpz_clear(&A); mpz_clear(&B); mpz_clear(&N); mpz_clear(&C);
    return {ops / elapsed, ops, elapsed};
}

// GMP: mpz_mul(self) + mpz_mod — equivalente ao mont_sq_batch do GPU.
static BenchResult bench_gmp_sq(int n_bits, bool is_last)
{
    __mpz_struct A, N, C;
    mpz_init(&A); mpz_init(&N); mpz_init(&C);
    rand_odd_mpz(&A, n_bits, rng_state);
    rand_odd_mpz(&N, n_bits, rng_state);

    long long ops = 0;
    auto t0 = hrc::now();
    double elapsed = 0;
    do {
        mpz_mul(&C, &A, &A);
        mpz_mod(&C, &C, &N);
        ops++;
        elapsed = dsec(hrc::now() - t0).count();
    } while (elapsed < BENCH_SECS || (is_last && ops < 1));

    mpz_clear(&A); mpz_clear(&N); mpz_clear(&C);
    return {ops / elapsed, ops, elapsed};
}

// ── Benchmark GPU ─────────────────────────────────────────────────────────────

// mpz_t é typedef __mpz_struct[1] — não pode ser usado em std::vector diretamente.
// Usamos __mpz_struct como elemento e reinterpret_cast para mpz_t*.
static void make_nums(std::vector<__mpz_struct> &storage, std::vector<mpz_t *> &ptrs,
                      int n_bits)
{
    storage.resize(N_BATCH);
    ptrs.resize(N_BATCH);
    for (int i = 0; i < N_BATCH; i++)
    {
        mpz_init(&storage[i]);
        rand_odd_mpz(&storage[i], n_bits, rng_state);
        ptrs[i] = reinterpret_cast<mpz_t *>(&storage[i]);
    }
}

static void free_nums(std::vector<__mpz_struct> &storage)
{
    for (auto &m : storage)
        mpz_clear(&m);
}

static BenchResult bench_gpu_mul_only(int n_bits, bool is_last)
{
    std::vector<__mpz_struct> storage;
    std::vector<mpz_t *> nums;
    make_nums(storage, nums, n_bits);

    BenchResult res = {};
    try {
        BatchMontCtx ctx(nums, 0);
        size_t nb = (size_t)N_BATCH * ctx.n_limbs * sizeof(Data64);
        size_t nb_out = (size_t)N_BATCH * ctx.n_sum * sizeof(Data64);
        Data64 *d_A, *d_B, *d_out;
        cudaMalloc(&d_A, nb); cudaMalloc(&d_B, nb); cudaMalloc(&d_out, nb_out);
        cudaMemcpy(d_A, ctx.d_one_mont, nb, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_B, ctx.d_Nm1_mont, nb, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();

        long long rounds = 0;
        auto t0 = hrc::now();
        double elapsed = 0;
        do {
            ctx.mul_no_redc_batch(d_A, d_B, d_out);
            rounds++;
            elapsed = dsec(hrc::now() - t0).count();
        } while (elapsed < BENCH_SECS || (is_last && rounds < 1));

        res = {(double)(rounds * N_BATCH) / elapsed, rounds * N_BATCH, elapsed};
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_out);
    } catch (const std::exception& e) {
        fprintf(stderr, "  [GPU mul-only %d-bit] ERRO: %s\n", n_bits, e.what());
        res.skipped = true;
    }

    free_nums(storage);
    return res;
}

static BenchResult bench_gpu_sq_only(int n_bits, bool is_last)
{
    std::vector<__mpz_struct> storage;
    std::vector<mpz_t *> nums;
    make_nums(storage, nums, n_bits);

    BenchResult res = {};
    try {
        BatchMontCtx ctx(nums, 0);
        size_t nb = (size_t)N_BATCH * ctx.n_limbs * sizeof(Data64);
        size_t nb_out = (size_t)N_BATCH * ctx.n_sum * sizeof(Data64);
        Data64 *d_A, *d_out;
        cudaMalloc(&d_A, nb); cudaMalloc(&d_out, nb_out);
        cudaMemcpy(d_A, ctx.d_one_mont, nb, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();

        long long rounds = 0;
        auto t0 = hrc::now();
        double elapsed = 0;
        do {
            ctx.sq_no_redc_batch(d_A, d_out);
            rounds++;
            elapsed = dsec(hrc::now() - t0).count();
        } while (elapsed < BENCH_SECS || (is_last && rounds < 1));

        res = {(double)(rounds * N_BATCH) / elapsed, rounds * N_BATCH, elapsed};
        cudaFree(d_A); cudaFree(d_out);
    } catch (const std::exception& e) {
        fprintf(stderr, "  [GPU sq-only %d-bit] ERRO: %s\n", n_bits, e.what());
        res.skipped = true;
    }

    free_nums(storage);
    return res;
}

static BenchResult bench_gpu_mul(int n_bits, bool is_last)
{
    std::vector<__mpz_struct> storage;
    std::vector<mpz_t *> nums;
    make_nums(storage, nums, n_bits);

    BenchResult res = {};
    try
    {
        BatchMontCtx ctx(nums, 0);

        size_t nb = (size_t)N_BATCH * ctx.n_limbs * sizeof(Data64);
        Data64 *d_A, *d_B, *d_out;
        cudaMalloc(&d_A, nb);
        cudaMalloc(&d_B, nb);
        cudaMalloc(&d_out, nb);
        cudaMemcpy(d_A, ctx.d_one_mont, nb, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_B, ctx.d_Nm1_mont, nb, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();

        long long rounds = 0;
        auto t0 = hrc::now();
        double elapsed = 0;
        do
        {
            ctx.mont_mul_batch(d_A, d_B, d_out);
            cudaDeviceSynchronize();
            rounds++;
            elapsed = dsec(hrc::now() - t0).count();
        } while (elapsed < BENCH_SECS || (is_last && rounds < 1));

        res = {(double)(rounds * N_BATCH) / elapsed, rounds * N_BATCH, elapsed};
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_out);
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "  [GPU mul %d-bit] ERRO: %s\n", n_bits, e.what());
        res.skipped = true;
    }

    free_nums(storage);
    return res;
}

static BenchResult bench_gpu_sq(int n_bits, bool is_last)
{
    std::vector<__mpz_struct> storage;
    std::vector<mpz_t *> nums;
    make_nums(storage, nums, n_bits);

    BenchResult res = {};
    try
    {
        BatchMontCtx ctx(nums, 0);

        size_t nb = (size_t)N_BATCH * ctx.n_limbs * sizeof(Data64);
        Data64 *d_A, *d_out;
        cudaMalloc(&d_A, nb);
        cudaMalloc(&d_out, nb);
        cudaMemcpy(d_A, ctx.d_one_mont, nb, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();

        long long rounds = 0;
        auto t0 = hrc::now();
        double elapsed = 0;
        do
        {
            ctx.mont_sq_batch(d_A, d_out);
            cudaDeviceSynchronize();
            rounds++;
            elapsed = dsec(hrc::now() - t0).count();
        } while (elapsed < BENCH_SECS || (is_last && rounds < 1));

        res = {(double)(rounds * N_BATCH) / elapsed, rounds * N_BATCH, elapsed};
        cudaFree(d_A);
        cudaFree(d_out);
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "  [GPU sq %d-bit] ERRO: %s\n", n_bits, e.what());
        res.skipped = true;
    }

    free_nums(storage);
    return res;
}

// ── Benchmark Miller-Rabin ────────────────────────────────────────────────────

// 5 witnesses fixos, iguais nos dois lados.
static const uint32_t MR_WIT[] = {2, 3, 5, 7, 11};
static const int N_MR_WIT = 5;

// Converte __mpz_struct* para array de limbs de 16 bits (little-endian).
static void mpz_to_limbs16(uint64_t *out, int n, __mpz_struct *x)
{
    mpz_t tmp;
    mpz_init_set(tmp, x);
    for (int i = 0; i < n; i++)
    {
        out[i] = mpz_get_ui(tmp) & 0xFFFF;
        mpz_tdiv_q_2exp(tmp, tmp, 16);
    }
    mpz_clear(tmp);
}

// Um teste Miller-Rabin com GMP para um único N.
// Retorna false se composto, true se provavelmente primo.
static bool gmp_mr_single(__mpz_struct *N, __mpz_struct *Nm1, __mpz_struct *d, int s,
                          __mpz_struct *tmp)
{
    for (int wi = 0; wi < N_MR_WIT; wi++)
    {
        mpz_set_ui(tmp, MR_WIT[wi]);
        mpz_powm(tmp, tmp, d, N); // a^d mod N
        if (mpz_cmp_ui(tmp, 1) == 0 || mpz_cmp(tmp, Nm1) == 0)
            continue;
        bool composite = true;
        for (int r = 1; r < s; r++)
        {
            mpz_mul(tmp, tmp, tmp);
            mpz_mod(tmp, tmp, N);
            if (mpz_cmp(tmp, Nm1) == 0)
            {
                composite = false;
                break;
            }
        }
        if (composite)
            return false;
    }
    return true;
}

// GMP Miller-Rabin: N ≡ 3 mod 4 forçado (s=1), d = (N-1)/2.
static BenchResult bench_gmp_mr(int n_bits, bool is_last)
{
    __mpz_struct num, Nm1, d, tmp;
    mpz_init(&num); mpz_init(&Nm1); mpz_init(&d); mpz_init(&tmp);
    rand_odd_mpz(&num, n_bits, rng_state);
    mpz_setbit(&num, 1);             // força N ≡ 3 mod 4
    mpz_sub_ui(&Nm1, &num, 1);       // N-1 = 2*d
    mpz_tdiv_q_2exp(&d, &Nm1, 1);    // d = (N-1)/2

    long long ops = 0;
    auto t0 = hrc::now();
    double elapsed = 0;
    do {
        gmp_mr_single(&num, &Nm1, &d, 1, &tmp);
        ops++;
        elapsed = dsec(hrc::now() - t0).count();
    } while (elapsed < BENCH_SECS || (is_last && ops < 1));

    mpz_clear(&num); mpz_clear(&Nm1); mpz_clear(&d); mpz_clear(&tmp);
    return {ops / elapsed, ops, elapsed};
}

// GPU Miller-Rabin: usa gpu_miller_rabin_s1 (s=1, N ≡ 3 mod 4).
// exp_all = d = (N-1)/2 para cada candidato.
static BenchResult bench_gpu_mr(int n_bits, bool is_last)
{
    std::vector<__mpz_struct> storage;
    std::vector<mpz_t *> nums;
    make_nums(storage, nums, n_bits);
    // força N ≡ 3 mod 4 em todos (bit 1 = 1)
    for (int i = 0; i < N_BATCH; i++)
        mpz_setbit(&storage[i], 1);

    BenchResult res = {};
    try
    {
        BatchMontCtx ctx(nums, 0);
        int nl = ctx.n_limbs;

        // Prepara exp_all = d = (N-1)/2 e Nm1_all = N-1
        std::vector<uint64_t> exp_all((size_t)N_BATCH * nl, 0);
        std::vector<uint64_t> Nm1_all((size_t)N_BATCH * nl, 0);
        for (int i = 0; i < N_BATCH; i++)
        {
            mpz_t Nm1, d;
            mpz_init(Nm1);
            mpz_init(d);
            mpz_sub_ui(Nm1, *nums[i], 1);
            mpz_tdiv_q_2exp(d, Nm1, 1);
            mpz_to_limbs16(Nm1_all.data() + (size_t)i * nl, nl, ((__mpz_struct *)Nm1));
            mpz_to_limbs16(exp_all.data() + (size_t)i * nl, nl, ((__mpz_struct *)d));
            mpz_clear(Nm1);
            mpz_clear(d);
        }

        const std::vector<uint32_t> witnesses(MR_WIT, MR_WIT + N_MR_WIT);
        long long rounds = 0;
        auto t0 = hrc::now();
        double elapsed = 0;
        do
        {
            gpu_miller_rabin_s1(ctx, exp_all, Nm1_all, N_BATCH, witnesses, "bench", false, false);
            rounds++;
            elapsed = dsec(hrc::now() - t0).count();
        } while (elapsed < BENCH_SECS || (is_last && rounds < 1));

        res = {(double)(rounds * N_BATCH) / elapsed, rounds * N_BATCH, elapsed};
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "  [GPU MR %d-bit] ERRO: %s\n", n_bits, e.what());
        res.skipped = true;
    }

    free_nums(storage);
    return res;
}

// ── run_bench_ops ─────────────────────────────────────────────────────────────

void run_bench_ops()
{
    gmp_randinit_mt(rng_state);
    gmp_randseed_ui(rng_state, 0xDEADBEEF);

    // rows: 0=GPU mul, 1=GPU sq, 2=GPU MR,
    //       3=GMP mul, 4=GMP sq, 5=GMP mul+mod, 6=GMP sq+mod, 7=GMP MR
    // rows: 0=GPU mul, 1=GPU sq, 2=GPU mul+mod, 3=GPU sq+mod, 4=GPU MR,
    //       5=GMP mul, 6=GMP sq, 7=GMP mul+mod, 8=GMP sq+mod, 9=GMP MR
    const char *row_names[] = {
        "GPU mul         (ops/s)",
        "GPU sq          (ops/s)",
        "GPU mont_mul    (ops/s)",
        "GPU mont_sq     (ops/s)",
        "GPU miller-rabin(ops/s)",
        "GMP mul         (ops/s)",
        "GMP sq          (ops/s)",
        "GMP mul+mod     (ops/s)",
        "GMP sq+mod      (ops/s)",
        "GMP miller-rabin(ops/s)",
    };
    const int N_ROWS = 10;

    const int speedup_pairs[][2] = {{0, 5}, {1, 6}, {2, 7}, {3, 8}, {4, 9}};
    const char *speedup_names[] = {
        "Speedup GPU/GMP mul     ",
        "Speedup GPU/GMP sq      ",
        "Speedup GPU/GMP mul+mod ",
        "Speedup GPU/GMP sq+mod  ",
        "Speedup GPU/GMP MR      ",
    };

    BenchResult results[N_ROWS][N_SIZES] = {};

    printf("=== Benchmark Montgomery GPU vs GMP  (batch=%d, %.0fs/teste) ===\n"
           "    GPU mul/sq: NTT+pmul+INTT sem REDC  |  mont_mul/sq: com REDC\n"
           "    GMP mul/sq: so multiplicacao         |  mul+mod/sq+mod: com reducao\n"
           "    MR: 1 witness {2}, N equiv 3 mod 4 (s=1)\n\n",
           N_BATCH, BENCH_SECS);

    for (int c = 0; c < N_SIZES; c++)
    {
        int bits = BIT_SIZES[c];
        bool last = (c == N_SIZES - 1);
        printf("── %d bits ──\n", bits);

        printf("  GPU mul          ... "); fflush(stdout);
        results[0][c] = bench_gpu_mul_only(bits, last);
        printf("%.2fs\n", results[0][c].elapsed_sec);

        printf("  GPU sq           ... "); fflush(stdout);
        results[1][c] = bench_gpu_sq_only(bits, last);
        printf("%.2fs\n", results[1][c].elapsed_sec);

        printf("  GPU mont_mul     ... "); fflush(stdout);
        results[2][c] = bench_gpu_mul(bits, last);
        printf("%.2fs\n", results[2][c].elapsed_sec);

        printf("  GPU mont_sq      ... "); fflush(stdout);
        results[3][c] = bench_gpu_sq(bits, last);
        printf("%.2fs\n", results[3][c].elapsed_sec);

        printf("  GPU miller-rabin ... "); fflush(stdout);
        results[4][c] = bench_gpu_mr(bits, last);
        printf("%.2fs\n", results[4][c].elapsed_sec);

        printf("  GMP mul          ... "); fflush(stdout);
        results[5][c] = bench_gmp_mul_only(bits, last);
        printf("%.2fs\n", results[5][c].elapsed_sec);

        printf("  GMP sq           ... "); fflush(stdout);
        results[6][c] = bench_gmp_sq_only(bits, last);
        printf("%.2fs\n", results[6][c].elapsed_sec);

        printf("  GMP mul+mod      ... "); fflush(stdout);
        results[7][c] = bench_gmp_mul(bits, last);
        printf("%.2fs\n", results[7][c].elapsed_sec);

        printf("  GMP sq+mod       ... "); fflush(stdout);
        results[8][c] = bench_gmp_sq(bits, last);
        printf("%.2fs\n", results[8][c].elapsed_sec);

        printf("  GMP miller-rabin ... "); fflush(stdout);
        results[9][c] = bench_gmp_mr(bits, last);
        printf("%.2fs\n\n", results[9][c].elapsed_sec);
    }

    // ── Tabela ────────────────────────────────────────────────────────────────
    const int COL_W = 16;
    const int ROW_W = 28;

    printf("\n");
    printf("%-*s", ROW_W, "Operação");
    for (int c = 0; c < N_SIZES; c++)
        printf("  %*d-bit", COL_W - 5, BIT_SIZES[c]);
    printf("\n%s\n", std::string(ROW_W + N_SIZES * COL_W, '-').c_str());

    for (int r = 0; r < N_ROWS; r++)
    {
        printf("%-*s", ROW_W, row_names[r]);
        for (int c = 0; c < N_SIZES; c++)
        {
            const auto &res = results[r][c];
            printf("  %*s", COL_W - 2,
                   res.skipped ? "N/A" : fmt_ops(res.ops_per_sec).c_str());
        }
        printf("\n");
        if (r == 4)
            printf("\n"); // separador entre GPU e GMP
    }

    printf("%s\n", std::string(ROW_W + N_SIZES * COL_W, '-').c_str());
    for (int p = 0; p < 5; p++)
    {
        int gi = speedup_pairs[p][0], mi = speedup_pairs[p][1];
        printf("%-*s", ROW_W, speedup_names[p]);
        for (int c = 0; c < N_SIZES; c++)
        {
            if (results[gi][c].skipped || results[mi][c].ops_per_sec <= 0)
            {
                printf("  %*s", COL_W - 2, "N/A");
            }
            else
            {
                char buf[32];
                snprintf(buf, sizeof(buf), "%6.1fx",
                         results[gi][c].ops_per_sec / results[mi][c].ops_per_sec);
                printf("  %*s", COL_W - 2, buf);
            }
        }
        printf("\n");
    }
    printf("\n");

    gmp_randclear(rng_state);
}
