// batch_mod_ctx.cu — Núcleo comum do contexto de aritmética modular batched.
//
// Contém o que independe do algoritmo de redução: construção/destruição, conversão
// de/para a forma de trabalho (delegando a mod_residue_forward/backward), check_passed,
// e os drivers modmul/modsq (produto polinomial → reduce_batch). A redução em si vive
// em reductions/montgomery.cu / reductions/barrett.cu; o relatório em helpers/mod_perf.cu.

#include "batch_mod_ctx.cuh"
#include "helpers/gmp_helpers.cuh"
#include "helpers/timers.cuh"
#include <vector>
#include <algorithm>
#include <gmp.h>

// ── kernel de verificação (genérico) ──────────────────────────────────────────

// Um bloco por candidato. Verifica se r == ref_a OU r == ref_b (forma de trabalho).
__global__ static void check_passed_kernel(
    const Data64 *__restrict__ r,
    const Data64 *__restrict__ ref_a, // 1   na forma de trabalho
    const Data64 *__restrict__ ref_b, // N-1 na forma de trabalho
    uint8_t *__restrict__ passed,
    int n_limbs)
{
    int t = blockIdx.x;
    __shared__ int match_a, match_b;
    if (threadIdx.x == 0)
    {
        match_a = 1;
        match_b = 1;
    }
    __syncthreads();

    const Data64 *rv = r + (size_t)t * n_limbs;
    const Data64 *ra = ref_a + (size_t)t * n_limbs;
    const Data64 *rb = ref_b + (size_t)t * n_limbs;

    for (int j = (int)threadIdx.x; j < n_limbs; j += (int)blockDim.x)
    {
        if (rv[j] != ra[j])
            atomicAnd(&match_a, 0);
        if (rv[j] != rb[j])
            atomicAnd(&match_b, 0);
    }
    __syncthreads();

    if (threadIdx.x == 0)
        passed[t] = (uint8_t)(match_a | match_b);
}

// ── construtores ──────────────────────────────────────────────────────────────

static int mpz_compute_n_limbs(const std::vector<mpz_t *> &numbers)
{
    int max_digits = 0;
    for (auto *p : numbers)
    {
        int d = (int)mpz_sizeinbase(*p, 10);
        if (d > max_digits)
            max_digits = d;
    }
    return limbs_for_digits(max_digits + 4);
}

static std::vector<uint64_t> mpz_build_N_all(const std::vector<mpz_t *> &numbers, int nl)
{
    int nb = (int)numbers.size();
    std::vector<uint64_t> N_all((size_t)nb * nl, 0);
    for (int i = 0; i < nb; i++)
        mpz_to_limbs(N_all.data() + i * nl, nl, *numbers[i]);
    return N_all;
}

BatchModCtx::BatchModCtx(const std::vector<mpz_t *> &numbers, int device_id_)
    : BatchModCtx(mpz_build_N_all(numbers, mpz_compute_n_limbs(numbers)),
                  mpz_compute_n_limbs(numbers),
                  (int)numbers.size(),
                  device_id_)
{
}

BatchModCtx::BatchModCtx(const std::vector<uint64_t> &N_all, int n_limbs_, int n_batch_,
                         int device_id_)
    : n_limbs(n_limbs_), n_batch(n_batch_), device_id(device_id_),
      padded(next_pow2_ntt(2 * (n_limbs_ + MOD_NTT_EXTRA))), n_sum(2 * n_limbs_ + 16),
      ntt(n_limbs_ + MOD_NTT_EXTRA, n_batch_)
{
    CU(cudaSetDevice(device_id_));
    const size_t nb = (size_t)n_batch * n_limbs * sizeof(Data64);
    const size_t pb = (size_t)n_batch * padded * sizeof(Data64);
    const size_t sb = (size_t)n_batch * n_sum * sizeof(Data64);

    CU(cudaMalloc(&d_N, nb));
    CU(cudaMalloc(&d_ntt_N, pb));
    CU(cudaMalloc(&d_T, sb));
    CU(cudaMalloc(&d_one_res, nb));
    CU(cudaMalloc(&d_Nm1_res, nb));

    CU(cudaMemcpy(d_N, N_all.data(), nb, cudaMemcpyHostToDevice));

    ntt.ntt_A(d_N, n_limbs);
    CU(cudaMemcpy(d_ntt_N, ntt.d_buf_A, pb, cudaMemcpyDeviceToDevice));

    precompute_reduction(N_all);

    std::vector<uint64_t> one_lims((size_t)n_batch * n_limbs, 0);
    for (int i = 0; i < n_batch; i++)
        one_lims[i * n_limbs] = 1;

    std::vector<uint64_t> Nm1_lims((size_t)n_batch * n_limbs, 0);
    for (int i = 0; i < n_batch; i++)
    {
        const uint64_t *Ni = N_all.data() + i * n_limbs;
        uint64_t *out = Nm1_lims.data() + i * n_limbs;
        std::copy(Ni, Ni + n_limbs, out);
        for (int j = 0; j < n_limbs; j++)
        {
            if (out[j] > 0)
            {
                out[j]--;
                break;
            }
            out[j] = 0xFFFF;
        }
    }

    std::vector<uint64_t> one_res_h, Nm1_res_h;
    to_residue_batch(one_lims, one_res_h);
    to_residue_batch(Nm1_lims, Nm1_res_h);
    CU(cudaMemcpy(d_one_res, one_res_h.data(), nb, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_Nm1_res, Nm1_res_h.data(), nb, cudaMemcpyHostToDevice));

    timer.init();
    perf_mul = build_perf_nodes("mul");
    perf_sq  = build_perf_nodes("sq");
}

BatchModCtx::~BatchModCtx()
{
    cudaFree(d_N);
    cudaFree(d_ntt_N);
    cudaFree(d_T);
    cudaFree(d_one_res);
    cudaFree(d_Nm1_res);
    free_reduction();
    timer.destroy();
}

// ── perfil ────────────────────────────────────────────────────────────────────

void BatchModCtx::perf_flush(cudaStream_t s)
{
    timer.flush(s);
}

// Monta a subárvore de uma via (mul/sq) sob perf_root e retorna o ramo raiz.
// Estrutura (filhos por índice PerfCtxIdx):
//   PERF_PROD → "produto" (PerfProdIdx: NTT, PMUL, INTT, CARRY)
//   PERF_RED  → "reducao" (estrutura interna varia por algoritmo — veja barrett/montgomery)
//   PERF_FIN  → "finalize" / "cond_sub"
PerfNode *BatchModCtx::build_perf_nodes(const char *ctx_name)
{
    PerfNode *ctx = perf_root.branch(ctx_name);

    // PERF_PROD = child(0)
    ctx->branch("produto", {"ntt_input", "pmul/psq", "intt_product", "carry_product"});

#if MOD_REDUCTION_ALG == MOD_RED_BARRETT
    // PERF_RED = child(1): "reducao Barrett"
    // children: child(0) = shift, child(1) = q2 (4 filhos), child(2) = qn (4 filhos)
    PerfNode *red = ctx->branch("reducao Barrett");
    red->branch("shift (A1,q)");
    red->branch("q2 = A1.mu", {"ntt(A1)", "pmul(mu)", "intt(q2)", "carry(q2)"});
    red->branch("qn = q.N",   {"ntt(q)",  "pmul(N)",  "intt(qn)", "carry(qn)"});

    // PERF_FIN = child(2): "barrett_finalize"
    ctx->branch("barrett_finalize", {"sub (T-qn)", "cond_sub N (2x)", "copy_out"});
#else
    // PERF_RED = child(1): "reducao Montgomery"
    // children: child(0) = Multiplicacao (7 filhos), child(1) = Soma (2 filhos), child(2) = shift_right
    PerfNode *red = ctx->branch("reducao Montgomery");
    red->branch("Multiplicacao",
                {"ntt_Tlow", "pmul_Np", "intt_Np", "carry_m", "ntt_m", "pmul_N", "intt_N"});
    red->branch("Soma", {"vadd", "carry_add"});
    red->branch("shift_right");

    // PERF_FIN = child(2): "cond_sub" (folha)
    ctx->branch("cond_sub");
#endif
    return ctx;
}

// ── conversões host de/para a forma de trabalho ───────────────────────────────

void BatchModCtx::to_residue_batch(const std::vector<uint64_t> &x_all,
                                   std::vector<uint64_t> &out_all) const
{
    out_all.resize((size_t)n_batch * n_limbs, 0);
    std::vector<uint64_t> N_h((size_t)n_batch * n_limbs);
    CU(cudaMemcpy(N_h.data(), d_N, N_h.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    mpz_t xm, N, res;
    mpz_init(xm);
    mpz_init(N);
    mpz_init(res);
    for (int i = 0; i < n_batch; i++)
    {
        limbs_to_mpz(xm, x_all.data() + i * n_limbs, n_limbs);
        limbs_to_mpz(N, N_h.data() + i * n_limbs, n_limbs);
        mod_residue_forward(res, xm, N, n_limbs);
        mpz_to_limbs(out_all.data() + i * n_limbs, n_limbs, res);
    }
    mpz_clear(xm);
    mpz_clear(N);
    mpz_clear(res);
}

void BatchModCtx::from_residue_batch(const Data64 *d_x, std::vector<uint64_t> &out_all) const
{
    out_all.resize((size_t)n_batch * n_limbs, 0);

    std::vector<uint64_t> x_h((size_t)n_batch * n_limbs);
    std::vector<uint64_t> N_h((size_t)n_batch * n_limbs);
    CU(cudaMemcpy(x_h.data(), d_x, x_h.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(N_h.data(), d_N, N_h.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    mpz_t xm, N, res;
    mpz_init(xm);
    mpz_init(N);
    mpz_init(res);
    for (int i = 0; i < n_batch; i++)
    {
        limbs_to_mpz(xm, x_h.data() + i * n_limbs, n_limbs);
        limbs_to_mpz(N, N_h.data() + i * n_limbs, n_limbs);
        mod_residue_backward(res, xm, N, n_limbs);
        mpz_to_limbs(out_all.data() + i * n_limbs, n_limbs, res);
    }
    mpz_clear(xm);
    mpz_clear(N);
    mpz_clear(res);
}

void BatchModCtx::check_passed(const Data64 *d_r, uint8_t *d_passed,
                               cudaStream_t s) const
{
    check_passed_kernel<<<n_batch, MR_THR_CHECK, 0, s>>>(
        d_r, d_one_res, d_Nm1_res, d_passed, n_limbs);
}

// ── drivers modmul / modsq ────────────────────────────────────────────────────

void BatchModCtx::modmul_batch(const Data64 *d_A, const Data64 *d_B, Data64 *d_out,
                               cudaStream_t s)
{
    perf_cur = perf_mul;
    PerfNode *prod = perf_cur->child(PERF_PROD);

    TSTART();
#if MONT_MUL_ALG == MONT_MUL_ALG_NTT
    ntt.ntt_AB(d_A, d_B, n_limbs, s);
#elif MONT_MUL_ALG == MONT_MUL_ALG_SCHOOLBOOK
    ntt.schoolbook_mul(d_A, d_B, n_limbs, s);
#endif
    TSTOP(prod->child(PERF_PROD_NTT));

#if MONT_MUL_ALG == MONT_MUL_ALG_NTT
    TSTART();
    ntt.pmul(s);
    TSTOP(prod->child(PERF_PROD_PMUL));
    TSTART();
    ntt.intt_A(s);
    TSTOP(prod->child(PERF_PROD_INTT));
#endif

    TSTART();
    ntt.carry_to_limbs(d_T, n_sum, s);
    TSTOP(prod->child(PERF_PROD_CARRY));

    reduce_batch(d_out, s);
    perf_flush(s);
}

void BatchModCtx::modsq_batch(const Data64 *d_A, Data64 *d_out, cudaStream_t s)
{
    perf_cur = perf_sq;
    PerfNode *prod = perf_cur->child(PERF_PROD);

    TSTART();
#if MONT_MUL_ALG == MONT_MUL_ALG_NTT
    ntt.ntt_A(d_A, n_limbs, s);
#elif MONT_MUL_ALG == MONT_MUL_ALG_SCHOOLBOOK
    ntt.schoolbook_sq(d_A, n_limbs, s);
#endif
    TSTOP(prod->child(PERF_PROD_NTT));

#if MONT_MUL_ALG == MONT_MUL_ALG_NTT
    TSTART();
    ntt.psq(s);
    TSTOP(prod->child(PERF_PROD_PMUL));
    TSTART();
    ntt.intt_A(s);
    TSTOP(prod->child(PERF_PROD_INTT));
#endif

    TSTART();
    ntt.carry_to_limbs(d_T, n_sum, s);
    TSTOP(prod->child(PERF_PROD_CARRY));

    reduce_batch(d_out, s);
    perf_flush(s);
}

// ── multiplicações sem redução (benchmark) ────────────────────────────────────

void BatchModCtx::mul_no_redc_batch(const Data64 *d_A, const Data64 *d_B,
                                    Data64 *d_out, cudaStream_t s)
{
    ntt.ntt_AB(d_A, d_B, n_limbs, s);
    ntt.pmul_and_intt(s);
    ntt.carry_to_limbs(d_out, n_sum, s);
    cudaStreamSynchronize(s);
}

void BatchModCtx::sq_no_redc_batch(const Data64 *d_A, Data64 *d_out, cudaStream_t s)
{
    ntt.ntt_A(d_A, n_limbs, s);
    ntt.psq_and_intt(s);
    ntt.carry_to_limbs(d_out, n_sum, s);
    cudaStreamSynchronize(s);
}
