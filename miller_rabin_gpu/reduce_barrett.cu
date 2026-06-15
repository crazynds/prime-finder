// reduce_barrett.cu — Redução de Barrett batched.
//
// Forma de trabalho: resíduo plano (x mod N). Pré-computa μ_i = floor(b^{2k_i}/N_i)
// por candidato (b = 2^LIMB_BITS, k_i = limbs tight de N_i) e reduz via 2 multiplicações
// NTT + finalize. Compilado apenas quando MOD_REDUCTION_ALG == MOD_RED_BARRETT.

#include "batch_mod_ctx.cuh"
#include "gmp_helpers.cuh"
#include "mod_internal.cuh"
#include <vector>
#include <gmp.h>

#if MOD_REDUCTION_ALG == MOD_RED_BARRETT

// ── kernels ───────────────────────────────────────────────────────────────────

// Shift à direita com offset POR-CANDIDATO: dst_i = src_i >> (bar_k[i] + delta) limbs.
__global__ static void shift_right_var_batch(
    Data64 *__restrict__ dst,
    const Data64 *__restrict__ src,
    const int *__restrict__ bark, int delta,
    int n_out, int n_src, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= n_out)
        return;
    int off = bark[cand] + delta;
    int s = j + off;
    dst[cand * n_out + j] = (s >= 0 && s < n_src) ? src[cand * n_src + s] : 0ULL;
}

// Finalização (1 thread por candidato): r = T − q̂·N nos W1 limbs baixos (os limbs
// altos de T e q̂·N se cancelam pois 0 <= r_true < 3N < b^{W1}); depois até 2
// subtrações condicionais de N. bar_k POR-candidato, W1 = max(bar_k)+1.
__global__ static void barrett_finalize(
    Data64 *__restrict__ out,
    const Data64 *__restrict__ T,
    const Data64 *__restrict__ qn,
    const Data64 *__restrict__ N,
    Data64 *__restrict__ scratch,
    const int *__restrict__ bark,
    int W1, int out_limbs, int n_sum, int n_batch)
{
    int cand = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch)
        return;

    const int k = bark[cand];
    const Data64 *t = T + (size_t)cand * n_sum;
    const Data64 *q = qn + (size_t)cand * n_sum;
    const Data64 *nn = N + (size_t)cand * out_limbs;
    Data64 *r = scratch + (size_t)cand * W1;
    Data64 *o = out + (size_t)cand * out_limbs;
    const int64_t BASE = (int64_t)1 << LIMB_BITS;

    // r = T − q̂·N (low W1 limbs; borrow final é 0 por construção).
    int64_t borrow = 0;
    for (int j = 0; j < W1; j++)
    {
        int64_t d = (int64_t)t[j] - (int64_t)q[j] - borrow;
        borrow = (d < 0) ? 1 : 0;
        r[j] = (Data64)(d < 0 ? d + BASE : d);
    }

    // Até 2 subtrações condicionais de N (r_true < 3N).
    for (int it = 0; it < 2; it++)
    {
        int cmp = 0;
        for (int j = W1 - 1; j >= 0; j--)
        {
            Data64 nj = (j < k) ? nn[j] : 0ULL;
            if (r[j] != nj)
            {
                cmp = (r[j] > nj) ? 1 : -1;
                break;
            }
        }
        if (cmp < 0)
            break;
        int64_t bw = 0;
        for (int j = 0; j < W1; j++)
        {
            Data64 nj = (j < k) ? nn[j] : 0ULL;
            int64_t d = (int64_t)r[j] - (int64_t)nj - bw;
            bw = (d < 0) ? 1 : 0;
            r[j] = (Data64)(d < 0 ? d + BASE : d);
        }
    }

    for (int j = 0; j < out_limbs; j++)
        o[j] = (j < W1) ? r[j] : 0ULL;
}

// ── helpers GMP específicos ───────────────────────────────────────────────────

// μ = floor(b^{2k}/N), base b = 2^LIMB_BITS. Escreve k+1 limbs (little-endian).
static void compute_barrett_mu(uint64_t *mu_out, const uint64_t *N_lims, int k)
{
    mpz_t N, B2k, mu;
    mpz_init(N);
    mpz_init(B2k);
    mpz_init(mu);
    limbs_to_mpz(N, N_lims, k);
    if (mpz_sgn(N) == 0)
        throw std::runtime_error("Barrett: N == 0");
    mpz_ui_pow_ui(B2k, 2, (unsigned long)(LIMB_BITS * 2 * k)); // b^{2k}
    mpz_tdiv_q(mu, B2k, N);                                    // floor(b^{2k}/N)
    mpz_to_limbs(mu_out, k + 1, mu);
    mpz_clear(N);
    mpz_clear(B2k);
    mpz_clear(mu);
}

// Forma de trabalho Barrett = resíduo plano: res = x mod N (ida e volta idênticas).
void mod_residue_forward(mpz_t res, const mpz_t x, const mpz_t N, int)
{
    mpz_mod(res, x, N);
}
void mod_residue_backward(mpz_t res, const mpz_t x, const mpz_t N, int)
{
    mpz_mod(res, x, N);
}

// ── setup/teardown do backend ─────────────────────────────────────────────────

void BatchModCtx::precompute_reduction(const std::vector<uint64_t> &N_all)
{
    const size_t pb = (size_t)n_batch * padded * sizeof(Data64);
    const size_t sb = (size_t)n_batch * n_sum * sizeof(Data64);

    // bar_k[i] = limbs tight de N_i (índice do MSB não-nulo + 1), POR candidato.
    // bar_W1 = max(bar_k) + 1 = largura uniforme dos buffers A1/μ/q̂.
    std::vector<int> bar_k_all(n_batch, 0);
    int kmax = 0;
    for (int i = 0; i < n_batch; i++)
    {
        const uint64_t *Ni = N_all.data() + (size_t)i * n_limbs;
        int tight = 0;
        for (int j = n_limbs - 1; j >= 0; j--)
            if (Ni[j] != 0)
            {
                tight = j + 1;
                break;
            }
        if (tight < 2)
            throw std::runtime_error("Barrett: N muito pequeno (tight < 2 limbs).");
        bar_k_all[i] = tight;
        if (tight > kmax)
            kmax = tight;
    }
    bar_W1 = kmax + 1;
    CU(cudaMalloc(&d_bar_k, (size_t)n_batch * sizeof(int)));
    CU(cudaMemcpy(d_bar_k, bar_k_all.data(), (size_t)n_batch * sizeof(int), cudaMemcpyHostToDevice));

    // μ_i = floor(b^{2·bar_k_i}/N_i) (k_i+1 limbs, zero-pad a bar_W1) → NTT(μ).
    const size_t w1b = (size_t)n_batch * bar_W1 * sizeof(Data64);
    CU(cudaMalloc(&d_ntt_mu, pb));
    CU(cudaMalloc(&d_bar_a1, w1b));
    CU(cudaMalloc(&d_bar_q, w1b));
    CU(cudaMalloc(&d_bar_q2, sb));
    CU(cudaMalloc(&d_bar_qn, sb));

    std::vector<uint64_t> mu_all((size_t)n_batch * bar_W1, 0);
    for (int i = 0; i < n_batch; i++)
        compute_barrett_mu(mu_all.data() + (size_t)i * bar_W1,
                           N_all.data() + (size_t)i * n_limbs, bar_k_all[i]);
    Data64 *d_mu_tmp = nullptr;
    CU(cudaMalloc(&d_mu_tmp, w1b));
    CU(cudaMemcpy(d_mu_tmp, mu_all.data(), w1b, cudaMemcpyHostToDevice));
    ntt.ntt_A(d_mu_tmp, bar_W1);
    CU(cudaMemcpy(d_ntt_mu, ntt.d_buf_A, pb, cudaMemcpyDeviceToDevice));
    CU(cudaFree(d_mu_tmp));
}

void BatchModCtx::free_reduction()
{
    cudaFree(d_bar_k);
    cudaFree(d_ntt_mu);
    cudaFree(d_bar_a1);
    cudaFree(d_bar_q);
    cudaFree(d_bar_q2);
    cudaFree(d_bar_qn);
}

// cond_sub_batch não é usado no Barrett (finalize faz a subtração), mas a função
// é declarada no header; fornecemos uma definição vazia para satisfazer o linker
// caso alguém a referencie. O reduce_batch do Barrett não a chama.
void BatchModCtx::cond_sub_batch(Data64 *, cudaStream_t) {}

// ── redução ───────────────────────────────────────────────────────────────────

// Redução de Barrett: out = T mod N, com T = A·B em d_T [n_batch*n_sum].
//   q̂ = floor( floor(T/b^{k-1})·μ / b^{k+1} )   (q̂ ∈ {q, q-1, q-2}), k = bar_k[i].
//   out = T − q̂·N, com até 2 subtrações finais de N.
// As stages de perf reaproveitam os campos do REDC para a árvore de relatório.
void BatchModCtx::reduce_batch(Data64 *d_out, cudaStream_t s)
{
    const int thr = MR_THR_REDUCE;
    unsigned nb = (unsigned)n_batch;
    const int W1 = bar_W1; // max(bar_k) + 1
    unsigned bw1 = (unsigned)(W1 + thr - 1) / thr;

    // Passo 1: A1 = floor(T / b^{k_i-1}) → W1 limbs; q2 = A1 · μ (NTT) → d_bar_q2.
    TSTART();
    shift_right_var_batch<<<dim3(bw1, nb), thr, 0, s>>>(
        d_bar_a1, d_T, d_bar_k, -1, W1, n_sum, n_batch);
    ntt.ntt_A(d_bar_a1, W1, s);
    TSTOP(perf_cur->red_ntt_tlow);
    TSTART();
    ntt.pmul_ext(d_ntt_mu, s);
    TSTOP(perf_cur->red_pmul_np);
    TSTART();
    ntt.intt_A(s);
    TSTOP(perf_cur->red_intt_np);
    TSTART();
    ntt.carry_to_limbs(d_bar_q2, n_sum, s);
    TSTOP(perf_cur->red_carry_m);

    // Passo 2: q̂ = floor(q2 / b^{k_i+1}) → W1 limbs; qn = q̂ · N (NTT) → d_bar_qn.
    TSTART();
    shift_right_var_batch<<<dim3(bw1, nb), thr, 0, s>>>(
        d_bar_q, d_bar_q2, d_bar_k, +1, W1, n_sum, n_batch);
    ntt.ntt_A(d_bar_q, W1, s);
    TSTOP(perf_cur->red_ntt_m);
    TSTART();
    ntt.pmul_ext(d_ntt_N, s);
    TSTOP(perf_cur->red_pmul_n);
    TSTART();
    ntt.intt_A(s);
    TSTOP(perf_cur->red_intt_n);
    TSTART();
    ntt.carry_to_limbs(d_bar_qn, n_sum, s);
    TSTOP(perf_cur->red_carry_add);

    // Passo 3: out = (T − qn) mod N, com até 2 subtrações condicionais.
    TSTART();
    {
        const int fthr = 64;
        unsigned fb = (unsigned)(n_batch + fthr - 1) / fthr;
        barrett_finalize<<<fb, fthr, 0, s>>>(
            d_out, d_T, d_bar_qn, d_N, d_bar_q, d_bar_k, W1, n_limbs, n_sum, n_batch);
    }
    TSTOP(perf_cur->cond_sub);
}

#endif // MOD_RED_BARRETT
