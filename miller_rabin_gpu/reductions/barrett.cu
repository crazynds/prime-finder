// reductions/barrett.cu — Redução de Barrett batched.
//
// Forma de trabalho: resíduo plano (x mod N). Pré-computa μ_i = floor(b^{2k_i}/N_i)
// por candidato (b = 2^LIMB_BITS, k_i = limbs tight de N_i) e reduz via 2 multiplicações
// NTT + finalize. Compilado apenas quando MOD_REDUCTION_ALG == MOD_RED_BARRETT.

#include "batch_mod_ctx.cuh"
#include "helpers/gmp_helpers.cuh"
#include "helpers/timers.cuh"
#include "ops/shift/shift.cuh"
#include "ops/sub/sub.cuh"
#include <vector>
#include <gmp.h>

#if MOD_REDUCTION_ALG == MOD_RED_BARRETT

// Índices locais (filhos de perf_cur->child(PERF_RED)):
namespace {
    enum BarIdx { BAR_SHIFT = 0, BAR_Q2 = 1, BAR_QN = 2 };
    // Filhos de Q2 e QN:
    enum QIdx   { Q_NTT = 0, Q_PMUL = 1, Q_INTT = 2, Q_CARRY = 3 };
    // Filhos de perf_cur->child(PERF_FIN):
    enum FinIdx { FIN_SUB = 0, FIN_CONDSUB = 1, FIN_COPY = 2 };
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

    const size_t w1b = (size_t)n_batch * bar_W1 * sizeof(Data64);
    CU(cudaMalloc(&d_ntt_mu, pb));
    CU(cudaMalloc(&d_bar_w1, w1b));
    CU(cudaMalloc(&d_bar_prod, sb));

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

    n_cs_tiles = ops::sub_n_tiles(bar_W1);
    const size_t csb = (size_t)n_batch * n_cs_tiles * sizeof(int);
    CU(cudaMalloc(&d_cs_tile_cmp, csb));
    CU(cudaMalloc(&d_cs_tile_bstate, csb));
}

void BatchModCtx::free_reduction()
{
    cudaFree(d_bar_k);
    cudaFree(d_ntt_mu);
    cudaFree(d_bar_w1);
    cudaFree(d_bar_prod);
    cudaFree(d_cs_tile_cmp);
    cudaFree(d_cs_tile_bstate);
}

// cond_sub_batch não é usado no Barrett (finalize faz a subtração), mas a função
// é declarada no header; fornecemos uma definição vazia para satisfazer o linker.
void BatchModCtx::cond_sub_batch(Data64 *, cudaStream_t) {}

// ── redução ───────────────────────────────────────────────────────────────────

// Redução de Barrett: out = T mod N, com T = A·B em d_T [n_batch*n_sum].
//   q̂ = floor( floor(T/b^{k-1})·μ / b^{k+1} )   (q̂ ∈ {q, q-1, q-2}), k = bar_k[i].
//   out = T − q̂·N, com até 2 subtrações finais de N.
void BatchModCtx::reduce_batch(Data64 *d_out, cudaStream_t s)
{
    const int thr = MR_THR_REDUCE;
    const int W1  = bar_W1;

    PerfNode *red = perf_cur->child(PERF_RED);
    PerfNode *q2  = red->child(BAR_Q2);
    PerfNode *qn  = red->child(BAR_QN);
    PerfNode *fin = perf_cur->child(PERF_FIN);

    // Passo 1: A1 = floor(T / b^{k_i-1}) → W1 limbs; q2 = A1·μ (NTT) → d_bar_prod.
    TSTART();
    ops::shift_right_var(d_bar_w1, d_T, d_bar_k, -1, W1, n_sum, n_batch, thr, s);
    TSTOP(red->child(BAR_SHIFT));
    TSTART();
    ntt.ntt_A(d_bar_w1, W1, s);
    TSTOP(q2->child(Q_NTT));
    TSTART();
    ntt.pmul_ext(d_ntt_mu, s);
    TSTOP(q2->child(Q_PMUL));
    TSTART();
    ntt.intt_A(s);
    TSTOP(q2->child(Q_INTT));
    TSTART();
    ntt.carry_to_limbs(d_bar_prod, n_sum, s);
    TSTOP(q2->child(Q_CARRY));

    // Passo 2: q̂ = floor(q2 / b^{k_i+1}) → W1 limbs; qn = q̂·N (NTT) → d_bar_prod.
    TSTART();
    ops::shift_right_var(d_bar_w1, d_bar_prod, d_bar_k, +1, W1, n_sum, n_batch, thr, s);
    TSTOP(red->child(BAR_SHIFT));
    TSTART();
    ntt.ntt_A(d_bar_w1, W1, s);
    TSTOP(qn->child(Q_NTT));
    TSTART();
    ntt.pmul_ext(d_ntt_N, s);
    TSTOP(qn->child(Q_PMUL));
    TSTART();
    ntt.intt_A(s);
    TSTOP(qn->child(Q_INTT));
    // Lever 2: normaliza só os W1 limbs baixos de qn (os altos cancelam em T−qn).
    // carry_to_limbs grava d_bar_prod com stride = W1; a subtração abaixo lê qn
    // com stride W1 (sb=W1) — NÃO n_sum — senão os offsets por-candidato divergem.
    TSTART();
    ntt.carry_to_limbs(d_bar_prod, W1, s);
    TSTOP(qn->child(Q_CARRY));

    // Passo 3: out = (T − qn) mod N — finalize via subtractor tileado (ops/sub).
    //   (a) r = T − qn  (incondicional, W1 limbs → d_bar_w1). qn tem stride W1.
    TSTART();
    ops::sub_phase1(d_T, n_sum, d_bar_prod, W1, nullptr, W1,
                    d_cs_tile_cmp, d_cs_tile_bstate, n_batch, s);
    ops::sub_apply(d_bar_w1, W1, d_T, n_sum, d_bar_prod, W1, nullptr, W1,
                   d_cs_tile_cmp, d_cs_tile_bstate, /*uncond=*/1, n_batch, s);
    TSTOP(fin->child(FIN_SUB));

    //   (b) até 2 subtrações condicionais de N (in-place em r = d_bar_w1).
    TSTART();
    for (int it = 0; it < 2; it++)
    {
        ops::sub_phase1(d_bar_w1, W1, d_N, n_limbs, d_bar_k, W1,
                        d_cs_tile_cmp, d_cs_tile_bstate, n_batch, s);
        ops::sub_apply(d_bar_w1, W1, d_bar_w1, W1, d_N, n_limbs, d_bar_k, W1,
                       d_cs_tile_cmp, d_cs_tile_bstate, /*uncond=*/0, n_batch, s);
    }
    TSTOP(fin->child(FIN_CONDSUB));

    //   (c) out = r[0..n_limbs).
    TSTART();
    ops::copy_low(d_out, d_bar_w1, n_limbs, W1, n_batch, MR_THR_COPY, s);
    TSTOP(fin->child(FIN_COPY));
}

#endif // MOD_RED_BARRETT
