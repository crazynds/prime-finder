// reductions/montgomery.cu — Redução de Montgomery (REDC) batched + subtração condicional.
//
// Forma de trabalho: x·R mod N, R = 2^(LIMB_BITS·n_limbs). Todo o conteúdo é
// compilado apenas quando MOD_REDUCTION_ALG == MOD_RED_MONTGOMERY (params.cmake).

#include "batch_mod_ctx.cuh"
#include "helpers/gmp_helpers.cuh"
#include "helpers/timers.cuh"
#include "ops/shift/shift.cuh"
#include "ops/sub/sub.cuh"
#include <vector>
#include <gmp.h>

#if MOD_REDUCTION_ALG == MOD_RED_MONTGOMERY

// Índices locais (filhos de perf_cur->child(PERF_RED)):
namespace {
    enum MontRedIdx { MONTR_MUL = 0, MONTR_SOMA = 1, MONTR_SHIFT = 2 };
    // Filhos de MONTR_MUL:
    enum MontMulIdx { MONTM_NTT_TLOW = 0, MONTM_PMUL_NP = 1, MONTM_INTT_NP = 2,
                      MONTM_CARRY_M  = 3, MONTM_NTT_M   = 4, MONTM_PMUL_N  = 5,
                      MONTM_INTT_N   = 6 };
    // Filhos de MONTR_SOMA:
    enum MontSomaIdx { MONTS_VADD = 0, MONTS_CARRY = 1 };
}

// ── helpers GMP específicos ───────────────────────────────────────────────────

// N' = R - N^{-1} mod R (fator de correção do REDC). R = 2^(LIMB_BITS·n).
static void compute_Nprime(uint64_t *Np_out, const uint64_t *N_lims, int n)
{
    mpz_t N, R, Np;
    mpz_init(N);
    mpz_init(R);
    mpz_init(Np);
    limbs_to_mpz(N, N_lims, n);
    mpz_ui_pow_ui(R, 2, (unsigned long)(LIMB_BITS * n));
    if (!mpz_invert(Np, N, R))
        throw std::runtime_error("N nao tem inverso mod R");
    mpz_sub(Np, R, Np);
    mpz_to_limbs(Np_out, n, Np);
    mpz_clear(N);
    mpz_clear(R);
    mpz_clear(Np);
}

// Forma de trabalho Montgomery: res = x·R mod N.
void mod_residue_forward(mpz_t res, const mpz_t x, const mpz_t N, int n_limbs)
{
    mpz_t R;
    mpz_init(R);
    mpz_ui_pow_ui(R, 2, (unsigned long)(LIMB_BITS * n_limbs));
    mpz_mul(res, x, R);
    mpz_mod(res, res, N);
    mpz_clear(R);
}

// Saída da forma Montgomery: res = x·R^{-1} mod N.
void mod_residue_backward(mpz_t res, const mpz_t x, const mpz_t N, int n_limbs)
{
    mpz_t R, Rinv;
    mpz_init(R);
    mpz_init(Rinv);
    mpz_ui_pow_ui(R, 2, (unsigned long)(LIMB_BITS * n_limbs));
    mpz_invert(Rinv, R, N);
    mpz_mul(res, x, Rinv);
    mpz_mod(res, res, N);
    mpz_clear(R);
    mpz_clear(Rinv);
}

// ── setup/teardown do backend ─────────────────────────────────────────────────

void BatchModCtx::precompute_reduction(const std::vector<uint64_t> &N_all)
{
    const size_t nb = (size_t)n_batch * n_limbs * sizeof(Data64);
    const size_t pb = (size_t)n_batch * padded * sizeof(Data64);

    n_cs_tiles = ops::sub_n_tiles(n_limbs);
    const size_t csb = (size_t)n_batch * n_cs_tiles * sizeof(int);
    CU(cudaMalloc(&d_cs_tile_cmp, csb));
    CU(cudaMalloc(&d_cs_tile_bstate, csb));

    // Workspace de m (NTT) — exclusivo do REDC de Montgomery.
    CU(cudaMalloc(&d_m, pb));

    CU(cudaMalloc(&d_Nprime, nb));
    CU(cudaMalloc(&d_ntt_Nprime, pb));
    std::vector<uint64_t> Np_all((size_t)n_batch * n_limbs, 0);
    for (int i = 0; i < n_batch; i++)
        compute_Nprime(Np_all.data() + (size_t)i * n_limbs, N_all.data() + (size_t)i * n_limbs, n_limbs);
    CU(cudaMemcpy(d_Nprime, Np_all.data(), nb, cudaMemcpyHostToDevice));
    ntt.ntt_A(d_Nprime, n_limbs);
    CU(cudaMemcpy(d_ntt_Nprime, ntt.d_buf_A, pb, cudaMemcpyDeviceToDevice));
}

void BatchModCtx::free_reduction()
{
    cudaFree(d_m);
    cudaFree(d_Nprime);
    cudaFree(d_ntt_Nprime);
    cudaFree(d_cs_tile_cmp);
    cudaFree(d_cs_tile_bstate);
}

// ── redução ───────────────────────────────────────────────────────────────────

// Subtração condicional final do REDC: se x >= N, x -= N.
void BatchModCtx::cond_sub_batch(Data64 *d_x, cudaStream_t s)
{
    ops::sub_phase1(d_x, n_limbs, d_N, n_limbs, nullptr, n_limbs,
                    d_cs_tile_cmp, d_cs_tile_bstate, n_batch, s);
    ops::sub_apply(d_x, n_limbs, d_x, n_limbs, d_N, n_limbs, nullptr, n_limbs,
                   d_cs_tile_cmp, d_cs_tile_bstate, /*uncond=*/0, n_batch, s);
}

// Redução Montgomery (REDC): dado T = A·B em d_T [n_batch * n_sum],
// calcula out = T · R^{-1} mod N para cada candidato.
void BatchModCtx::reduce_batch(Data64 *d_out, cudaStream_t s)
{
    const int thr = MR_THR_REDUCE;

    PerfNode *red  = perf_cur->child(PERF_RED);
    PerfNode *mul  = red->child(MONTR_MUL);
    PerfNode *soma = red->child(MONTR_SOMA);
    PerfNode *csub = perf_cur->child(PERF_FIN);

    // Passo 1: m = (T mod R) · N' mod R. T_low = primeiros n_limbs limbs de T.
    TSTART();
    ops::extract_low(ntt.d_buf_A, d_T, n_limbs, padded, n_sum, n_batch, thr, s);
    ntt.fwd_A(s);
    TSTOP(mul->child(MONTM_NTT_TLOW));

    TSTART();
    ntt.pmul_ext(d_ntt_Nprime, s);
    TSTOP(mul->child(MONTM_PMUL_NP));
    TSTART();
    ntt.intt_A(s);
    TSTOP(mul->child(MONTM_INTT_NP));
    TSTART();
    ntt.carry_to_limbs(d_m, n_limbs, s);
    TSTOP(mul->child(MONTM_CARRY_M));

    // Passo 2: mN = m · N.
    TSTART();
    ntt.ntt_A(d_m, n_limbs, s);
    TSTOP(mul->child(MONTM_NTT_M));
    TSTART();
    ntt.pmul_ext(d_ntt_N, s);
    TSTOP(mul->child(MONTM_PMUL_N));
    TSTART();
    ntt.intt_A(s);
    TSTOP(mul->child(MONTM_INTT_N));

    // Passo 3: T += mN, normaliza carries.
#if CARRY_NORM_ALG == CARRY_ALG_SEQUENTIAL
    TSTART();
    ntt.add_raw_buf_and_carry(d_T, n_sum, s);
    TSTOP(soma->child(MONTS_CARRY));
#else
    TSTART();
    ntt.vadd_raw_buf(d_T, n_sum, s);
    TSTOP(soma->child(MONTS_VADD));
    TSTART();
    ntt.carry_after_vadd(d_T, n_sum, s);
    TSTOP(soma->child(MONTS_CARRY));
#endif

    // Passo 4: out = (T + mN) / R = shift direito por n_limbs posições.
    TSTART();
    ops::shift_right(d_out, d_T, n_limbs, n_limbs, n_sum, n_batch, thr, s);
    TSTOP(red->child(MONTR_SHIFT));

    // Passo 5: subtração condicional — garante out < N.
    TSTART();
    cond_sub_batch(d_out, s);
    TSTOP(csub);
}

#endif // MOD_RED_MONTGOMERY
