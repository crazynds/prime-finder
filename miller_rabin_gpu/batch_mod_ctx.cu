// batch_mod_ctx.cu — Núcleo comum do contexto de aritmética modular batched.
//
// Contém o que independe do algoritmo de redução: construção/destruição, conversão
// de/para a forma de trabalho (delegando a mod_residue_forward/backward), check_passed,
// e os drivers modmul/modsq (produto polinomial → reduce_batch). A redução em si vive
// em reduce_montgomery.cu / reduce_barrett.cu; o relatório de perfil em mod_perf.cu.

#include "batch_mod_ctx.cuh"
#include "gmp_helpers.cuh"
#include "mod_internal.cuh"
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

// Helpers para o construtor vector<mpz_t*> — usados na delegação.
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

    // Buffers comuns a todos os backends. (d_m é exclusivo do Montgomery e é
    // alocado em precompute_reduction; não desperdiça padded-bytes no Barrett.)
    CU(cudaMalloc(&d_N, nb));
    CU(cudaMalloc(&d_ntt_N, pb));
    CU(cudaMalloc(&d_T, sb));
    CU(cudaMalloc(&d_one_res, nb));
    CU(cudaMalloc(&d_Nm1_res, nb));

    CU(cudaMemcpy(d_N, N_all.data(), nb, cudaMemcpyHostToDevice));

    // NTT(N) pré-computado (usado por REDC e Barrett).
    ntt.ntt_A(d_N, n_limbs);
    CU(cudaMemcpy(d_ntt_N, ntt.d_buf_A, pb, cudaMemcpyDeviceToDevice));

    // Estruturas específicas do backend de redução (reduce_*.cu).
    precompute_reduction(N_all);
    // Não precisa de cudaDeviceSynchronize aqui: to_residue_batch() começa com um
    // cudaMemcpy(D2H) no default stream, que garante ordering implicitamente.

    // 1 e (N-1) na forma de trabalho (via GMP, uma única vez).
    std::vector<uint64_t> one_lims((size_t)n_batch * n_limbs, 0);
    for (int i = 0; i < n_batch; i++)
        one_lims[i * n_limbs] = 1;

    std::vector<uint64_t> Nm1_lims((size_t)n_batch * n_limbs, 0);
    for (int i = 0; i < n_batch; i++)
    {
        // Nm1 = N - 1: subtrai 1 com borrow.
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
            out[j] = 0xFFFF; // borrow
        }
    }

    std::vector<uint64_t> one_res_h, Nm1_res_h;
    to_residue_batch(one_lims, one_res_h);
    to_residue_batch(Nm1_lims, Nm1_res_h);
    CU(cudaMemcpy(d_one_res, one_res_h.data(), nb, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_Nm1_res, Nm1_res_h.data(), nb, cudaMemcpyHostToDevice));

    for (int i = 0; i <= PERF_RING; i++)
        CU(cudaEventCreate(&ev_ring[i]));
}

BatchModCtx::~BatchModCtx()
{
    cudaFree(d_N);
    cudaFree(d_ntt_N);
    cudaFree(d_T);
    cudaFree(d_one_res);
    cudaFree(d_Nm1_res);
    free_reduction();
    for (int i = 0; i <= PERF_RING; i++)
        if (ev_ring[i])
            cudaEventDestroy(ev_ring[i]);
}

// ── perfil ────────────────────────────────────────────────────────────────────

// Sincroniza o último evento gravado e acumula todos os tempos pendentes.
// Chamada UMA VEZ por modmul_batch / modsq_batch.
void BatchModCtx::perf_flush(cudaStream_t s)
{
    if (!perf_enabled || ring_cur == 0)
        return;
    CU(cudaEventSynchronize(ev_ring[ring_cur]));
    for (int i = 0; i < ring_cur; i++)
    {
        float ms = 0;
        CU(cudaEventElapsedTime(&ms, ev_ring[i], ev_ring[i + 1]));
        *acc_ring[i] += ms;
    }
    ring_cur = 0;
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

// out = A · B mod N (na forma de trabalho). Produto polinomial → reduce_batch.
void BatchModCtx::modmul_batch(const Data64 *d_A, const Data64 *d_B, Data64 *d_out,
                               cudaStream_t s)
{
    perf_cur = &perf_mul; // acumula tempos no contexto de multiplicação
    // Passo 1 — produto polinomial A · B → d_buf_A (raw, sem carry).
    TSTART();
#if MONT_MUL_ALG == MONT_MUL_ALG_NTT
    ntt.ntt_AB(d_A, d_B, n_limbs, s);
#elif MONT_MUL_ALG == MONT_MUL_ALG_SCHOOLBOOK
    ntt.schoolbook_mul(d_A, d_B, n_limbs, s);
#endif
    TSTOP(perf_cur->ntt_input);

#if MONT_MUL_ALG == MONT_MUL_ALG_NTT
    TSTART();
    ntt.pmul(s);
    TSTOP(perf_cur->pmul);
    TSTART();
    ntt.intt_A(s);
    TSTOP(perf_cur->intt_product);
#endif

    // Passo 2: normaliza T (carries) → d_T [n_batch * n_sum].
    TSTART();
    ntt.carry_to_limbs(d_T, n_sum, s);
    TSTOP(perf_cur->carry_product);

    // Passo 3: redução modular (backend) + finalização.
    reduce_batch(d_out, s);

    perf_flush(s); // único sync por chamada
}

// out = A² mod N. Usa psq (pointwise square) — economiza NTT(B) e metade das muls.
void BatchModCtx::modsq_batch(const Data64 *d_A, Data64 *d_out, cudaStream_t s)
{
    perf_cur = &perf_sq; // acumula tempos no contexto de quadrado
    TSTART();
#if MONT_MUL_ALG == MONT_MUL_ALG_NTT
    ntt.ntt_A(d_A, n_limbs, s);
#elif MONT_MUL_ALG == MONT_MUL_ALG_SCHOOLBOOK
    ntt.schoolbook_sq(d_A, n_limbs, s);
#endif
    TSTOP(perf_cur->ntt_input);

#if MONT_MUL_ALG == MONT_MUL_ALG_NTT
    TSTART();
    ntt.psq(s);
    TSTOP(perf_cur->pmul);
    TSTART();
    ntt.intt_A(s);
    TSTOP(perf_cur->intt_product);
#endif

    TSTART();
    ntt.carry_to_limbs(d_T, n_sum, s);
    TSTOP(perf_cur->carry_product);

    reduce_batch(d_out, s);

    perf_flush(s); // único sync por chamada
}

// ── multiplicações sem redução (benchmark) ────────────────────────────────────

// Apenas NTT(A)·NTT(B) + INTT + carry — sem redução modular.
void BatchModCtx::mul_no_redc_batch(const Data64 *d_A, const Data64 *d_B,
                                    Data64 *d_out, cudaStream_t s)
{
    ntt.ntt_AB(d_A, d_B, n_limbs, s);
    ntt.pmul_and_intt(s);
    ntt.carry_to_limbs(d_out, n_sum, s);
    cudaStreamSynchronize(s);
}

// Apenas NTT(A)^2 + INTT + carry — sem redução modular.
void BatchModCtx::sq_no_redc_batch(const Data64 *d_A, Data64 *d_out, cudaStream_t s)
{
    ntt.ntt_A(d_A, n_limbs, s);
    ntt.psq_and_intt(s);
    ntt.carry_to_limbs(d_out, n_sum, s);
    cudaStreamSynchronize(s);
}
