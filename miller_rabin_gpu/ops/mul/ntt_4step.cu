// ops/mul/ntt_4step.cu — implementação do backend NTT "4-step" (radix) da GPU-NTT.
//
// Pipeline (ver exemplos da lib em example/ntt_4step/):
//   forward(src→dst):  Transpose(src→dst) · 4STEP_NTT(dst→src,FWD) · Transpose(src→dst)
//                      → resultado em dst (3 ops, troca de buffer).
//   inverse(buf,other): 4STEP_NTT(buf→other,INV) · Transpose(other→buf)
//                      → resultado em buf (2 ops).
//
// Usa a variante RNS de GPU_4STEP_NTT (modulus/ninverse no device, mod_count=1),
// que é o caminho exercitado pelos testes/benchmarks da própria GPU-NTT.

#include "config.h"
#include "ops/mul/ntt_4step.cuh"
#include "ops/mul/ntt_check.cuh"
#include "gpuntt/ntt_4step/ntt_4step_cpu.cuh" // NTTParameters4Step usa esse cabeçalho de geração
#include <stdexcept>
#include <string>

#define CU(expr)                                                                                  \
    do                                                                                            \
    {                                                                                             \
        cudaError_t _e = (expr);                                                                  \
        if (_e != cudaSuccess)                                                                    \
            throw std::runtime_error(std::string("[CUDA] " #expr ": ") + cudaGetErrorString(_e)); \
    } while (0)

static constexpr int CARRY_TILE = MR_CARRY_TILE;

// ── kernels (mesmos do backend merge; o domínio transformado é elementwise) ────

__global__ static void load_padded_batch(Data64 *__restrict__ dst,
                                          const Data64 *__restrict__ src,
                                          int n_src, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= padded)
        return;
    dst[(size_t)cand * padded + j] = (j < n_src) ? src[(size_t)cand * n_src + j] : 0ULL;
}

__global__ static void pmul_batch(Data64 *__restrict__ a, const Data64 *__restrict__ b,
                                  int total, Data64 p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    a[i] = (Data64)((__uint128_t)a[i] * b[i] % (__uint128_t)p);
}

__global__ static void psq_batch(Data64 *__restrict__ a, int total, Data64 p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    a[i] = (Data64)((__uint128_t)a[i] * a[i] % (__uint128_t)p);
}

__global__ static void schoolbook_mul_kernel(
    Data64 *__restrict__ d_buf_A, const Data64 *__restrict__ d_A,
    const Data64 *__restrict__ d_B, int n_limbs, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= padded)
        return;
    if (j >= 2 * n_limbs) { d_buf_A[(size_t)cand * padded + j] = 0ULL; return; }
    const Data64 *A = d_A + (size_t)cand * n_limbs;
    const Data64 *B = d_B + (size_t)cand * n_limbs;
    uint64_t acc = 0;
    int i_lo = (j >= n_limbs) ? j - n_limbs + 1 : 0;
    int i_hi = (j < n_limbs) ? j + 1 : n_limbs;
    for (int i = i_lo; i < i_hi; i++)
        acc += A[i] * B[j - i];
    d_buf_A[(size_t)cand * padded + j] = acc;
}

__global__ static void schoolbook_sq_kernel(
    Data64 *__restrict__ d_buf_A, const Data64 *__restrict__ d_A,
    int n_limbs, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= padded)
        return;
    if (j >= 2 * n_limbs) { d_buf_A[(size_t)cand * padded + j] = 0ULL; return; }
    const Data64 *A = d_A + (size_t)cand * n_limbs;
    uint64_t acc = 0;
    int i_lo = (j >= n_limbs) ? j - n_limbs + 1 : 0;
    int i_hi_excl = (j < n_limbs) ? j + 1 : n_limbs;
    int i_cross = (j + 1) / 2;
    for (int i = i_lo; i < i_cross && i < i_hi_excl; i++)
        acc += 2ULL * A[i] * A[j - i];
    if (j % 2 == 0)
    {
        int m = j / 2;
        if (m >= i_lo && m < i_hi_excl)
            acc += A[m] * A[m];
    }
    d_buf_A[(size_t)cand * padded + j] = acc;
}

// ── construtor / destrutor ─────────────────────────────────────────────────────

// O 4-step da GPU-NTT só suporta logn ∈ [12,24]. Para n pequeno fazemos
// over-padding até 2^12 (transformada maior que o necessário; o produto cabe e a
// correção se mantém). O BatchModCtx adota este `padded` (ver ctor do ctx).
static int clamp_padded_4step(int n_limbs)
{
    int p = next_pow2_ntt(2 * n_limbs);
    if (p < (1 << 12))
        p = (1 << 12);
    return p;
}

Ntt4StepBatch::Ntt4StepBatch(int n_limbs_, int n_batch_)
    : n_limbs(n_limbs_),
      padded(clamp_padded_4step(n_limbs_)),
      logn(__builtin_ctz(clamp_padded_4step(n_limbs_))),
      n_batch(n_batch_)
{
    if (logn > 24)
        throw std::runtime_error(
            "[ntt_4step] logn=" + std::to_string(logn) +
            " > 24 (limitação do backend 4-step da GPU-NTT). Use MUL_NTT_MERGE.");

    NTTParameters4Step<Data64> params(logn, ReductionPolynomial::X_N_minus);
    p_val = params.modulus.value;
    n_inv = params.n_inv;
    modulus = params.modulus;
    n1 = params.n1;
    n2 = params.n2;

    check_ntt_precision(padded, p_val);

    // Tabelas n1/n2 via gerador; tabela W copiada direta (mesma convenção do exemplo).
    auto h_n1f = params.gpu_root_of_unity_table_generator(params.n1_based_root_of_unity_table);
    auto h_n2f = params.gpu_root_of_unity_table_generator(params.n2_based_root_of_unity_table);
    auto h_n1i = params.gpu_root_of_unity_table_generator(params.n1_based_inverse_root_of_unity_table);
    auto h_n2i = params.gpu_root_of_unity_table_generator(params.n2_based_inverse_root_of_unity_table);

    const size_t n1b = (size_t)(n1 >> 1) * sizeof(Root64);
    const size_t n2b = (size_t)(n2 >> 1) * sizeof(Root64);
    const size_t wb = (size_t)padded * sizeof(Root64);

    CU(cudaMalloc(&d_n1_fwd, n1b));
    CU(cudaMalloc(&d_n2_fwd, n2b));
    CU(cudaMalloc(&d_W_fwd, wb));
    CU(cudaMalloc(&d_n1_inv, n1b));
    CU(cudaMalloc(&d_n2_inv, n2b));
    CU(cudaMalloc(&d_W_inv, wb));

    CU(cudaMemcpy(d_n1_fwd, h_n1f.data(), n1b, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_n2_fwd, h_n2f.data(), n2b, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_W_fwd, params.W_root_of_unity_table.data(), wb, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_n1_inv, h_n1i.data(), n1b, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_n2_inv, h_n2i.data(), n2b, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_W_inv, params.W_inverse_root_of_unity_table.data(), wb, cudaMemcpyHostToDevice));

    // modulus / ninverse no device (RNS, mod_count=1).
    CU(cudaMalloc(&d_modulus, sizeof(Modulus<Data64>)));
    CU(cudaMalloc(&d_ninverse, sizeof(Ninverse64)));
    Modulus<Data64> mod_h[1] = {modulus};
    Ninverse64 ninv_h[1] = {n_inv};
    CU(cudaMemcpy(d_modulus, mod_h, sizeof(Modulus<Data64>), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_ninverse, ninv_h, sizeof(Ninverse64), cudaMemcpyHostToDevice));

    const size_t pb = (size_t)n_batch * padded * sizeof(Data64);
    CU(cudaMalloc(&d_buf_AB, 2 * pb));
    d_buf_A = d_buf_AB;
    d_buf_B = d_buf_AB + (size_t)n_batch * padded;
    CU(cudaMalloc(&d_scratch, 2 * pb));

    int n_tiles_max = (padded + CARRY_TILE - 1) / CARRY_TILE;
    CU(cudaMalloc(&d_tile_carry, (size_t)n_batch * n_tiles_max * sizeof(Data64)));
}

Ntt4StepBatch::~Ntt4StepBatch()
{
    cudaFree(d_n1_fwd);
    cudaFree(d_n2_fwd);
    cudaFree(d_W_fwd);
    cudaFree(d_n1_inv);
    cudaFree(d_n2_inv);
    cudaFree(d_W_inv);
    cudaFree(d_modulus);
    cudaFree(d_ninverse);
    cudaFree(d_buf_AB);
    cudaFree(d_scratch);
    cudaFree(d_tile_carry);
}

// ── transformadas (ping-pong) ──────────────────────────────────────────────────

// Transformada 4-step completa: T · NTT · T (3 ops). Entrada em `src`, resultado
// em `dst` (src vira scratch).
//
// As dims do PRIMEIRO transpose diferem por direção (derivado do CPU da lib):
//   forward  → vector_to_matrix(n1,n2)  ⇒ GPU_Transpose(n1, n2)
//   inverse  → vector_to_matrix_intt    ⇒ GPU_Transpose(n2, n1)  (dims trocadas!)
// O último transpose é (n1,n2) nas duas direções.
void Ntt4StepBatch::transform(Data64 *src, Data64 *dst, bool fwd, int batch, cudaStream_t s)
{
    ntt4step_rns_configuration<Data64> cfg = {
        .n_power = logn,
        .ntt_type = fwd ? FORWARD : INVERSE,
        .mod_inverse = d_ninverse,
        .stream = s};
    Root64 *n1t = fwd ? d_n1_fwd : d_n1_inv;
    Root64 *n2t = fwd ? d_n2_fwd : d_n2_inv;
    Root64 *Wt = fwd ? d_W_fwd : d_W_inv;
    int t1r = fwd ? n1 : n2; // row do 1º transpose
    int t1c = fwd ? n2 : n1; // col do 1º transpose
    GPU_Transpose(src, dst, t1r, t1c, logn, batch);
    GPU_4STEP_NTT(dst, src, n1t, n2t, Wt, d_modulus, cfg, batch, 1);
    GPU_Transpose(src, dst, n1, n2, logn, batch);
}

void Ntt4StepBatch::ntt_A(const Data64 *d_src, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(padded + thr - 1) / thr;
    load_padded_batch<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(
        d_scratch, d_src, n_src, padded, n_batch);
    transform(d_scratch, d_buf_A, /*fwd=*/true, n_batch, s);
}

void Ntt4StepBatch::ntt_B(const Data64 *d_src, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(padded + thr - 1) / thr;
    Data64 *scr_B = d_scratch + (size_t)n_batch * padded;
    load_padded_batch<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(
        scr_B, d_src, n_src, padded, n_batch);
    transform(scr_B, d_buf_B, /*fwd=*/true, n_batch, s);
}

void Ntt4StepBatch::ntt_AB(const Data64 *d_srcA, const Data64 *d_srcB, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(padded + thr - 1) / thr;
    Data64 *scr_B = d_scratch + (size_t)n_batch * padded;
    load_padded_batch<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(
        d_scratch, d_srcA, n_src, padded, n_batch);
    load_padded_batch<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(
        scr_B, d_srcB, n_src, padded, n_batch);
    // A e B contíguos em d_scratch e d_buf_AB → transforma os 2*n_batch de uma vez.
    transform(d_scratch, d_buf_AB, /*fwd=*/true, 2 * n_batch, s);
}

void Ntt4StepBatch::fwd_A(cudaStream_t s)
{
    // Dados já em d_buf_A. transform termina em d_scratch → copia de volta.
    transform(d_buf_A, d_scratch, /*fwd=*/true, n_batch, s);
    CU(cudaMemcpyAsync(d_buf_A, d_scratch, (size_t)n_batch * padded * sizeof(Data64),
                       cudaMemcpyDeviceToDevice, s));
}

void Ntt4StepBatch::intt_A(cudaStream_t s)
{
    // Inverse simétrica (3 ops); termina em d_scratch → copia de volta p/ d_buf_A.
    transform(d_buf_A, d_scratch, /*fwd=*/false, n_batch, s);
    CU(cudaMemcpyAsync(d_buf_A, d_scratch, (size_t)n_batch * padded * sizeof(Data64),
                       cudaMemcpyDeviceToDevice, s));
}

// ── pointwise ───────────────────────────────────────────────────────────────────

void Ntt4StepBatch::pmul(cudaStream_t s)
{
    int total = n_batch * padded;
    constexpr int thr = MR_THR_PMUL;
    const int blk = (total + thr - 1) / thr;
    pmul_batch<<<blk, thr, 0, s>>>(d_buf_A, d_buf_B, total, p_val);
}

void Ntt4StepBatch::psq(cudaStream_t s)
{
    int total = n_batch * padded;
    constexpr int thr = MR_THR_PMUL;
    const int blk = (total + thr - 1) / thr;
    psq_batch<<<blk, thr, 0, s>>>(d_buf_A, total, p_val);
}

void Ntt4StepBatch::pmul_ext(const Data64 *d_ext, cudaStream_t s)
{
    int total = n_batch * padded;
    constexpr int thr = MR_THR_PMUL;
    const int blk = (total + thr - 1) / thr;
    pmul_batch<<<blk, thr, 0, s>>>(d_buf_A, d_ext, total, p_val);
}

void Ntt4StepBatch::pmul_and_intt(cudaStream_t s) { pmul(s); intt_A(s); }
void Ntt4StepBatch::psq_and_intt(cudaStream_t s) { psq(s); intt_A(s); }
void Ntt4StepBatch::pmul_ext_and_intt(const Data64 *d_ext, cudaStream_t s) { pmul_ext(d_ext, s); intt_A(s); }

// ── schoolbook ───────────────────────────────────────────────────────────────────

void Ntt4StepBatch::schoolbook_mul(const Data64 *d_A, const Data64 *d_B, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_PMUL;
    unsigned bx = (unsigned)(padded + thr - 1) / thr;
    schoolbook_mul_kernel<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(
        d_buf_A, d_A, d_B, n_src, padded, n_batch);
}

void Ntt4StepBatch::schoolbook_sq(const Data64 *d_A, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_PMUL;
    unsigned bx = (unsigned)(padded + thr - 1) / thr;
    schoolbook_sq_kernel<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(
        d_buf_A, d_A, n_src, padded, n_batch);
}
