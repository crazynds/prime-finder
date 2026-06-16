// bigint_ntt.cu
#include "config.h"
#include "ops/mul/ntt_merge.cuh"
#include <stdexcept>
#include <string>

#define CU(expr)                                                                                  \
    do                                                                                            \
    {                                                                                             \
        cudaError_t _e = (expr);                                                                  \
        if (_e != cudaSuccess)                                                                    \
            throw std::runtime_error(std::string("[CUDA] " #expr ": ") + cudaGetErrorString(_e)); \
    } while (0)

// CARRY_TILE: usado pelo construtor para dimensionar d_tile_carry.
// Os algoritmos de carry em si vivem em carry_norm.cu.
static constexpr int CARRY_TILE = MR_CARRY_TILE;

// ── kernels NTT ───────────────────────────────────────────────────────────────

__global__ static void load_padded_batch(Data64 *__restrict__ dst,
                                         const Data64 *__restrict__ src,
                                         int n_src, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= padded)
        return;
    dst[cand * padded + j] = (j < n_src) ? src[cand * n_src + j] : 0ULL;
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

// ── BigIntNTTBatch ────────────────────────────────────────────────────────────

BigIntNTTBatch::BigIntNTTBatch(int n_limbs_, int n_batch_)
    : n_limbs(n_limbs_), padded(next_pow2_ntt(2 * n_limbs_)), logn(__builtin_ctz(next_pow2_ntt(2 * n_limbs_))), n_batch(n_batch_)
{
    NTTParameters<Data64> params(logn, ReductionPolynomial::X_N_minus);
    p_val = params.modulus.value;
    n_inv = params.n_inv;
    modulus = params.modulus;

    auto fwd_h = params.gpu_root_of_unity_table_generator(params.forward_root_of_unity_table);
    auto inv_h = params.gpu_root_of_unity_table_generator(params.inverse_root_of_unity_table);

    const size_t tbytes = fwd_h.size() * sizeof(Root64);
    const size_t pbytes = (size_t)n_batch * padded * sizeof(Data64);
    int n_tiles_max = (padded + CARRY_TILE - 1) / CARRY_TILE;
    const size_t tcbytes = (size_t)n_batch * n_tiles_max * sizeof(Data64);

    CU(cudaMalloc(&d_fwd_table, tbytes));
    CU(cudaMalloc(&d_inv_table, tbytes));
    CU(cudaMalloc(&d_buf_AB, 2 * pbytes));
    d_buf_A = d_buf_AB;
    d_buf_B = d_buf_AB + (size_t)n_batch * padded;
    CU(cudaMalloc(&d_tile_carry, tcbytes));

    CU(cudaMemcpy(d_fwd_table, fwd_h.data(), tbytes, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_inv_table, inv_h.data(), tbytes, cudaMemcpyHostToDevice));
}

BigIntNTTBatch::~BigIntNTTBatch()
{
    cudaFree(d_fwd_table);
    cudaFree(d_inv_table);
    cudaFree(d_buf_AB);
    cudaFree(d_tile_carry);
}

ntt_configuration<Data64> BigIntNTTBatch::make_cfg(type t, cudaStream_t s)
{
    return {
        .n_power = logn,
        .ntt_type = t,
        .ntt_layout = (logn >= 10) ? PerPolynomial : GPUNTT_NTT_LAYOUT,
        .reduction_poly = ReductionPolynomial::X_N_minus,
        .zero_padding = false,
        .mod_inverse = n_inv,
        .stream = s};
}

void BigIntNTTBatch::ntt_A(const Data64 *d_src, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(padded + thr - 1) / thr;
    load_padded_batch<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(
        d_buf_A, d_src, n_src, padded, n_batch);
    GPU_NTT_Inplace(d_buf_A, d_fwd_table, modulus, make_cfg(FORWARD, s), n_batch);
}

void BigIntNTTBatch::ntt_B(const Data64 *d_src, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(padded + thr - 1) / thr;
    load_padded_batch<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(
        d_buf_B, d_src, n_src, padded, n_batch);
    GPU_NTT_Inplace(d_buf_B, d_fwd_table, modulus, make_cfg(FORWARD, s), n_batch);
}

void BigIntNTTBatch::ntt_AB(const Data64 *d_srcA, const Data64 *d_srcB, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_LOAD;
    unsigned bx = (unsigned)(padded + thr - 1) / thr;
    load_padded_batch<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(
        d_buf_A, d_srcA, n_src, padded, n_batch);
    load_padded_batch<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(
        d_buf_B, d_srcB, n_src, padded, n_batch);
    GPU_NTT_Inplace(d_buf_A, d_fwd_table, modulus, make_cfg(FORWARD, s), 2 * n_batch);
}

void BigIntNTTBatch::fwd_A(cudaStream_t s)
{
    GPU_NTT_Inplace(d_buf_A, d_fwd_table, modulus, make_cfg(FORWARD, s), n_batch);
}

void BigIntNTTBatch::pmul(cudaStream_t s)
{
    int total = n_batch * padded;
    constexpr int thr = MR_THR_PMUL;
    const int blk = (total + thr - 1) / thr;
    pmul_batch<<<blk, thr, 0, s>>>(d_buf_A, d_buf_B, total, p_val);
}

void BigIntNTTBatch::psq(cudaStream_t s)
{
    int total = n_batch * padded;
    constexpr int thr = MR_THR_PMUL;
    const int blk = (total + thr - 1) / thr;
    psq_batch<<<blk, thr, 0, s>>>(d_buf_A, total, p_val);
}

void BigIntNTTBatch::pmul_ext(const Data64 *d_ext, cudaStream_t s)
{
    int total = n_batch * padded;
    constexpr int thr = MR_THR_PMUL;
    const int blk = (total + thr - 1) / thr;
    pmul_batch<<<blk, thr, 0, s>>>(d_buf_A, d_ext, total, p_val);
}

void BigIntNTTBatch::intt_A(cudaStream_t s)
{
    GPU_INTT_Inplace(d_buf_A, d_inv_table, modulus, make_cfg(INVERSE, s), n_batch);
}

void BigIntNTTBatch::pmul_and_intt(cudaStream_t s)
{
    pmul(s);
    intt_A(s);
}

void BigIntNTTBatch::pmul_ext_and_intt(const Data64 *d_ext, cudaStream_t s)
{
    pmul_ext(d_ext, s);
    intt_A(s);
}

void BigIntNTTBatch::psq_and_intt(cudaStream_t s)
{
    psq(s);
    intt_A(s);
}

// ── Schoolbook (MONT_MUL_ALG_SCHOOLBOOK) ─────────────────────────────────────
//
// Convolução polinomial direta O(n²): cada thread calcula um coeficiente de saída.
// Escreve em d_buf_A com stride=padded — compatível com carry_to_limbs().
//
// Thread = um coeficiente de saída j ∈ [0, padded).
// Loop interno soma A[i]*B[j-i] para i em [max(0,j-n+1), min(j+1,n)).
// Overflow: A[i],B[j-i] ≤ 2^16-1, loop ≤ n iterações, acc ≤ n*(2^16)² < 2^49 para
// n ≤ 2^17. Cabe em uint64. ✓

__global__ static void schoolbook_mul_kernel(
    Data64 *__restrict__ d_buf_A,
    const Data64 *__restrict__ d_A,
    const Data64 *__restrict__ d_B,
    int n_limbs, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= padded)
        return;

    if (j >= 2 * n_limbs)
    {
        d_buf_A[(size_t)cand * padded + j] = 0ULL;
        return;
    }

    const Data64 *A = d_A + (size_t)cand * n_limbs;
    const Data64 *B = d_B + (size_t)cand * n_limbs;
    uint64_t acc = 0;
    int i_lo = (j >= n_limbs) ? j - n_limbs + 1 : 0;
    int i_hi = (j < n_limbs) ? j + 1 : n_limbs;
    for (int i = i_lo; i < i_hi; i++)
        acc += A[i] * B[j - i];
    d_buf_A[(size_t)cand * padded + j] = acc;
}

// Squaring schoolbook com otimização de simetria: pares (i, j-i) contam 2x,
// exceto o termo do meio quando j é par.
__global__ static void schoolbook_sq_kernel(
    Data64 *__restrict__ d_buf_A,
    const Data64 *__restrict__ d_A,
    int n_limbs, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= padded)
        return;

    if (j >= 2 * n_limbs)
    {
        d_buf_A[(size_t)cand * padded + j] = 0ULL;
        return;
    }

    const Data64 *A = d_A + (size_t)cand * n_limbs;
    uint64_t acc = 0;
    int i_lo = (j >= n_limbs) ? j - n_limbs + 1 : 0;
    int i_hi_excl = (j < n_limbs) ? j + 1 : n_limbs;
    // Pares (i, j-i) com i < j-i  →  contribuem 2*A[i]*A[j-i]
    int i_cross = (j + 1) / 2; // primeiro i onde 2*i >= j
    for (int i = i_lo; i < i_cross && i < i_hi_excl; i++)
        acc += 2ULL * A[i] * A[j - i];
    // Termo do meio (j par, i == j/2)
    if (j % 2 == 0)
    {
        int m = j / 2;
        if (m >= i_lo && m < i_hi_excl)
            acc += A[m] * A[m];
    }
    d_buf_A[(size_t)cand * padded + j] = acc;
}

void BigIntNTTBatch::schoolbook_mul(const Data64 *d_A, const Data64 *d_B, int n_src,
                                    cudaStream_t s)
{
    constexpr int thr = MR_THR_PMUL;
    unsigned bx = (unsigned)(padded + thr - 1) / thr;
    schoolbook_mul_kernel<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(
        d_buf_A, d_A, d_B, n_src, padded, n_batch);
}

void BigIntNTTBatch::schoolbook_sq(const Data64 *d_A, int n_src, cudaStream_t s)
{
    constexpr int thr = MR_THR_PMUL;
    unsigned bx = (unsigned)(padded + thr - 1) / thr;
    schoolbook_sq_kernel<<<dim3(bx, (unsigned)n_batch), thr, 0, s>>>(
        d_buf_A, d_A, n_src, padded, n_batch);
}

// ─────────────────────────────────────────────────────────────────────────────
