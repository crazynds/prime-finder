// bigint_ntt.cu
#include "config.h"
#include "bigint_ntt.cuh"
#include <stdexcept>
#include <string>

#define CU(expr)                                                                                  \
    do                                                                                            \
    {                                                                                             \
        cudaError_t _e = (expr);                                                                  \
        if (_e != cudaSuccess)                                                                    \
            throw std::runtime_error(std::string("[CUDA] " #expr ": ") + cudaGetErrorString(_e)); \
    } while (0)

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

__global__ static void vadd_batch(
    Data64 *__restrict__ d_c,
    const Data64 *__restrict__ d_a,
    const Data64 *__restrict__ d_b,
    int n, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= n)
        return;
    d_c[cand * n + j] = d_a[cand * n + j] + d_b[cand * n + j];
}

#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE || CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE
// Soma d_buf_A (stride=padded, raw INTT) em d_dst (stride=n_dst) element-wise.
__global__ static void vadd_from_raw_batch(
    Data64 *__restrict__ d_dst,
    const Data64 *__restrict__ d_raw,
    int n_dst, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= n_dst)
        return;
    if (j < padded)
        d_dst[cand * n_dst + j] += d_raw[cand * padded + j];
}
#endif

// ── Algoritmos de carry normalização ─────────────────────────────────────────
//
// Selecione via CARRY_NORM_ALG em config.h:
//   CARRY_ALG_SINGLE_TILE — 1 bloco/candidato, CARRY_TILE threads, shared-mem carry
//   CARRY_ALG_MULTI_TILE  — intra-tile paralelo + inter-tile sequencial (2 kernels)
//   CARRY_ALG_SEQUENTIAL  — 1 thread/candidato, loop sequencial puro

static constexpr int CARRY_TILE = MR_CARRY_TILE;

// ── CARRY_ALG_SINGLE_TILE ────────────────────────────────────────────────────
// CARRY_TILE deve ser exatamente 32 (um warp) para usar __ballot_sync /
// __shfl_up_sync e eliminar has_carry de shared memory.
#if CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE

static_assert(CARRY_TILE == 32, "CARRY_ALG_SINGLE_TILE requer CARRY_TILE == 32 (um warp)");

#ifdef MR_ADVANCED_MONITOR
__device__ unsigned long long g_for_count = 0;
__device__ unsigned long long g_dowhile_count = 0;
#endif

__global__ static void carry_16bits(
    Data64 *d_src,
    Data64 *d_dst,
    int n, int src_stride, int n_batch)
{
    int tid = threadIdx.x;
    int cand = blockIdx.x;
    if (cand >= n_batch)
        return;
    int src_offset = cand * src_stride;
    int dst_offset = cand * n;

    Data64 tile_carry = 0;
#ifdef MR_ADVANCED_MONITOR
    unsigned long long local_for = 0, local_dowhile = 0;
#endif

    for (int tile = tid; tile < n; tile += CARRY_TILE)
    {
#ifdef MR_ADVANCED_MONITOR
        if (tid == 0)
            local_for++;
#endif
        Data64 currVal = d_src[src_offset + tile];
        Data64 c = (tid == 0) ? tile_carry : 0ULL;
        Data64 escape = 0;

        unsigned ballot;
        do
        {
#ifdef MR_ADVANCED_MONITOR
            if (tid == 0)
                local_dowhile++;
#endif
            c += currVal;
            currVal = c & LIMB_MASK;
            c >>= LIMB_BITS;

            escape += __shfl_sync(0xFFFFFFFFu, c, CARRY_TILE - 1);
            c = (tid == CARRY_TILE - 1) ? 0 : c;

            Data64 from_left = __shfl_up_sync(0xFFFFFFFFu, c, 1);
            c = (tid > 0) ? from_left : 0ULL;

            ballot = __ballot_sync(0xFFFFFFFFu, c > 0);
        } while (ballot);

        tile_carry = escape;

        d_dst[dst_offset + tile] = currVal;
    }

#ifdef MR_ADVANCED_MONITOR
    if (tid == 0)
    {
        atomicAdd(&g_for_count, local_for);
        atomicAdd(&g_dowhile_count, local_dowhile);
    }
#endif
}

// ── CARRY_ALG_MULTI_TILE ─────────────────────────────────────────────────────
#elif CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE

// Fase 1 — copia src→dst e normaliza carries intra-tile em paralelo.
// Cada bloco (tile, cand) processa CARRY_TILE elementos independentemente.
// O carry que escapa do tile é salvo em d_tile_carry[cand*n_tiles + tile].
__global__ static void carry_intra_copy(
    Data64 *__restrict__ d_dst,
    const Data64 *__restrict__ d_src,
    Data64 *__restrict__ d_tile_carry,
    int n_dst, int n_src, int n_batch)
{
    int cand = blockIdx.y, tile = blockIdx.x, tid = threadIdx.x;
    if (cand >= n_batch)
        return;

    int n_tiles = (n_dst + CARRY_TILE - 1) / CARRY_TILE;
    int j_start = tile * CARRY_TILE;

    int j = j_start + tid;
    __shared__ Data64 carry[CARRY_TILE + 1];
    __shared__ int has_carry[2];
    if (tid == 0)
    {
        carry[CARRY_TILE] = 0;
        has_carry[0] = false;
        has_carry[1] = false;
    }
    carry[tid] = ((j < n_src) ? d_src[cand * n_src + j] : 0ULL);
    __syncthreads();
    Data64 currVal = 0;
    int currIter = 0;
    do
    {
        Data64 v = carry[tid];
        currIter = currIter ^ 1;
        carry[tid] = 0;
        if (tid == 0)
        {
            has_carry[currIter] = false;
        }
        __syncthreads();
        v += currVal;
        currVal = v & LIMB_MASK;
        v >>= LIMB_BITS;
        if (v > 0)
            has_carry[currIter] = true;
        carry[tid + 1] += v;
        __syncthreads();
    } while (has_carry[currIter]);

    if (j < n_dst)
        d_dst[cand * n_dst + j] = currVal;
    if (tid == 0)
        d_tile_carry[cand * n_tiles + tile] = carry[CARRY_TILE];
}

// Fase 2 — propaga carries entre tiles sequencialmente (1 thread por candidato).
__global__ static void carry_inter_tiles(
    Data64 *__restrict__ d_dst,
    Data64 *__restrict__ d_tile_carry,
    int n, int n_batch)
{
    int cand = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch)
        return;

    int n_tiles = (n + CARRY_TILE - 1) / CARRY_TILE;
    for (int t = 0; t < n_tiles - 1; t++)
    {
        Data64 c = d_tile_carry[cand * n_tiles + t];
        if (c == 0)
            continue;
        int j_start = (t + 1) * CARRY_TILE;
        int j_end = min(j_start + CARRY_TILE, n);
        for (int j = j_start; c > 0 && j < j_end; j++)
        {
            Data64 v = d_dst[cand * n + j] + c;
            d_dst[cand * n + j] = v & LIMB_MASK;
            c = v >> LIMB_BITS;
        }
        // Carry que escapa do tile seguinte — raro mas possível
        if (c > 0 && j_end < n)
            d_tile_carry[cand * n_tiles + t + 1] += c;
    }
}

// ── CARRY_ALG_SEQUENTIAL ─────────────────────────────────────────────────────
#elif CARRY_NORM_ALG == CARRY_ALG_SEQUENTIAL

// 1 thread por candidato — copia d_src (stride=n_src) → d_dst (stride=n_dst)
// normalizando carries sequencialmente.
__global__ static void carry_sequential(
    Data64 *__restrict__ d_dst,
    const Data64 *__restrict__ d_src,
    int n_dst, int n_src, int n_batch)
{
    int cand = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch)
        return;

    Data64 carry = 0;
    for (int j = 0; j < n_dst; j++)
    {
        Data64 v = (j < n_src ? d_src[cand * n_src + j] : 0ULL) + carry;
        d_dst[cand * n_dst + j] = v & LIMB_MASK;
        carry = v >> LIMB_BITS;
    }
}

// Versão fundida para add_raw_buf_and_carry: soma d_raw (raw INTT, stride=padded)
// em d_dst e normaliza carries em uma única passagem sequencial por candidato.
__global__ static void vadd_carry_sequential(
    Data64 *__restrict__ d_dst,
    const Data64 *__restrict__ d_raw,
    int n_dst, int padded, int n_batch)
{
    int cand = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch)
        return;

    Data64 carry = 0;
    for (int j = 0; j < n_dst; j++)
    {
        Data64 raw = (j < padded ? d_raw[cand * padded + j] : 0ULL);
        Data64 v = d_dst[cand * n_dst + j] + raw + carry;
        d_dst[cand * n_dst + j] = v & LIMB_MASK;
        carry = v >> LIMB_BITS;
    }
}

#else
#error "CARRY_NORM_ALG deve ser CARRY_ALG_SINGLE_TILE, CARRY_ALG_MULTI_TILE ou CARRY_ALG_SEQUENTIAL"
#endif

// ── Monitor avançado: stats globais de carry ─────────────────────────────────

void carry_stats_print_and_reset()
{
#ifdef MR_ADVANCED_MONITOR
    unsigned long long h_for, h_dowhile;
    cudaMemcpyFromSymbol(&h_for, g_for_count, sizeof(h_for));
    cudaMemcpyFromSymbol(&h_dowhile, g_dowhile_count, sizeof(h_dowhile));
    if (h_for > 0)
        printf("[carry_16bits] for=%llu  do-while=%llu  media=%.3f iter/tile\n",
               h_for, h_dowhile, (double)h_dowhile / (double)h_for);
    unsigned long long zero = 0;
    cudaMemcpyToSymbol(g_for_count, &zero, sizeof(zero));
    cudaMemcpyToSymbol(g_dowhile_count, &zero, sizeof(zero));
#endif
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

void BigIntNTTBatch::carry_to_limbs(Data64 *d_out, int n_out, cudaStream_t s)
{
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    constexpr int THR = MR_CARRY_INTER_THR;
    int n_tiles = (n_out + CARRY_TILE - 1) / CARRY_TILE;
    int inter_blk = (n_batch + THR - 1) / THR;
    carry_intra_copy<<<dim3(n_tiles, n_batch), CARRY_TILE, 0, s>>>(
        d_out, d_buf_A, d_tile_carry, n_out, padded, n_batch);
    carry_inter_tiles<<<inter_blk, THR, 0, s>>>(
        d_out, d_tile_carry, n_out, n_batch);

#elif CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE
    carry_16bits<<<n_batch, CARRY_TILE, 0, s>>>(d_buf_A, d_out, n_out, padded, n_batch);

#elif CARRY_NORM_ALG == CARRY_ALG_SEQUENTIAL
    int blk = (n_batch + 31) / 32;
    carry_sequential<<<blk, 32, 0, s>>>(d_out, d_buf_A, n_out, padded, n_batch);
#endif
}

void BigIntNTTBatch::vadd_raw_buf(Data64 *d_dst, int n_dst, cudaStream_t s)
{
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    int n_tiles = (n_dst + CARRY_TILE - 1) / CARRY_TILE;
    vadd_from_raw_batch<<<dim3(n_tiles, n_batch), CARRY_TILE, 0, s>>>(
        d_dst, d_buf_A, n_dst, padded, n_batch);
#elif CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE
    constexpr int THR = MR_THR_REDUCE;
    unsigned bp = (unsigned)(n_dst + THR - 1) / THR;
    vadd_from_raw_batch<<<dim3(bp, (unsigned)n_batch), THR, 0, s>>>(
        d_dst, d_buf_A, n_dst, padded, n_batch);
#endif
    // SEQUENTIAL: não tem vadd separado — usar add_raw_buf_and_carry diretamente
}

void BigIntNTTBatch::carry_after_vadd(Data64 *d_dst, int n_dst, cudaStream_t s)
{
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    constexpr int THR = MR_CARRY_INTER_THR;
    int n_tiles = (n_dst + CARRY_TILE - 1) / CARRY_TILE;
    int inter_blk = (n_batch + THR - 1) / THR;
    carry_intra_copy<<<dim3(n_tiles, n_batch), CARRY_TILE, 0, s>>>(
        d_dst, d_dst, d_tile_carry, n_dst, n_dst, n_batch);
    carry_inter_tiles<<<inter_blk, THR, 0, s>>>(
        d_dst, d_tile_carry, n_dst, n_batch);
#elif CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE
    carry_16bits<<<n_batch, CARRY_TILE, 0, s>>>(d_dst, d_dst, n_dst, n_dst, n_batch);
#endif
    // SEQUENTIAL: no-op — carry já foi feito em add_raw_buf_and_carry
}

void BigIntNTTBatch::add_raw_buf_and_carry(Data64 *d_dst, int n_dst,
                                           cudaStream_t s)
{
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE || CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE
    vadd_raw_buf(d_dst, n_dst, s);
    carry_after_vadd(d_dst, n_dst, s);
#elif CARRY_NORM_ALG == CARRY_ALG_SEQUENTIAL
    int blk = (n_batch + 31) / 32;
    vadd_carry_sequential<<<blk, 32, 0, s>>>(d_dst, d_buf_A, n_dst, padded, n_batch);
#endif
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

void BigIntNTTBatch::add_and_carry(Data64 *d_a, const Data64 *d_b, int n, int n_passes,
                                   cudaStream_t s)
{
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    constexpr int THR = MR_CARRY_INTER_THR;
    int n_tiles = (n + CARRY_TILE - 1) / CARRY_TILE;
    int inter_blk = (n_batch + THR - 1) / THR;
    vadd_batch<<<dim3(n_tiles, n_batch), CARRY_TILE, 0, s>>>(
        d_a, d_a, d_b, n, n_batch);
    carry_intra_copy<<<dim3(n_tiles, n_batch), CARRY_TILE, 0, s>>>(
        d_a, d_a, d_tile_carry, n, n, n_batch, n_passes);
    carry_inter_tiles<<<inter_blk, THR, 0, s>>>(
        d_a, d_tile_carry, n, n_batch);

#elif CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE
    constexpr int THR = MR_THR_REDUCE;
    unsigned bp = (unsigned)(n + THR - 1) / THR;
    vadd_batch<<<dim3(bp, (unsigned)n_batch), THR, 0, s>>>(
        d_a, d_a, d_b, n, n_batch);
    carry_16bits<<<n_batch, CARRY_TILE, 0, s>>>(d_a, d_a, n, n, n_batch);

#elif CARRY_NORM_ALG == CARRY_ALG_SEQUENTIAL
    // carry_sequential lê d_src e escreve d_dst; em-place é seguro (j cresce)
    // Mas primeiro precisamos somar d_b em d_a
    constexpr int THR = MR_THR_REDUCE;
    unsigned bp = (unsigned)(n + THR - 1) / THR;
    vadd_batch<<<dim3(bp, (unsigned)n_batch), THR, 0, s>>>(
        d_a, d_a, d_b, n, n_batch);
    int blk = (n_batch + 31) / 32;
    carry_sequential<<<blk, 32, 0, s>>>(d_a, d_a, n, n, n_batch);
#endif
}
