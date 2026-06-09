// bigint_ntt.cu
#include "bigint_ntt.cuh"
#include <stdexcept>
#include <string>

#define CU(expr) \
    do { cudaError_t _e=(expr); if(_e!=cudaSuccess) \
        throw std::runtime_error(std::string("[CUDA] " #expr ": ")+cudaGetErrorString(_e)); \
    } while(0)

// ── kernels NTT ───────────────────────────────────────────────────────────────

__global__ static void load_padded_batch(Data64* __restrict__ dst,
                                          const Data64* __restrict__ src,
                                          int n_src, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j    = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= padded) return;
    dst[cand * padded + j] = (j < n_src) ? src[cand * n_src + j] : 0ULL;
}

__global__ static void pmul_batch(Data64* __restrict__ a,
                                   const Data64* __restrict__ b,
                                   int total, Data64 p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    a[i] = (Data64)((__uint128_t)a[i] * b[i] % (__uint128_t)p);
}

__global__ static void psq_batch(Data64* __restrict__ a, int total, Data64 p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    a[i] = (Data64)((__uint128_t)a[i] * a[i] % (__uint128_t)p);
}

// ── Carry normalização com tiles sequenciais ──────────────────────────────────
//
// Substituição da versão paralela com n_passes (que falhava porque os valores
// brutos do INTT chegam com 34+ bits, exigindo >4000 passes paralelos).
//
// Nova abordagem: 1 thread por tile processa os CARRY_TILE elementos
// sequencialmente, acumulando o carry corretamente.
//
// Fase 1 (intra): cada thread normaliza seu tile paralelamente.
// Fase 2 (inter): propaga carries entre tiles sequencialmente (1 thread/candidato).

static constexpr int CARRY_TILE = 512;

#define CARRY_INTRA_SEQUENTIAL  // descomente para usar versão sequencial (1 thread por tile)

// Fase 1 — copy + normaliza carries intra-tile.
__global__ static void carry_intra_copy(
        Data64* __restrict__       d_dst,
        const Data64* __restrict__ d_src,
        Data64* __restrict__       d_tile_carry,
        int n_dst, int n_src, int n_batch, int /*n_passes*/)
{
    int cand = blockIdx.y, tile = blockIdx.x, tid = threadIdx.x;
    if (cand >= n_batch) return;

    const int THREADS_NUM = CARRY_TILE;
    int n_tiles = (n_dst + THREADS_NUM - 1) / THREADS_NUM;
    int j_start = tile * THREADS_NUM;

#ifdef CARRY_INTRA_SEQUENTIAL
    if (tid != 0) return;
    Data64 carry = 0;
    for (int i = 0; i < THREADS_NUM; i++) {
        int j = j_start + i;
        Data64 v = (j < n_src ? d_src[cand * n_src + j] : 0ULL) + carry;
        if (j < n_dst) d_dst[cand * n_dst + j] = v & LIMB_MASK;
        carry = v >> LIMB_BITS;
    }
    d_tile_carry[cand * n_tiles + tile] = carry;
#else
    int j = j_start + tid;
    __shared__ Data64 carry[THREADS_NUM + 1];
    __shared__ int has_carry[2];
    if (tid == 0) {
        carry[THREADS_NUM] = 0;
        has_carry[0] = false;
        has_carry[1] = false;
    }
    carry[tid] = ((j < n_src) ? d_src[cand * n_src + j] : 0ULL);
    __syncthreads();


    Data64 currVal = 0;
    int currIter = 0;
    do {
        Data64 v = carry[tid];
        currIter = currIter ^ 1;
        carry[tid] = 0;
        if (tid == 0) {
            has_carry[currIter] = false;
        }
        __syncthreads();
        v += currVal;
        currVal = v & LIMB_MASK;
        v >>= LIMB_BITS;
        if (v > 0) has_carry[currIter] = true;
        carry[tid + 1] += v;
        __syncthreads();
    } while (has_carry[currIter]);

    if (j < n_dst)
        d_dst[cand * n_dst + j] = currVal;
    if (tid == 0)
        d_tile_carry[cand * n_tiles + tile] = carry[THREADS_NUM];
#endif
}

__global__ static void vadd_batch(
        Data64* __restrict__       d_c,
        const Data64* __restrict__ d_a,
        const Data64* __restrict__ d_b,
        int n, int n_batch)
{
    int cand = blockIdx.y;
    int j    = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= n) return;
    d_c[cand * n + j] = d_a[cand * n + j] + d_b[cand * n + j];
}

// Soma d_buf_A (stride=padded, raw INTT) em d_dst (stride=n_dst) element-wise.
// Para j >= padded, d_buf_A não tem dados (resultado do produto é 0 ali), não soma.
__global__ static void vadd_from_raw_batch(
        Data64* __restrict__       d_dst,
        const Data64* __restrict__ d_raw,
        int n_dst, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j    = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= n_dst) return;
    if (j < padded)
        d_dst[cand * n_dst + j] += d_raw[cand * padded + j];
}

// Fase 2 — propaga carries entre tiles sequencialmente (1 thread por candidato).
__global__ static void carry_inter_tiles(
        Data64* __restrict__ d_dst,
        Data64* __restrict__ d_tile_carry,
        int n, int n_batch)
{
    int cand = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch) return;

    int n_tiles = (n + CARRY_TILE - 1) / CARRY_TILE;
    for (int t = 0; t < n_tiles - 1; t++) {
        Data64 c = d_tile_carry[cand * n_tiles + t];
        if (c == 0) continue;
        int j_start = (t + 1) * CARRY_TILE;
        int j_end   = min(j_start + CARRY_TILE, n);
        for (int j = j_start; c > 0 && j < j_end; j++) {
            Data64 v = d_dst[cand * n + j] + c;
            d_dst[cand * n + j] = v & LIMB_MASK;
            c = v >> LIMB_BITS;
        }
        // Carry que escapa do tile seguinte — raro mas possível
        if (c > 0 && j_end < n) d_tile_carry[cand * n_tiles + t + 1] += c;
    }
}

// ── BigIntNTTBatch ────────────────────────────────────────────────────────────

BigIntNTTBatch::BigIntNTTBatch(int n_limbs_, int n_batch_)
    : n_limbs(n_limbs_)
    , padded(next_pow2_ntt(2 * n_limbs_))
    , logn(__builtin_ctz(next_pow2_ntt(2 * n_limbs_)))
    , n_batch(n_batch_)
{
    NTTParameters<Data64> params(logn, ReductionPolynomial::X_N_minus);
    p_val   = params.modulus.value;
    n_inv   = params.n_inv;
    modulus = params.modulus;

    auto fwd_h = params.gpu_root_of_unity_table_generator(params.forward_root_of_unity_table);
    auto inv_h = params.gpu_root_of_unity_table_generator(params.inverse_root_of_unity_table);

    const size_t tbytes  = fwd_h.size() * sizeof(Root64);
    const size_t pbytes  = (size_t)n_batch * padded * sizeof(Data64);
    int n_tiles_max      = (padded + CARRY_TILE - 1) / CARRY_TILE;
    const size_t tcbytes = (size_t)n_batch * n_tiles_max * sizeof(Data64);

    CU(cudaMalloc(&d_fwd_table,  tbytes));
    CU(cudaMalloc(&d_inv_table,  tbytes));
    CU(cudaMalloc(&d_buf_A,      pbytes));
    CU(cudaMalloc(&d_buf_B,      pbytes));
    CU(cudaMalloc(&d_tile_carry, tcbytes));

    CU(cudaMemcpy(d_fwd_table, fwd_h.data(), tbytes, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_inv_table, inv_h.data(), tbytes, cudaMemcpyHostToDevice));
}

BigIntNTTBatch::~BigIntNTTBatch()
{
    cudaFree(d_fwd_table); cudaFree(d_inv_table);
    cudaFree(d_buf_A);     cudaFree(d_buf_B);
    cudaFree(d_tile_carry);
}

ntt_configuration<Data64> BigIntNTTBatch::make_cfg(type t, cudaStream_t s)
{
    return {
        .n_power        = logn,
        .ntt_type       = t,
        .ntt_layout     = PerPolynomial,
        .reduction_poly = ReductionPolynomial::X_N_minus,
        .zero_padding   = false,
        .mod_inverse    = n_inv,
        .stream         = s
    };
}

void BigIntNTTBatch::ntt_A(const Data64* d_src, int n_src, cudaStream_t s)
{
    const int thr = 256;
    unsigned bx = (unsigned)(padded + thr-1)/thr;
    load_padded_batch<<<dim3(bx,(unsigned)n_batch), thr, 0, s>>>(
        d_buf_A, d_src, n_src, padded, n_batch);
    GPU_NTT_Inplace(d_buf_A, d_fwd_table, modulus, make_cfg(FORWARD, s), n_batch);
}

void BigIntNTTBatch::ntt_B(const Data64* d_src, int n_src, cudaStream_t s)
{
    const int thr = 256;
    unsigned bx = (unsigned)(padded + thr-1)/thr;
    load_padded_batch<<<dim3(bx,(unsigned)n_batch), thr, 0, s>>>(
        d_buf_B, d_src, n_src, padded, n_batch);
    GPU_NTT_Inplace(d_buf_B, d_fwd_table, modulus, make_cfg(FORWARD, s), n_batch);
}

void BigIntNTTBatch::fwd_A(cudaStream_t s)
{
    GPU_NTT_Inplace(d_buf_A, d_fwd_table, modulus, make_cfg(FORWARD, s), n_batch);
}

void BigIntNTTBatch::pmul_and_intt(cudaStream_t s)
{
    int total = n_batch * padded;
    const int thr = 256, blk = (total + thr-1)/thr;
    pmul_batch<<<blk, thr, 0, s>>>(d_buf_A, d_buf_B, total, p_val);
    GPU_INTT_Inplace(d_buf_A, d_inv_table, modulus, make_cfg(INVERSE, s), n_batch);
}

void BigIntNTTBatch::pmul_ext_and_intt(const Data64* d_ext, cudaStream_t s)
{
    int total = n_batch * padded;
    const int thr = 256, blk = (total + thr-1)/thr;
    pmul_batch<<<blk, thr, 0, s>>>(d_buf_A, d_ext, total, p_val);
    GPU_INTT_Inplace(d_buf_A, d_inv_table, modulus, make_cfg(INVERSE, s), n_batch);
}

void BigIntNTTBatch::psq_and_intt(cudaStream_t s)
{
    int total = n_batch * padded;
    const int thr = 256, blk = (total + thr-1)/thr;
    psq_batch<<<blk, thr, 0, s>>>(d_buf_A, total, p_val);
    GPU_INTT_Inplace(d_buf_A, d_inv_table, modulus, make_cfg(INVERSE, s), n_batch);
}
#include <iostream>
void BigIntNTTBatch::carry_to_limbs(Data64* d_out, int n_out, int n_passes, cudaStream_t s)
{
    constexpr int THR = 128;
    int n_tiles    = (n_out + CARRY_TILE - 1) / CARRY_TILE;
    int inter_blk  = (n_batch + THR - 1) / THR;
    carry_intra_copy<<<dim3(n_tiles, n_batch), CARRY_TILE, 0, s>>>(
        d_out, d_buf_A, d_tile_carry, n_out, padded, n_batch, n_passes);

    carry_inter_tiles<<<inter_blk, THR, 0, s>>>(
        d_out, d_tile_carry, n_out, n_batch);
}

void BigIntNTTBatch::add_and_carry(Data64* d_a, const Data64* d_b, int n, int n_passes,
                                    cudaStream_t s)
{
    constexpr int THR = 128;
    int n_tiles   = (n + CARRY_TILE - 1) / CARRY_TILE;
    int inter_blk = (n_batch + THR - 1) / THR;
    vadd_batch<<<dim3(n_tiles, n_batch), CARRY_TILE, 0, s>>>(
        d_a, d_a, d_b, n, n_batch);
    carry_intra_copy<<<dim3(n_tiles, n_batch), CARRY_TILE, 0, s>>>(
        d_a, d_a, d_tile_carry, n, n, n_batch, n_passes);
    carry_inter_tiles<<<inter_blk, THR, 0, s>>>(
        d_a, d_tile_carry, n, n_batch);
}

void BigIntNTTBatch::add_raw_buf_and_carry(Data64* d_dst, int n_dst, int n_passes,
                                            cudaStream_t s)
{
    constexpr int THR = 128;
    int n_tiles   = (n_dst + CARRY_TILE - 1) / CARRY_TILE;
    int inter_blk = (n_batch + THR - 1) / THR;
    vadd_from_raw_batch<<<dim3(n_tiles, n_batch), CARRY_TILE, 0, s>>>(
        d_dst, d_buf_A, n_dst, padded, n_batch);
    carry_intra_copy<<<dim3(n_tiles, n_batch), CARRY_TILE, 0, s>>>(
        d_dst, d_dst, d_tile_carry, n_dst, n_dst, n_batch, n_passes);
    carry_inter_tiles<<<inter_blk, THR, 0, s>>>(
        d_dst, d_tile_carry, n_dst, n_batch);
}
