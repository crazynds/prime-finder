// ops/shift/shift.cu — kernels e lançadores de shift/extract de limbs.
#include "ops/shift/shift.cuh"

namespace
{
    __global__ void shift_right_k(Data64 *__restrict__ dst, const Data64 *__restrict__ src,
                                  int offset, int n_out, int n_src, int n_batch)
    {
        int cand = blockIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (cand >= n_batch || j >= n_out)
            return;
        int sidx = j + offset;
        dst[(size_t)cand * n_out + j] = (sidx < n_src) ? src[(size_t)cand * n_src + sidx] : 0ULL;
    }

    __global__ void shift_right_var_k(Data64 *__restrict__ dst, const Data64 *__restrict__ src,
                                      const int *__restrict__ bark, int delta,
                                      int n_out, int n_src, int n_batch)
    {
        int cand = blockIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (cand >= n_batch || j >= n_out)
            return;
        int off = bark[cand] + delta;
        int sidx = j + off;
        dst[(size_t)cand * n_out + j] =
            (sidx >= 0 && sidx < n_src) ? src[(size_t)cand * n_src + sidx] : 0ULL;
    }

    __global__ void extract_low_k(Data64 *__restrict__ dst, const Data64 *__restrict__ src,
                                  int n_low, int padded, int n_sum, int n_batch)
    {
        int cand = blockIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (cand >= n_batch || j >= padded)
            return;
        dst[(size_t)cand * padded + j] = (j < n_low) ? src[(size_t)cand * n_sum + j] : 0ULL;
    }
}

namespace ops
{
    void shift_right(Data64 *dst, const Data64 *src, int offset,
                     int n_out, int n_src, int n_batch, int thr, cudaStream_t s)
    {
        dim3 g((unsigned)(n_out + thr - 1) / thr, (unsigned)n_batch);
        shift_right_k<<<g, thr, 0, s>>>(dst, src, offset, n_out, n_src, n_batch);
    }

    void shift_right_var(Data64 *dst, const Data64 *src, const int *bark, int delta,
                         int n_out, int n_src, int n_batch, int thr, cudaStream_t s)
    {
        dim3 g((unsigned)(n_out + thr - 1) / thr, (unsigned)n_batch);
        shift_right_var_k<<<g, thr, 0, s>>>(dst, src, bark, delta, n_out, n_src, n_batch);
    }

    void extract_low(Data64 *dst, const Data64 *src, int n_low, int padded,
                     int n_sum, int n_batch, int thr, cudaStream_t s)
    {
        dim3 g((unsigned)(padded + thr - 1) / thr, (unsigned)n_batch);
        extract_low_k<<<g, thr, 0, s>>>(dst, src, n_low, padded, n_sum, n_batch);
    }
}
