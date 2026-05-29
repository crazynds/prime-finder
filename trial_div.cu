#include "trial_div.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* ---------- precomputed lookup tables ---------- */

static void build_rev_tables(uint8_t rev_val[101], uint8_t rev_ndig[101])
{
    for (int x = 1; x <= 100; x++) {
        int nd = (x < 10) ? 1 : (x < 100) ? 2 : 3;
        int r = 0, tmp = x;
        for (int i = 0; i < nd; i++) { r = r*10 + tmp%10; tmp /= 10; }
        rev_val[x]  = (uint8_t)r;
        rev_ndig[x] = (uint8_t)nd;
    }
}

__constant__ uint8_t c_rev_val[101];
__constant__ uint8_t c_rev_ndig[101];

/* ---------- device helpers ---------- */

__device__ static uint64_t dev_powmod(uint64_t base, uint64_t exp, uint32_t mod)
{
    uint64_t result = 1; base %= mod;
    while (exp > 0) {
        if (exp & 1) result = result * base % mod;
        base = base * base % mod; exp >>= 1;
    }
    return result;
}


/* ---------- kernel ----------
 *
 * Strategy: each block accumulates its results in shared memory (KJ_WORDS words).
 * Shared-memory atomicOr has much lower contention than global.
 * At the end, one pass merges per-block results into global bitset.
 *
 * This avoids the O(10000) flat-loop: normal sieve is O(100),
 * rev sieve is O(100 * ~1) for large p. All threads in a warp run
 * the same k-loop with no divergence.
 */
__global__ void trial_div_kernel(
    const uint32_t * __restrict__ primes,
    uint32_t   n_primes,
    uint64_t   a,
    uint64_t   b,
    uint64_t   c,
    uint32_t  *bitset
)
{
    /* Per-block accumulator in shared memory */
    __shared__ uint32_t sbits[KJ_WORDS];  /* KJ_WORDS = KJ_WORDS */
    for (int i = (int)threadIdx.x; i < KJ_WORDS; i += (int)blockDim.x)
        sbits[i] = 0;
    __syncthreads();

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_primes) {
        uint32_t p = primes[tid];

        uint64_t ra = dev_powmod(10, a, p);
        uint64_t rb = dev_powmod(10, b, p);
        uint64_t rc = dev_powmod(10, c, p);

        /* ---- Normal sieve ---- */
        if (rc != 0) {
            uint64_t inv_rc = dev_powmod(rc, (uint64_t)(p-2), p);
            for (uint32_t k = 1; k <= 100; k++) {
                uint64_t num = (ra + 2*(uint64_t)p - (uint64_t)k * rb % p - 1) % p;
                uint64_t j0  = num * inv_rc % p;
                if (j0 == 0) j0 = p;
                for (uint64_t j = j0; j <= 100; j += p) {
                    int bit = (int)((k-1)*100 + (j-1));
                    atomicOr(&sbits[bit >> 5], 1u << (bit & 31));
                }
            }
        }

        /* ---- Rev sieve: p | rev(N)(k,j) ----
         * rev(N) = 10^a - rev(k)*10^(a-b-lk) - rev(j)*10^(a-c-lj) - 1
         *
         * For each k: solve algebraically for j in each digit-group.
         * O(100) per thread — same complexity as normal sieve.
         * inv_rc_rev[1,2] hoisted outside k-loop (don't depend on k).
         */
        uint64_t rb_rev[4], rc_rev[4];
        for (int l = 1; l <= 3; l++) {
            long long eb = (long long)a - (long long)b - l;
            long long ec = (long long)a - (long long)c - l;
            rb_rev[l] = (eb >= 0) ? dev_powmod(10, (uint64_t)eb, p) : 0;
            rc_rev[l] = (ec >= 0) ? dev_powmod(10, (uint64_t)ec, p) : 0;
        }
        for (uint32_t k = 1; k <= 100; k++) {
            uint32_t rk     = c_rev_val[k];
            uint64_t rb_eff = rb_rev[c_rev_ndig[k]];
            if (rb_eff == 0) continue;

            uint64_t base_k = (ra + 2*(uint64_t)p - (uint64_t)rk * rb_eff % p - 1) % p;

            /* Flat loop over j=1..100, no ifs per digit-group.
             * c_rev_ndig[j] indexes into rc_rev[] — branch-free table lookup. */
            for (uint32_t j = 1; j <= 100; j++) {
                uint32_t rj     = c_rev_val[j];
                uint64_t rc_eff = rc_rev[c_rev_ndig[j]];
                uint64_t val    = (base_k + (uint64_t)p - (uint64_t)rj * rc_eff % p) % p;
                if (val == 0) {
                    int bit = (int)((k-1)*100 + (j-1));
                    atomicOr(&sbits[bit >> 5], 1u << (bit & 31));
                }
            }
        }
    }

    __syncthreads();

    /* Merge block results into global bitset */
    for (int i = (int)threadIdx.x; i < KJ_WORDS; i += (int)blockDim.x)
        atomicOr(&bitset[i], sbits[i]);
}

/* ---------- configurable launcher (used by bench_gpu) ---------- */

void trial_div_launch(const uint32_t *d_primes, uint32_t n_primes,
                      uint64_t a, uint64_t b, uint64_t c,
                      uint32_t *d_bits, int block_size)
{
    int blocks = (n_primes + block_size - 1) / block_size;
    trial_div_kernel<<<blocks, block_size>>>(d_primes, n_primes, a, b, c, d_bits);
}

/* ---------- public API ---------- */

extern "C" int cuda_get_device_count(void)
{
    int n = 0; cudaGetDeviceCount(&n); return n;
}

extern "C" int cuda_set_device(int index)
{
    return (cudaSetDevice(index) == cudaSuccess) ? 0 : -1;
}

extern "C" int trial_div_gpu(const prime_list_t *pl,
                              long long a, long long b, long long c,
                              uint32_t *sieve_bits)
{
    static int tables_loaded = 0;
    if (!tables_loaded) {
        uint8_t rv[101], rn[101];
        build_rev_tables(rv, rn);
        cudaMemcpyToSymbol(c_rev_val,  rv, sizeof(rv));
        cudaMemcpyToSymbol(c_rev_ndig, rn, sizeof(rn));
        tables_loaded = 1;
    }

    uint32_t *d_primes = NULL, *d_bits = NULL;
    int rc = 0;
    size_t pb = (size_t)pl->count * sizeof(uint32_t);
    size_t bb = KJ_WORDS * sizeof(uint32_t);

    if (cudaMalloc(&d_primes, pb) != cudaSuccess) { rc = -1; goto done; }
    if (cudaMemcpy(d_primes, pl->primes, pb, cudaMemcpyHostToDevice) != cudaSuccess)
        { rc = -1; goto done; }
    if (cudaMalloc(&d_bits, bb) != cudaSuccess) { rc = -1; goto done; }
    if (cudaMemcpy(d_bits, sieve_bits, bb, cudaMemcpyHostToDevice) != cudaSuccess)
        { rc = -1; goto done; }

    {
#ifndef TRIAL_DIV_BLOCK_SIZE
#define TRIAL_DIV_BLOCK_SIZE 256
#endif
        int threads = TRIAL_DIV_BLOCK_SIZE;
        int blocks  = (pl->count + threads - 1) / threads;
        trial_div_kernel<<<blocks, threads>>>(
            d_primes, pl->count,
            (uint64_t)a, (uint64_t)b, (uint64_t)c, d_bits);
        if (cudaDeviceSynchronize() != cudaSuccess) { rc = -1; goto done; }
    }

    if (cudaMemcpy(sieve_bits, d_bits, bb, cudaMemcpyDeviceToHost) != cudaSuccess)
        { rc = -1; goto done; }

done:
    if (d_primes) cudaFree(d_primes);
    if (d_bits)   cudaFree(d_bits);
    if (rc != 0) fprintf(stderr, "trial_div_gpu: CUDA error\n");
    return rc;
}
