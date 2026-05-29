/*
 * bench_gpu: measures trial_div_gpu throughput for different block sizes.
 *
 * Usage:  ./bench_gpu <primes.bin> [a] [b] [c]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>

#include "sieve_table.h"  /* KJ_WORDS */

#ifdef __cplusplus
extern "C" {
#endif
#include "prime_list.h"
#ifdef __cplusplus
}
#endif

#include "trial_div.h"

/* trial_div_launch is a C++ function defined in trial_div.cu */
void trial_div_launch(const uint32_t *d_primes, uint32_t n_primes,
                      uint64_t a, uint64_t b, uint64_t c,
                      uint32_t *d_bits, int block_size);

static double bench_one(const prime_list_t *pl,
                        long long a, long long b, long long c,
                        int block_size, int n_runs)
{
    uint32_t *d_primes = NULL, *d_bits = NULL;
    uint32_t host_bits[KJ_WORDS];
    memset(host_bits, 0, sizeof(host_bits));

    size_t pb = (size_t)pl->count * sizeof(uint32_t);
    size_t bb = KJ_WORDS * sizeof(uint32_t);

    cudaMalloc(&d_primes, pb);
    cudaMalloc(&d_bits,   bb);
    cudaMemcpy(d_primes, pl->primes, pb, cudaMemcpyHostToDevice);

    /* warm-up */
    cudaMemcpy(d_bits, host_bits, bb, cudaMemcpyHostToDevice);
    trial_div_launch(d_primes, pl->count, a, b, c, d_bits, block_size);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    for (int r = 0; r < n_runs; r++) {
        cudaMemcpy(d_bits, host_bits, bb, cudaMemcpyHostToDevice);
        trial_div_launch(d_primes, pl->count, a, b, c, d_bits, block_size);
    }
    cudaEventRecord(t1);
    cudaDeviceSynchronize();

    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_primes);
    cudaFree(d_bits);

    return (double)ms / n_runs;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <primes.bin> [a] [b] [c]\n", argv[0]);
        return 1;
    }

    prime_list_t pl;
    if (prime_list_load(&pl, argv[1]) != 0) return 1;

    long long a = (argc > 2) ? atoll(argv[2]) : 50000;
    long long b = (argc > 3) ? atoll(argv[3]) : 33333;
    long long c = (argc > 4) ? atoll(argv[4]) : 200;

    printf("Primes: %u  a=%lld b=%lld c=%lld\n\n", pl.count, a, b, c);
    printf("%-12s  %-10s  %-16s  %-8s\n",
           "block_size", "time_ms", "Mprimes/s", "vs_256");
    printf("%-12s  %-10s  %-16s  %-8s\n",
           "----------", "-------", "---------", "------");

    int sizes[] = {32,64, 128, 256, 512, 1024};
    int n = sizeof(sizes)/sizeof(sizes[0]);
    int n_runs = 10;

    double ref_ms = -1;
    double best_ms = 1e18;
    int    best_bs = 256;

    for (int i = 0; i < n; i++) {
        int bs = sizes[i];
        double ms = bench_one(&pl, a, b, c, bs, n_runs);
        double mpms = (double)pl.count / ms / 1e3;

        if (bs == 256) ref_ms = ms;
        double ratio = (ref_ms > 0) ? ref_ms / ms : 0;

        printf("%-12d  %-10.3f  %-16.2f  %.2fx\n", bs, ms, mpms, ratio);

        if (ms < best_ms) { best_ms = ms; best_bs = bs; }
    }

    printf("\nBest: block_size=%d  %.3f ms/task\n", best_bs, best_ms);
    printf("Tip: set BLOCK_SIZE=%d in CMakeLists.txt -DTRIAL_DIV_BLOCK_SIZE=%d\n",
           best_bs, best_bs);

    prime_list_free(&pl);
    return 0;
}
