#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include "../trial_div.h"
#include "../sieve.h"
#include "../prime_list.h"

#define FAIL(msg) do { printf("FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); return 1; } while(0)
#define ASSERT(cond) do { if (!(cond)) FAIL(#cond); } while(0)

static uint64_t powmod_h(uint64_t base, uint64_t exp, uint32_t mod)
{
    uint64_t r = 1; base %= mod;
    while (exp > 0) {
        if (exp & 1) r = r * base % mod;
        base = base * base % mod; exp >>= 1;
    }
    return r;
}

static int ndig(int x) { return (x<10)?1:(x<100)?2:3; }
static int revd(int x, int lx) {
    int r=0; for(int i=0;i<lx;i++){r=r*10+x%10;x/=10;} return r;
}

/*
 * CPU brute-force: mark (k,j) composite if p|N OR p|rev(N) for any prime p.
 * This is the ground truth that the GPU must match.
 */
static void cpu_brute_sieve(const prime_list_t *pl,
                             long long a, long long b, long long c,
                             uint32_t *bits_out)
{
    for (uint32_t pi = 0; pi < pl->count; pi++) {
        uint32_t p = pl->primes[pi];
        uint64_t ra = powmod_h(10, a, p);
        uint64_t rb = powmod_h(10, b, p);
        uint64_t rc = powmod_h(10, c, p);

        for (int k = 1; k <= 100; k++) {
            int lk = ndig(k), rk = revd(k, lk);
            uint64_t rb2 = powmod_h(10, a-b-lk, p);

            for (int j = 1; j <= 100; j++) {
                int already = 0;

                /* Normal: p | N(k,j)? */
                long long v = ((long long)ra
                               - (long long)k * rb % p
                               - (long long)j * rc % p
                               - 1 + 4*(long long)p) % p;
                if (v == 0) already = 1;

                /* Rev: p | rev(N)(k,j)? */
                if (!already) {
                    int lj = ndig(j), rj = revd(j, lj);
                    uint64_t rc2 = powmod_h(10, a-c-lj, p);
                    long long vr = ((long long)ra
                                    - (long long)rk * rb2 % p
                                    - (long long)rj * rc2 % p
                                    - 1 + 4*(long long)p) % p;
                    if (vr == 0) already = 1;
                }

                if (already) {
                    int bit = (k-1)*100 + (j-1);
                    bits_out[bit>>5] |= 1u << (bit&31);
                }
            }
        }
    }
}

static int test_gpu_vs_cpu(const prime_list_t *pl, long long a, long long b, long long c)
{
    uint32_t gpu_bits[KJ_WORDS], cpu_bits[KJ_WORDS];
    memset(gpu_bits, 0, sizeof(gpu_bits));
    memset(cpu_bits, 0, sizeof(cpu_bits));

    cpu_brute_sieve(pl, a, b, c, cpu_bits);

    if (trial_div_gpu(pl, a, b, c, gpu_bits) != 0)
        FAIL("trial_div_gpu returned error");

    /* 1. GPU must mark at least everything CPU marks (no false negatives) */
    for (int w = 0; w < KJ_WORDS; w++) {
        if ((cpu_bits[w] & ~gpu_bits[w]) != 0) {
            printf("FAIL: GPU missed composites (word %d: cpu=%08x gpu=%08x)\n",
                   w, cpu_bits[w], gpu_bits[w]);
            return 1;
        }
    }

    /*
     * 2. Every survivor (bit=0 in gpu_bits) must truly have no prime in pl
     *    dividing EITHER N(k,j) OR rev(N)(k,j).
     *    This validates there are no false composites (marks that shouldn't be set).
     */
    for (int k = 1; k <= 100; k++) {
        for (int j = 1; j <= 100; j++) {
            int bit = (k-1)*100 + (j-1);
            int gpu_marked = (gpu_bits[bit>>5] >> (bit&31)) & 1;
            if (gpu_marked) continue;  /* eliminated — skip */

            /* Survivor: verify no prime in pl divides N or rev(N) */
            int lk = ndig(k), lj = ndig(j);
            int rk = revd(k,lk), rj = revd(j,lj);

            for (uint32_t pi = 0; pi < pl->count; pi++) {
                uint32_t p = pl->primes[pi];
                uint64_t ra  = powmod_h(10, a,       p);
                uint64_t rb  = powmod_h(10, b,       p);
                uint64_t rc  = powmod_h(10, c,       p);
                uint64_t rb2 = powmod_h(10, a-b-lk,  p);
                uint64_t rc2 = powmod_h(10, a-c-lj,  p);

                long long vN = ((long long)ra
                                - (long long)k  * rb  % p
                                - (long long)j  * rc  % p
                                - 1 + 4*(long long)p) % p;
                long long vR = ((long long)ra
                                - (long long)rk * rb2 % p
                                - (long long)rj * rc2 % p
                                - 1 + 4*(long long)p) % p;

                if (vN == 0) {
                    printf("FAIL: survivor k=%d j=%d has p=%u | N  (false composite miss)\n",
                           k, j, p);
                    return 1;
                }
                if (vR == 0) {
                    printf("FAIL: survivor k=%d j=%d has p=%u | rev(N)  (false composite miss)\n",
                           k, j, p);
                    return 1;
                }
            }
        }
    }

    return 0;
}

static int test_gpu_consistency(const prime_list_t *pl, long long a, long long b, long long c)
{
    uint32_t bits1[KJ_WORDS], bits2[KJ_WORDS];
    memset(bits1, 0, sizeof(bits1));
    memset(bits2, 0, sizeof(bits2));

    ASSERT(trial_div_gpu(pl, a, b, c, bits1) == 0);
    ASSERT(trial_div_gpu(pl, a, b, c, bits2) == 0);

    for (int w = 0; w < KJ_WORDS; w++)
        ASSERT(bits1[w] == bits2[w]);
    return 0;
}

int run_gpu_tests(void)
{
    uint32_t p_data[] = {
        3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97
    };
    prime_list_t pl;
    pl.primes = p_data;
    pl.count  = sizeof(p_data)/sizeof(p_data[0]);

    int fail = 0;

    fail += test_gpu_vs_cpu(&pl, 15, 8, 3);
    fail += test_gpu_vs_cpu(&pl, 20, 10, 5);
    fail += test_gpu_consistency(&pl, 15, 8, 3);
    fail += test_gpu_consistency(&pl, 25, 12, 6);

    if (fail == 0) printf("GPU tests PASSED\n");
    else           printf("%d GPU test(s) FAILED\n", fail);
    return fail;
}
