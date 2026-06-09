#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../miller_rabin.h"

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/*
 * Uses a=50001 b=33334 c=2749 — same as early real workload values.
 * Note: pairs that quickly return 0 (composite) are NOT representative
 * of real Phase 2 cost; survivors from Phase 1 require all MR rounds.
 * The "worst case" timing comes from pairs returning result >= 1.
 */
/* Real Phase 1 survivors from a=50001 b=33334 c=2749 */
static const int BENCH_KS[] = {25,24,23,18,18,15,13,13,10,6,1,1,1};
static const int BENCH_JS[] = {91,34,23,98,13,58,67,19,18,47,42,15,13};

int bench_miller_rabin_run(void)
{
    long long a = 50001;
    long long b = 33334;
    long long c = 2749;
    int n_pairs = 1;

    printf("Miller-Rabin bench: a=%lld b=%lld c=%lld  pairs=%d\n", a, b, c, n_pairs);
    printf("N has ~%lld decimal digits\n\n", a);

    double t_total = 0;
    double t_worst = 0;
    for (int i = 0; i < n_pairs; i++) {
        int k = BENCH_KS[i], j = BENCH_JS[i];

        double t0 = now_sec();
        int r = miller_rabin_test(a, b, c, k, j, NULL);
        double elapsed = now_sec() - t0;
        t_total += elapsed;
        if (elapsed > t_worst) t_worst = elapsed;

        printf("  k=%-3d j=%-3d  result=%d  time=%.3fs\n", k, j, r, elapsed);
    }

    printf("\nTotal: %.3fs  avg/pair: %.3fs  worst: %.3fs\n",
           t_total, t_total / n_pairs, t_worst);
    return 0;
}

