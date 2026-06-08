#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif
int run_unit_tests(void);
int run_gpu_tests(void);
int bench_miller_rabin_run(void);
#ifdef __cplusplus
}
#endif

int bench_gpu_run(const char *primes_path, long long a, long long b, long long c);

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage:\n"
        "  %s --test                          Run all tests\n"
        "  %s --bench                         Run Miller-Rabin benchmark\n"
        "  %s --bench-gpu <primes.bin> [a b c] Run GPU trial-division benchmark\n",
        prog, prog, prog);
}

int main(int argc, char **argv)
{
    if (argc < 2) { usage(argv[0]); return 1; }

    if (strcmp(argv[1], "--test") == 0) {
        int fail = 0;
        fail += run_unit_tests();
        fail += run_gpu_tests();
        return fail ? 1 : 0;
    }

    if (strcmp(argv[1], "--bench") == 0) {
        bench_miller_rabin_run();
        return 0;
    }

    if (strcmp(argv[1], "--bench-gpu") == 0) {
        if (argc < 3) {
            fprintf(stderr, "Error: --bench-gpu requires <primes.bin>\n");
            usage(argv[0]);
            return 1;
        }
        const char *primes_path = argv[2];
        long long a = argc > 3 ? atoll(argv[3]) : 50001;
        long long b = argc > 4 ? atoll(argv[4]) : 33334;
        long long c = argc > 5 ? atoll(argv[5]) : 2749;
        bench_gpu_run(primes_path, a, b, c);
        return 0;
    }

    usage(argv[0]);
    return 1;
}
