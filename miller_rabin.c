#include "miller_rabin.h"
#include <gmp.h>

/* Number of decimal digits of x (x >= 1) */
static int ndigits(int x)
{
    if (x < 10)  return 1;
    if (x < 100) return 2;
    return 3;  /* max k,j = 100 */
}

/* Reverse decimal digits of x with lx digits (e.g. rev(34)=43, rev(100)=1) */
static int rev_digits(int x, int lx)
{
    int r = 0;
    for (int i = 0; i < lx; i++) {
        r = r * 10 + (x % 10);
        x /= 10;
    }
    return r;
}

/* Build N = 10^a - k*10^b - j*10^c - 1 into out (must be init'd) */
static void build_N(mpz_t out, mpz_t tmp,
                    long long a, long long b, long long c, int k, int j)
{
    mpz_ui_pow_ui(out, 10, (unsigned long)a);

    mpz_ui_pow_ui(tmp, 10, (unsigned long)b);
    mpz_mul_ui(tmp, tmp, (unsigned long)k);
    mpz_sub(out, out, tmp);

    mpz_ui_pow_ui(tmp, 10, (unsigned long)c);
    mpz_mul_ui(tmp, tmp, (unsigned long)j);
    mpz_sub(out, out, tmp);

    mpz_sub_ui(out, out, 1);
}

/*
 * rev(N) = 10^a - rev(k)*10^(a-b-lk) - rev(j)*10^(a-c-lj) - 1
 *
 * The two exponents determine which term acts as "k" and which as "j"
 * in the formula (the larger exponent is the "b" position).
 */
static void build_revN(mpz_t out, mpz_t tmp,
                       long long a, long long b, long long c, int k, int j)
{
    int lk = ndigits(k), lj = ndigits(j);
    int rk = rev_digits(k, lk), rj = rev_digits(j, lj);

    long long eb = a - b - lk;  /* exponent for rev(k) term */
    long long ec = a - c - lj;  /* exponent for rev(j) term */

    mpz_ui_pow_ui(out, 10, (unsigned long)a);

    mpz_ui_pow_ui(tmp, 10, (unsigned long)eb);
    mpz_mul_ui(tmp, tmp, (unsigned long)rk);
    mpz_sub(out, out, tmp);

    mpz_ui_pow_ui(tmp, 10, (unsigned long)ec);
    mpz_mul_ui(tmp, tmp, (unsigned long)rj);
    mpz_sub(out, out, tmp);

    mpz_sub_ui(out, out, 1);
}

int miller_rabin_test(long long a, long long b, long long c, int k, int j)
{
    mpz_t N, tmp;
    mpz_init(N);
    mpz_init(tmp);

    build_N(N, tmp, a, b, c, k, j);

    int r_N = mpz_probab_prime_p(N, 25);
    if (r_N == 0) {
        mpz_clear(N); mpz_clear(tmp);
        return 0;
    }
    // if (r_N == 2) {
    //     mpz_clear(N); mpz_clear(tmp);
    //     return 3;  /* definitely prime (small N only) */
    // }

    /* N is probable prime — test rev(N) */
    build_revN(N, tmp, a, b, c, k, j);  /* reuse N to save memory */

    int r_R = mpz_probab_prime_p(N, 25);

    mpz_clear(N); mpz_clear(tmp);

    /* 2 = both N and rev(N) probable prime, 1 = only N */
    return (r_R > 0) ? 2 : 1;
}
