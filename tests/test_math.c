#include <stdio.h>
#include <stdint.h>

#define FAIL(msg) do { printf("FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); return 1; } while(0)
#define ASSERT(cond) do { if (!(cond)) FAIL(#cond); } while(0)

static uint64_t powmod(uint64_t base, uint64_t exp, uint32_t mod)
{
    uint64_t r = 1; base %= mod;
    while (exp > 0) {
        if (exp & 1) r = r * base % mod;
        base = base * base % mod; exp >>= 1;
    }
    return r;
}

static uint64_t modinv(uint64_t a, uint32_t p)
{
    return powmod(a, p-2, p);
}

static int mult_order(uint64_t base, uint32_t p)
{
    base %= p;
    if (base == 0) return 0;
    uint32_t phi = p - 1;
    uint32_t factors[20]; int nf = 0;
    uint32_t tmp = phi;
    for (uint32_t f = 2; (uint64_t)f*f <= tmp; f++) {
        if (tmp % f == 0) { factors[nf++] = f; while (tmp % f == 0) tmp /= f; }
    }
    if (tmp > 1) factors[nf++] = tmp;
    uint64_t ord = phi;
    for (int i = 0; i < nf; i++) {
        while (ord % factors[i] == 0) {
            uint64_t cand = ord / factors[i];
            if (powmod(base, cand, p) == 1) ord = cand;
            else break;
        }
    }
    return (int)ord;
}

static int test_powmod_basic(void)
{
    ASSERT(powmod(2, 10, 1000) == 24);   /* 1024 mod 1000 */
    ASSERT(powmod(3, 0, 7)     == 1);
    ASSERT(powmod(5, 1, 7)     == 5);
    ASSERT(powmod(2, 100, 101) == 1);    /* Fermat: 2^100 ≡ 1 mod 101 (101 prime) */
    ASSERT(powmod(10, 6, 7)    == 1);    /* ord_7(10) = 6, so 10^6 ≡ 1 mod 7 */
    return 0;
}

static int test_modinv_basic(void)
{
    uint32_t primes[] = {7, 11, 13, 97, 1009};
    for (int i = 0; i < 5; i++) {
        uint32_t p = primes[i];
        for (uint32_t a = 1; a < p; a++) {
            uint64_t inv = modinv(a, p);
            ASSERT(a * inv % p == 1);
        }
    }
    return 0;
}

static int test_ord_p_basic(void)
{
    /* ord_7(10) = 6 */
    ASSERT(mult_order(10, 7) == 6);
    /* ord_11(10) = 2  (10^2=100=9*11+1) */
    ASSERT(mult_order(10, 11) == 2);
    /* ord_13(10) = 6 */
    ASSERT(mult_order(10, 13) == 6);

    /* Verify: 10^ord ≡ 1 mod p and is smallest */
    uint32_t test_primes[] = {7, 11, 13, 17, 19, 23, 97};
    for (int i = 0; i < 7; i++) {
        uint32_t p = test_primes[i];
        int ord = mult_order(10, p);
        ASSERT(ord > 0);
        ASSERT(powmod(10, ord, p) == 1);
        for (int d = 1; d < ord; d++) {
            if (ord % d == 0)
                ASSERT(powmod(10, d, p) != 1);
        }
    }
    return 0;
}

int test_math(void)
{
    int fail = 0;
    fail += test_powmod_basic();
    fail += test_modinv_basic();
    fail += test_ord_p_basic();
    if (!fail) printf("[math] OK\n");
    return fail;
}
