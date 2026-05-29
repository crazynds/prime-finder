#include "trial_div_cpu.h"
#include <stdint.h>
#include <string.h>

static uint8_t cpu_rev_val[101];
static uint8_t cpu_rev_ndig[101];
static int     cpu_rev_built = 0;

static void build_rev(void)
{
    if (cpu_rev_built) return;
    for (int x = 1; x <= 100; x++) {
        int nd = (x < 10) ? 1 : (x < 100) ? 2 : 3;
        int r = 0, t = x;
        for (int i = 0; i < nd; i++) { r = r*10 + t%10; t /= 10; }
        cpu_rev_val[x]  = (uint8_t)r;
        cpu_rev_ndig[x] = (uint8_t)nd;
    }
    cpu_rev_built = 1;
}

static uint64_t powmod_c(uint64_t base, uint64_t exp, uint32_t mod)
{
    uint64_t r = 1; base %= mod;
    while (exp > 0) {
        if (exp & 1) r = r * base % mod;
        base = base * base % mod; exp >>= 1;
    }
    return r;
}

static void sieve_one_prime(uint32_t p,
                             uint64_t a, uint64_t b, uint64_t c,
                             uint32_t *bits)
{
    uint64_t ra = powmod_c(10, a, p);
    uint64_t rb = powmod_c(10, b, p);
    uint64_t rc = powmod_c(10, c, p);

    /* Normal sieve */
    if (rc != 0) {
        uint64_t inv_rc = powmod_c(rc, p-2, p);
        for (uint32_t k = 1; k <= 100; k++) {
            uint64_t num = (ra + 2*(uint64_t)p - (uint64_t)k * rb % p - 1) % p;
            uint64_t j0  = num * inv_rc % p;
            if (j0 == 0) j0 = p;
            for (uint64_t j = j0; j <= 100; j += p) {
                int bit = (int)((k-1)*100 + (j-1));
                bits[bit >> 5] |= 1u << (bit & 31);
            }
        }
    }

    /* Rev sieve */
    uint64_t rb_rev[4], rc_rev[4];
    for (int l = 1; l <= 3; l++) {
        long long eb = (long long)a - (long long)b - l;
        long long ec = (long long)a - (long long)c - l;
        rb_rev[l] = (eb >= 0) ? powmod_c(10, (uint64_t)eb, p) : 0;
        rc_rev[l] = (ec >= 0) ? powmod_c(10, (uint64_t)ec, p) : 0;
    }
    uint64_t inv1 = (rc_rev[1] != 0) ? powmod_c(rc_rev[1], p-2, p) : 0;
    uint64_t inv2 = (rc_rev[2] != 0) ? powmod_c(rc_rev[2], p-2, p) : 0;

    for (uint32_t k = 1; k <= 100; k++) {
        uint32_t rk     = cpu_rev_val[k];
        uint64_t rb_eff = rb_rev[cpu_rev_ndig[k]];
        if (rb_eff == 0) continue;

        uint64_t base_k = (ra + 2*(uint64_t)p - (uint64_t)rk * rb_eff % p - 1) % p;

        /* lj=1 */
        if (inv1 != 0) {
            uint64_t j0 = base_k * inv1 % p;
            if (j0 == 0) j0 = p;
            for (uint64_t j = j0; j <= 9; j += p) {
                int bit = (int)((k-1)*100 + (j-1));
                bits[bit >> 5] |= 1u << (bit & 31);
            }
        }
        /* lj=2 */
        if (inv2 != 0) {
            uint64_t trj = base_k * inv2 % p;
            if (trj == 0) trj = p;
            for (uint64_t rj = trj; rj <= 99; rj += p) {
                int j = -1;
                if (rj >= 1 && rj <= 9) j = (int)rj * 10;
                else if (rj >= 10 && rj <= 99 && rj % 10 != 0)
                    j = (int)(rj%10)*10 + (int)(rj/10);
                if (j >= 10 && j <= 99) {
                    int bit = (int)((k-1)*100 + (j-1));
                    bits[bit >> 5] |= 1u << (bit & 31);
                }
            }
        }
        /* lj=3 */
        if (rc_rev[3] != 0 && rc_rev[3] == base_k) {
            int bit = (int)((k-1)*100 + 99);
            bits[bit >> 5] |= 1u << (bit & 31);
        }
    }
}

void trial_div_cpu(const prime_list_t *pl,
                   long long a, long long b, long long c,
                   uint32_t *sieve_bits)
{
    build_rev();
    for (uint32_t i = 0; i < pl->count; i++)
        sieve_one_prime(pl->primes[i], (uint64_t)a, (uint64_t)b, (uint64_t)c, sieve_bits);
}
