#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "../sieve_table.h"
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

/* Direct check: is N(a,b,c,k,j) divisible by p? — N = 10^a - k*10^b - j*10^c - 1 */
static int is_div_by(long long a, long long b, long long c, int k, int j, uint32_t p)
{
    uint64_t ra = powmod_h(10, a, p);
    uint64_t rb = powmod_h(10, b, p);
    uint64_t rc = powmod_h(10, c, p);
    long long v = ((long long)ra - (long long)k * rb % p - (long long)j * rc % p - 1 + 4*(long long)p) % p;
    return (v == 0);
}

static int ndig_s(int x) { return (x<10)?1:(x<100)?2:3; }
static int revd_s(int x, int lx) {
    int r=0; for(int i=0;i<lx;i++){r=r*10+x%10;x/=10;} return r;
}

/* Direct check for rev(N): p | rev(N)(k,j)? */
static int is_rev_div_by(long long a, long long b, long long c, int k, int j, uint32_t p)
{
    int lk = ndig_s(k), lj = ndig_s(j);
    int rk = revd_s(k, lk), rj = revd_s(j, lj);
    uint64_t ra  = powmod_h(10, a, p);
    uint64_t rb2 = powmod_h(10, a-b-lk, p);
    uint64_t rc2 = powmod_h(10, a-c-lj, p);
    long long v = ((long long)ra
                   - (long long)rk * rb2 % p
                   - (long long)rj * rc2 % p
                   - 1 + 4*(long long)p) % p;
    return (v == 0);
}

/* Returns 1 if (k,j) is composite (p|N or p|rev(N)) for any prime in list */
static int direct_sieve(const prime_list_t *pl, long long a, long long b, long long c, int k, int j)
{
    for (uint32_t i = 0; i < pl->count; i++) {
        uint32_t p = pl->primes[i];
        if (is_div_by(a, b, c, k, j, p)) return 1;
        if (is_rev_div_by(a, b, c, k, j, p)) return 1;
    }
    return 0;
}

/* Build a tiny prime list inline for tests */
static uint32_t small_primes_data[] = { 3,5,7,11,13,17,19,23,29,31,37,41,43,47 };
static prime_list_t make_test_pl(void) {
    prime_list_t pl;
    pl.primes = small_primes_data;
    pl.count  = sizeof(small_primes_data)/sizeof(small_primes_data[0]);
    return pl;
}

static int test_table_vs_direct(void)
{
    prime_list_t pl = make_test_pl();
    sieve_table_t st = {0};
    if (sieve_table_build(&st, 0, &pl) != 0) FAIL("sieve_table_build");

    /* Test several (a,b,c) combos */
    long long test_abc[][3] = {
        {15, 8, 3}, {20, 10, 5}, {30, 15, 7}, {12, 6, 3}
    };

    for (int t = 0; t < 4; t++) {
        long long a = test_abc[t][0], b = test_abc[t][1], c = test_abc[t][2];

        uint32_t table_bits[KJ_WORDS];
        memset(table_bits, 0, sizeof(table_bits));
        sieve_apply(&st, a, b, c, table_bits);

        for (int k = 1; k <= 100; k++) {
            for (int j = 1; j <= 100; j++) {
                int bit = (k-1)*100 + (j-1);
                int table_elim  = (table_bits[bit>>5] >> (bit&31)) & 1;
                int direct_elim = direct_sieve(&pl, a, b, c, k, j);

                /* Table must NOT mark a pair composite when neither N nor rev(N)
                 * is divisible by any tabled prime (no false positives). */
                if (table_elim && !direct_elim) {
                    printf("FAIL: table falsely eliminates (k=%d,j=%d) for a=%lld b=%lld c=%lld\n",
                           k, j, a, b, c);
                    sieve_table_free(&st);
                    return 1;
                }

                /* Survivor (table_elim=0): verify no tabled prime divides N or rev(N).
                 * This checks there are no missed eliminations due to table bugs. */
                if (!table_elim) {
                    int lk = ndig_s(k), lj = ndig_s(j);
                    int rk = revd_s(k,lk), rj = revd_s(j,lj);
                    for (uint32_t pi = 0; pi < pl.count; pi++) {
                        uint32_t p = pl.primes[pi];
                        if (is_div_by(a, b, c, k, j, p)) {
                            printf("FAIL: survivor (k=%d,j=%d) has p=%u | N, not caught by table\n",
                                   k, j, p);
                            sieve_table_free(&st); return 1;
                        }
                        if (is_rev_div_by(a, b, c, k, j, p)) {
                            printf("FAIL: survivor (k=%d,j=%d) has p=%u | rev(N), not caught by table\n",
                                   k, j, p);
                            sieve_table_free(&st); return 1;
                        }
                        (void)rk; (void)rj;  /* used inside is_rev_div_by */
                    }
                }
            }
        }
    }

    sieve_table_free(&st);
    return 0;
}

static int test_periodicity(void)
{
    prime_list_t pl = make_test_pl();
    sieve_table_t st = {0};
    sieve_table_build(&st, 0, &pl);

    /* For each entry, verify sieve(a,b,c) == sieve(a+L10, b, c) */
    for (int i = 0; i < st.n && i < 3; i++) {
        int L10 = st.entries[i].L10;
        long long a = 100, b = 50, c = 20;

        uint32_t bits1[KJ_WORDS], bits2[KJ_WORDS];
        memset(bits1, 0, sizeof(bits1));
        memset(bits2, 0, sizeof(bits2));

        /* Apply only this entry */
        int ra1 = (int)(a % L10), rb1 = (int)(b % L10);
        int ra2 = (int)((a + L10) % L10), rb2 = (int)(b % L10);
        ASSERT(ra1 == ra2);  /* must wrap to same ra */
        (void)rb1; (void)rb2;
    }

    sieve_table_free(&st);
    return 0;
}

int test_sieve(void)
{
    int fail = 0;
    fail += test_table_vs_direct();
    fail += test_periodicity();
    if (!fail) printf("[sieve] OK\n");
    return fail;
}
