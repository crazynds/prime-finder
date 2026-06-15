#pragma once
// correctness_tests.cuh — Testes de corretude usando GMP como referência.
// Ativado em runtime via flag --test.

#include <gmp.h>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <cstring>
#include <cstdlib>
#include "batch_mod_ctx.cuh"
#include "miller_rabin_runner.cuh"

#ifndef CU
#define CU(expr)                                                                                  \
    do                                                                                            \
    {                                                                                             \
        cudaError_t _e = (expr);                                                                  \
        if (_e != cudaSuccess)                                                                    \
            throw std::runtime_error(std::string("[CUDA] " #expr ": ") + cudaGetErrorString(_e)); \
    } while (0)
#endif

// ── Funções auxiliares GMP ────────────────────────────────────────────────────

static void lims_to_gmp(mpz_t out, const uint64_t *lims, int nn)
{
    mpz_set_ui(out, 0);
    for (int j = nn - 1; j >= 0; j--)
    {
        mpz_mul_2exp(out, out, LIMB_BITS);
        mpz_add_ui(out, out, (unsigned long)lims[j]);
    }
}

static void gmp_to_lims(uint64_t *lims, int nn, mpz_t src)
{
    mpz_t t;
    mpz_init_set(t, src);
    for (int j = 0; j < nn; j++)
    {
        lims[j] = mpz_get_ui(t) & LIMB_MASK;
        mpz_tdiv_q_2exp(t, t, LIMB_BITS);
    }
    mpz_clear(t);
}

static bool limbs_eq(const uint64_t *a, const uint64_t *b, int n)
{
    for (int i = 0; i < n; i++)
        if (a[i] != b[i])
            return false;
    return true;
}

static void gmp_sq_mod(uint64_t *out, const uint64_t *x, const uint64_t *N, int n)
{
    mpz_t xm, Nm, res;
    mpz_init(xm);
    mpz_init(Nm);
    mpz_init(res);
    lims_to_gmp(xm, x, n);
    lims_to_gmp(Nm, N, n);
    mpz_mul(res, xm, xm);
    mpz_mod(res, res, Nm);
    gmp_to_lims(out, n, res);
    mpz_clear(xm);
    mpz_clear(Nm);
    mpz_clear(res);
}

static void gmp_mul_mod(uint64_t *out, const uint64_t *x, const uint64_t *y,
                        const uint64_t *N, int n)
{
    mpz_t xm, ym, Nm, res;
    mpz_init(xm);
    mpz_init(ym);
    mpz_init(Nm);
    mpz_init(res);
    lims_to_gmp(xm, x, n);
    lims_to_gmp(ym, y, n);
    lims_to_gmp(Nm, N, n);
    mpz_mul(res, xm, ym);
    mpz_mod(res, res, Nm);
    gmp_to_lims(out, n, res);
    mpz_clear(xm);
    mpz_clear(ym);
    mpz_clear(Nm);
    mpz_clear(res);
}

// ── Testes de corretude ───────────────────────────────────────────────────────

static void run_correctness_tests(BatchModCtx &mont,
                                  const std::vector<uint64_t> &N_all)
{
    int n = mont.n_limbs, nb = mont.n_batch;
    size_t total_bytes = (size_t)nb * n * sizeof(Data64);

    printf("\n=== Testes de corretude (LIMB_BITS=%d, n_batch=%d) ===\n",
           LIMB_BITS, nb);

    std::vector<uint64_t> N_h((size_t)nb * n);
    CU(cudaMemcpy(N_h.data(), mont.d_N, N_h.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // ── Conjuntos de valores de teste ─────────────────────────────────────────
    // small: x = i+2  (valores pequenos)
    // large: x = N - (i+2)  (próximo de N)
    // rand:  hash deterministico, todos os limbs preenchidos, < N

    auto make_large = [&](std::vector<uint64_t> &v, int offset)
    {
        v.assign((size_t)nb * n, 0);
        for (int i = 0; i < nb; i++)
        {
            const uint64_t *Ni = N_h.data() + i * n;
            uint64_t *dst = v.data() + i * n;
            std::copy(Ni, Ni + n, dst);
            uint64_t borrow = (uint64_t)(offset + i + 1);
            for (int j = 0; j < n && borrow > 0; j++)
            {
                if (dst[j] >= borrow)
                {
                    dst[j] -= borrow;
                    borrow = 0;
                }
                else
                {
                    dst[j] = dst[j] + ((1ULL << LIMB_BITS) - borrow);
                    borrow = 1;
                }
            }
        }
    };

    auto make_rand = [&](std::vector<uint64_t> &v, uint64_t seed)
    {
        v.assign((size_t)nb * n, 0);
        for (int i = 0; i < nb; i++)
        {
            mpz_t Nm, rval;
            mpz_init(Nm);
            mpz_init(rval);
            lims_to_gmp(Nm, N_h.data() + i * n, n);
            uint64_t state = seed ^ (uint64_t)(i + 1) * 6364136223846793005ULL;
            if (state == 0)
                state = 1;
            for (int j = 0; j < n; j++)
            {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                v[i * n + j] = state & LIMB_MASK;
            }
            lims_to_gmp(rval, v.data() + i * n, n);
            mpz_mod(rval, rval, Nm);
            gmp_to_lims(v.data() + i * n, n, rval);
            mpz_clear(Nm);
            mpz_clear(rval);
        }
    };

    std::vector<uint64_t> x_small((size_t)nb * n, 0);
    std::vector<uint64_t> y_small((size_t)nb * n, 0);
    for (int i = 0; i < nb; i++)
    {
        x_small[i * n] = (uint64_t)(i + 2) & LIMB_MASK;
        y_small[i * n] = (uint64_t)(i + 3) & LIMB_MASK;
    }

    std::vector<uint64_t> x_large, y_large;
    make_large(x_large, 2);
    make_large(y_large, 3);

    std::vector<uint64_t> x_rand, y_rand;
    make_rand(x_rand, 0xDEADBEEF12345678ULL);
    make_rand(y_rand, 0xCAFEBABE87654321ULL);

    struct TestPair
    {
        const std::vector<uint64_t> *xa;
        const std::vector<uint64_t> *ya;
        const char *name;
    };
    std::vector<TestPair> pairs = {
        {&x_small, &y_small, "small"},
        {&x_large, &y_large, "large"},
        {&x_rand, &y_rand, "rand "},
    };

    // Aloca buffers GPU reutilizados em todos os pares
    Data64 *d_x, *d_y, *d_out;
    CU(cudaMalloc(&d_x, total_bytes));
    CU(cudaMalloc(&d_y, total_bytes));
    CU(cudaMalloc(&d_out, total_bytes));

    auto print_limbs = [&](const char *label, const uint64_t *v, int cnt)
    {
        printf("      %s: [", label);
        for (int k = 0; k < cnt && k < n; k++)
            printf("%llu ", (unsigned long long)v[k]);
        printf("...]\n");
    };

    static constexpr int NPRINT = 6;

    // ══ Loop sobre os 3 conjuntos de valores ══════════════════════════════════
    for (auto &tp : pairs)
    {
        const auto &x_cur = *tp.xa;
        const auto &y_cur = *tp.ya;
        printf("\n  -- valores: %s --\n", tp.name);

        std::vector<uint64_t> x_mont, y_mont;
        mont.to_residue_batch(x_cur, x_mont);
        mont.to_residue_batch(y_cur, y_mont);
        CU(cudaMemcpy(d_x, x_mont.data(), total_bytes, cudaMemcpyHostToDevice));
        CU(cudaMemcpy(d_y, y_mont.data(), total_bytes, cudaMemcpyHostToDevice));

        int pass = 0, fail = 0;

        // ── Teste 0: roundtrip to_mont → from_mont ────────────────────────────
        printf("  [roundtrip] ");
        {
            std::vector<uint64_t> rt_result;
            mont.from_residue_batch(d_x, rt_result);
            int rp = 0, rf = 0;
            for (int i = 0; i < nb; i++)
            {
                if (limbs_eq(rt_result.data() + i * n, x_cur.data() + i * n, n))
                {
                    rp++;
                }
                else
                {
                    rf++;
                    if (rf <= 2)
                    {
                        printf("\n    FAIL cand %d:", i);
                        print_limbs("esp", x_cur.data() + i * n, NPRINT);
                        print_limbs("got", rt_result.data() + i * n, NPRINT);
                    }
                }
            }
            printf("%d/%d OK\n", rp, rp + rf);
            pass += rp;
            fail += rf;
        }

        // ── Teste 0b: NTT raw multiply (x*x sem redução) ─────────────────────
        printf("  [ntt_raw_sq]");
        {
            mont.ntt.ntt_A(d_x, n, 0);
            mont.ntt.psq_and_intt(0);
            CU(cudaDeviceSynchronize());

            int n_sum = mont.n_sum;
            Data64 *d_T_test;
            CU(cudaMalloc(&d_T_test, (size_t)nb * n_sum * sizeof(Data64)));
            mont.ntt.carry_to_limbs(d_T_test, n_sum, 0);
            CU(cudaDeviceSynchronize());

            std::vector<uint64_t> raw_mul((size_t)nb * n_sum);
            CU(cudaMemcpy(raw_mul.data(), d_T_test, raw_mul.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));

            int rp2 = 0, rf2 = 0;
            for (int i = 0; i < nb && i < 2; i++)
            {
                mpz_t xm, res;
                mpz_init(xm);
                mpz_init(res);
                lims_to_gmp(xm, x_mont.data() + i * n, n);
                mpz_mul(res, xm, xm);
                std::vector<uint64_t> ref2(n_sum, 0);
                gmp_to_lims(ref2.data(), n_sum, res);
                mpz_clear(xm);
                mpz_clear(res);
                if (limbs_eq(raw_mul.data() + i * n_sum, ref2.data(), n_sum))
                {
                    rp2++;
                }
                else
                {
                    rf2++;
                    int first_diff = -1;
                    for (int j = 0; j < n_sum; j++)
                        if (raw_mul[i * n_sum + j] != ref2[j])
                        {
                            first_diff = j;
                            break;
                        }
                    printf("\n    FAIL cand %d raw ntt_sq (first diff at limb %d / %d):", i, first_diff, n_sum);
                    if (first_diff >= 0)
                    {
                        int start = std::max(0, first_diff - 1);
                        printf("\n      esperado[%d..]: ", start);
                        for (int j = start; j < std::min(n_sum, start + 8); j++)
                            printf("%llu ", (unsigned long long)ref2[j]);
                        printf("\n      obtido  [%d..]: ", start);
                        for (int j = start; j < std::min(n_sum, start + 8); j++)
                            printf("%llu ", (unsigned long long)raw_mul[i * n_sum + j]);
                        printf("\n");
                    }
                }
            }
            printf("%d/2 OK\n", rp2);
            CU(cudaFree(d_T_test));
            // Restaura d_x com x_mont
            CU(cudaMemcpy(d_x, x_mont.data(), total_bytes, cudaMemcpyHostToDevice));
        }

        // ── Teste 1: mont_sq ──────────────────────────────────────────────────
        pass = 0;
        fail = 0;
        printf("  [mont_sq]  ");
        mont.modsq_batch(d_x, d_out);
        CU(cudaDeviceSynchronize());
        {
            std::vector<uint64_t> sq_result;
            mont.from_residue_batch(d_out, sq_result);
            for (int i = 0; i < nb; i++)
            {
                std::vector<uint64_t> ref(n, 0);
                gmp_sq_mod(ref.data(), x_cur.data() + i * n, N_h.data() + i * n, n);
                if (limbs_eq(sq_result.data() + i * n, ref.data(), n))
                {
                    pass++;
                }
                else
                {
                    fail++;
                    if (fail <= 2)
                    {
                        printf("\n    FAIL cand %d (x=%llu):", i, (unsigned long long)x_cur[i * n]);
                        print_limbs("esperado", ref.data(), NPRINT);
                        print_limbs("obtido  ", sq_result.data() + i * n, NPRINT);
                        std::vector<uint64_t> raw_out((size_t)nb * n);
                        CU(cudaMemcpy(raw_out.data(), d_out, (size_t)nb * n * sizeof(uint64_t), cudaMemcpyDeviceToHost));
                        print_limbs("raw GPU ", raw_out.data() + i * n, NPRINT);
                    }
                }
            }
            printf("%d/%d OK\n", pass, pass + fail);
        }

        // ── Teste 2: mont_mul ─────────────────────────────────────────────────
        pass = 0;
        fail = 0;
        printf("  [mont_mul] ");
        CU(cudaMemcpy(d_x, x_mont.data(), total_bytes, cudaMemcpyHostToDevice));
        mont.modmul_batch(d_x, d_y, d_out);
        CU(cudaDeviceSynchronize());
        {
            std::vector<uint64_t> mul_result;
            mont.from_residue_batch(d_out, mul_result);
            for (int i = 0; i < nb; i++)
            {
                std::vector<uint64_t> ref(n, 0);
                gmp_mul_mod(ref.data(), x_cur.data() + i * n, y_cur.data() + i * n, N_h.data() + i * n, n);
                if (limbs_eq(mul_result.data() + i * n, ref.data(), n))
                {
                    pass++;
                }
                else
                {
                    fail++;
                    if (fail <= 2)
                        printf("\n    FAIL cand %d: x=%llu y=%llu, esperado x*y%%N != obtido",
                               i, (unsigned long long)x_cur[i * n], (unsigned long long)y_cur[i * n]);
                }
            }
            printf("%d/%d OK\n", pass, pass + fail);
        }

        // ── Teste 3: cadeia de 16 squarings (x^(2^16) mod N) ─────────────────
        pass = 0;
        fail = 0;
        printf("  [sq x16]   ");
        {
            CU(cudaMemcpy(d_x, x_mont.data(), total_bytes, cudaMemcpyHostToDevice));
            Data64 *cur = d_x;
            Data64 *tmp = d_out;
            for (int k = 0; k < 16; k++)
            {
                mont.modsq_batch(cur, tmp);
                std::swap(cur, tmp);
            }
            CU(cudaDeviceSynchronize());
            std::vector<uint64_t> res_gpu;
            mont.from_residue_batch(cur, res_gpu);
            for (int i = 0; i < nb; i++)
            {
                mpz_t xm, Nm, res;
                mpz_init(xm);
                mpz_init(Nm);
                mpz_init(res);
                lims_to_gmp(xm, x_cur.data() + i * n, n);
                lims_to_gmp(Nm, N_h.data() + i * n, n);
                mpz_set(res, xm);
                for (int k = 0; k < 16; k++)
                {
                    mpz_mul(res, res, res);
                    mpz_mod(res, res, Nm);
                }
                std::vector<uint64_t> ref(n, 0);
                gmp_to_lims(ref.data(), n, res);
                mpz_clear(xm);
                mpz_clear(Nm);
                mpz_clear(res);
                if (limbs_eq(res_gpu.data() + i * n, ref.data(), n))
                    pass++;
                else
                {
                    fail++;
                    if (fail <= 2)
                    {
                        printf("\n    FAIL cand %d", i);
                        print_limbs("esp", ref.data(), NPRINT);
                        print_limbs("gpu", res_gpu.data() + i * n, NPRINT);
                    }
                }
            }
            printf("%d/%d OK\n", pass, pass + fail);
        }

        // ── Teste 4: mont_sq(x) == mont_mul(x, x) ────────────────────────────
        pass = 0;
        fail = 0;
        printf("  [sq==mul]  ");
        {
            CU(cudaMemcpy(d_x, x_mont.data(), total_bytes, cudaMemcpyHostToDevice));
            CU(cudaMemcpy(d_y, x_mont.data(), total_bytes, cudaMemcpyHostToDevice));
            Data64 *d_sq_out;
            CU(cudaMalloc(&d_sq_out, total_bytes));
            mont.modsq_batch(d_x, d_out);
            mont.modmul_batch(d_x, d_y, d_sq_out);
            CU(cudaDeviceSynchronize());
            std::vector<uint64_t> sq_r, mul_r;
            mont.from_residue_batch(d_out, sq_r);
            mont.from_residue_batch(d_sq_out, mul_r);
            for (int i = 0; i < nb; i++)
            {
                if (limbs_eq(sq_r.data() + i * n, mul_r.data() + i * n, n))
                    pass++;
                else
                {
                    fail++;
                    if (fail <= 2)
                    {
                        printf("\n    FAIL cand %d: sq!=mul", i);
                        print_limbs("sq ", sq_r.data() + i * n, NPRINT);
                        print_limbs("mul", mul_r.data() + i * n, NPRINT);
                    }
                }
            }
            cudaFree(d_sq_out);
            printf("%d/%d OK\n", pass, pass + fail);
        }

        // ── Teste 5: (N-1)^2 mod N == 1 ──────────────────────────────────────
        pass = 0;
        fail = 0;
        printf("  [(N-1)^2]  ");
        {
            std::vector<uint64_t> nm1_all((size_t)nb * n, 0);
            for (int i = 0; i < nb; i++)
            {
                const uint64_t *Ni = N_h.data() + i * n;
                uint64_t *out2 = nm1_all.data() + i * n;
                std::copy(Ni, Ni + n, out2);
                for (int j = 0; j < n; j++)
                {
                    if (out2[j] > 0)
                    {
                        out2[j]--;
                        break;
                    }
                    out2[j] = LIMB_MASK;
                }
            }
            std::vector<uint64_t> nm1_mont;
            mont.to_residue_batch(nm1_all, nm1_mont);
            CU(cudaMemcpy(d_x, nm1_mont.data(), total_bytes, cudaMemcpyHostToDevice));
            mont.modsq_batch(d_x, d_out);
            CU(cudaDeviceSynchronize());
            std::vector<uint64_t> nm1sq;
            mont.from_residue_batch(d_out, nm1sq);
            for (int i = 0; i < nb; i++)
            {
                std::vector<uint64_t> one(n, 0);
                one[0] = 1;
                if (limbs_eq(nm1sq.data() + i * n, one.data(), n))
                    pass++;
                else
                {
                    fail++;
                    if (fail <= 2)
                    {
                        printf("\n    FAIL cand %d: (N-1)^2%%N != 1", i);
                        print_limbs("got", nm1sq.data() + i * n, NPRINT);
                    }
                }
            }
            printf("%d/%d OK\n", pass, pass + fail);
        }

        // ── Teste 6: comutatividade mont_mul(x,y) == mont_mul(y,x) ──────────
        pass = 0;
        fail = 0;
        printf("  [commut]   ");
        {
            CU(cudaMemcpy(d_x, x_mont.data(), total_bytes, cudaMemcpyHostToDevice));
            CU(cudaMemcpy(d_y, y_mont.data(), total_bytes, cudaMemcpyHostToDevice));
            Data64 *d_xy, *d_yx;
            CU(cudaMalloc(&d_xy, total_bytes));
            CU(cudaMalloc(&d_yx, total_bytes));
            mont.modmul_batch(d_x, d_y, d_xy);
            mont.modmul_batch(d_y, d_x, d_yx);
            CU(cudaDeviceSynchronize());
            std::vector<uint64_t> xy_h((size_t)nb * n), yx_h((size_t)nb * n);
            CU(cudaMemcpy(xy_h.data(), d_xy, total_bytes, cudaMemcpyDeviceToHost));
            CU(cudaMemcpy(yx_h.data(), d_yx, total_bytes, cudaMemcpyDeviceToHost));
            for (int i = 0; i < nb; i++)
            {
                if (limbs_eq(xy_h.data() + i * n, yx_h.data() + i * n, n))
                    pass++;
                else
                {
                    fail++;
                    if (fail <= 2)
                        printf("\n    FAIL cand %d: xy!=yx", i);
                }
            }
            cudaFree(d_xy);
            cudaFree(d_yx);
            printf("%d/%d OK\n", pass, pass + fail);
        }

        // ── Teste 7: identidade mont_mul(x, 1_mont) == x ─────────────────────
        pass = 0;
        fail = 0;
        printf("  [identity] ");
        {
            std::vector<uint64_t> one_plain((size_t)nb * n, 0);
            for (int i = 0; i < nb; i++)
                one_plain[i * n] = 1;
            std::vector<uint64_t> one_mont_v;
            mont.to_residue_batch(one_plain, one_mont_v);
            Data64 *d_one2;
            CU(cudaMalloc(&d_one2, total_bytes));
            CU(cudaMemcpy(d_x, x_mont.data(), total_bytes, cudaMemcpyHostToDevice));
            CU(cudaMemcpy(d_one2, one_mont_v.data(), total_bytes, cudaMemcpyHostToDevice));
            mont.modmul_batch(d_x, d_one2, d_out);
            CU(cudaDeviceSynchronize());
            std::vector<uint64_t> id_res;
            mont.from_residue_batch(d_out, id_res);
            for (int i = 0; i < nb; i++)
            {
                if (limbs_eq(id_res.data() + i * n, x_cur.data() + i * n, n))
                    pass++;
                else
                {
                    fail++;
                    if (fail <= 2)
                    {
                        printf("\n    FAIL cand %d: x*1 != x", i);
                        print_limbs("esp", x_cur.data() + i * n, NPRINT);
                        print_limbs("got", id_res.data() + i * n, NPRINT);
                    }
                }
            }
            cudaFree(d_one2);
            printf("%d/%d OK\n", pass, pass + fail);
        }

        // ── Teste 8: modpow com expoente fixo de 32 bits vs GMP ──────────────
        pass = 0;
        fail = 0;
        printf("  [modpow32] ");
        {
            const uint32_t EXP32 = 0xDEADBEEFU;
            std::vector<uint64_t> one_plain((size_t)nb * n, 0);
            for (int i = 0; i < nb; i++)
                one_plain[i * n] = 1;
            std::vector<uint64_t> one_m;
            mont.to_residue_batch(one_plain, one_m);
            Data64 *d_acc, *d_base32, *d_tmp32;
            CU(cudaMalloc(&d_acc, total_bytes));
            CU(cudaMalloc(&d_base32, total_bytes));
            CU(cudaMalloc(&d_tmp32, total_bytes));
            CU(cudaMemcpy(d_acc, one_m.data(), total_bytes, cudaMemcpyHostToDevice));
            CU(cudaMemcpy(d_base32, x_mont.data(), total_bytes, cudaMemcpyHostToDevice));
            for (int bit = 31; bit >= 0; bit--)
            {
                mont.modsq_batch(d_acc, d_tmp32);
                std::swap(d_acc, d_tmp32);
                if ((EXP32 >> bit) & 1)
                {
                    mont.modmul_batch(d_acc, d_base32, d_tmp32);
                    std::swap(d_acc, d_tmp32);
                }
            }
            CU(cudaDeviceSynchronize());
            std::vector<uint64_t> modpow_gpu;
            mont.from_residue_batch(d_acc, modpow_gpu);
            for (int i = 0; i < nb; i++)
            {
                mpz_t xm, Nm, res;
                mpz_init(xm);
                mpz_init(Nm);
                mpz_init(res);
                lims_to_gmp(xm, x_cur.data() + i * n, n);
                lims_to_gmp(Nm, N_h.data() + i * n, n);
                mpz_powm_ui(res, xm, (unsigned long)EXP32, Nm);
                std::vector<uint64_t> ref(n, 0);
                gmp_to_lims(ref.data(), n, res);
                mpz_clear(xm);
                mpz_clear(Nm);
                mpz_clear(res);
                if (limbs_eq(modpow_gpu.data() + i * n, ref.data(), n))
                    pass++;
                else
                {
                    fail++;
                    if (fail <= 2)
                    {
                        printf("\n    FAIL cand %d: x^0xDEADBEEF%%N errado", i);
                        print_limbs("esp", ref.data(), NPRINT);
                        print_limbs("got", modpow_gpu.data() + i * n, NPRINT);
                    }
                }
            }
            cudaFree(d_acc);
            cudaFree(d_base32);
            cudaFree(d_tmp32);
            printf("%d/%d OK\n", pass, pass + fail);
        }

        // ── Teste 9: window modpow vs sq-and-mul simples (expoente 32-bit) ────
        pass = 0;
        fail = 0;
        printf("  [window_ok] ");
        {
            const uint32_t EXP32W = 0xCAFEBABEU;
            std::vector<uint64_t> one_plain2((size_t)nb * n, 0);
            for (int i = 0; i < nb; i++)
                one_plain2[i * n] = 1;
            std::vector<uint64_t> one_m2;
            mont.to_residue_batch(one_plain2, one_m2);
            Data64 *d_ref_acc, *d_win_acc, *d_base_w, *d_tmp_w;
            CU(cudaMalloc(&d_ref_acc, total_bytes));
            CU(cudaMalloc(&d_win_acc, total_bytes));
            CU(cudaMalloc(&d_base_w, total_bytes));
            CU(cudaMalloc(&d_tmp_w, total_bytes));
            CU(cudaMemcpy(d_ref_acc, one_m2.data(), total_bytes, cudaMemcpyHostToDevice));
            CU(cudaMemcpy(d_base_w, x_mont.data(), total_bytes, cudaMemcpyHostToDevice));
            // Referência: sq-and-mul bit-a-bit
            for (int bit = 31; bit >= 0; bit--)
            {
                mont.modsq_batch(d_ref_acc, d_tmp_w);
                std::swap(d_ref_acc, d_tmp_w);
                if ((EXP32W >> bit) & 1)
                {
                    mont.modmul_batch(d_ref_acc, d_base_w, d_tmp_w);
                    std::swap(d_ref_acc, d_tmp_w);
                }
            }
            // Janela k=4
            {
                constexpr int K = 4, SZ = 1 << K;
                std::vector<Data64 *> tbl(SZ);
                for (int w = 0; w < SZ; w++)
                    CU(cudaMalloc(&tbl[w], total_bytes));
                CU(cudaMemcpy(tbl[0], one_m2.data(), total_bytes, cudaMemcpyHostToDevice));
                CU(cudaMemcpy(tbl[1], x_mont.data(), total_bytes, cudaMemcpyHostToDevice));
                for (int w = 2; w < SZ; w++)
                    mont.modmul_batch(tbl[w - 1], d_base_w, tbl[w]);
                Data64 *d_table_w;
                CU(cudaMalloc(&d_table_w, (size_t)SZ * total_bytes));
                for (int w = 0; w < SZ; w++)
                    CU(cudaMemcpy(d_table_w + (size_t)w * nb * n, tbl[w], total_bytes, cudaMemcpyDeviceToDevice));
                std::vector<uint64_t> exp32w_all((size_t)nb * n, 0);
                for (int i = 0; i < nb; i++)
                {
                    uint32_t e = EXP32W;
                    for (int j = 0; j < (int)(32 / LIMB_BITS) && j < n; j++)
                    {
                        exp32w_all[i * n + j] = e & LIMB_MASK;
                        e >>= LIMB_BITS;
                    }
                }
                Data64 *d_exp_w;
                CU(cudaMalloc(&d_exp_w, total_bytes));
                CU(cudaMemcpy(d_exp_w, exp32w_all.data(), total_bytes, cudaMemcpyHostToDevice));
                Data64 *d_cur_w;
                CU(cudaMalloc(&d_cur_w, total_bytes));
                CU(cudaMemcpy(d_win_acc, one_m2.data(), total_bytes, cudaMemcpyHostToDevice));
                const int thr = 256;
                dim3 gsel((unsigned)(n + thr - 1) / thr, (unsigned)nb);
                int n_win32 = (32 + K - 1) / K;
                int start32 = n_win32 * K - 1;
                for (int win = 0; win < n_win32; win++)
                {
                    int msb_pos = start32 - win * K;
                    for (int sq = 0; sq < K; sq++)
                    {
                        mont.modsq_batch(d_win_acc, d_tmp_w);
                        std::swap(d_win_acc, d_tmp_w);
                    }
                    select_window_kernel<<<gsel, thr>>>(d_cur_w, d_table_w, d_exp_w, msb_pos, K, n, nb);
                    mont.modmul_batch(d_win_acc, d_cur_w, d_tmp_w);
                    std::swap(d_win_acc, d_tmp_w);
                }
                CU(cudaDeviceSynchronize());
                for (int w = 0; w < SZ; w++)
                    cudaFree(tbl[w]);
                cudaFree(d_table_w);
                cudaFree(d_exp_w);
                cudaFree(d_cur_w);
            }
            std::vector<uint64_t> ref_r, win_r;
            mont.from_residue_batch(d_ref_acc, ref_r);
            mont.from_residue_batch(d_win_acc, win_r);
            for (int i = 0; i < nb; i++)
            {
                if (limbs_eq(ref_r.data() + i * n, win_r.data() + i * n, n))
                    pass++;
                else
                {
                    fail++;
                    if (fail <= 2)
                    {
                        printf("\n    FAIL cand %d: window k=4 != sq-mul", i);
                        print_limbs("sq-mul", ref_r.data() + i * n, NPRINT);
                        print_limbs("window", win_r.data() + i * n, NPRINT);
                    }
                }
            }
            cudaFree(d_ref_acc);
            cudaFree(d_win_acc);
            cudaFree(d_base_w);
            cudaFree(d_tmp_w);
            printf("%d/%d OK\n", pass, pass + fail);
        }

    } // fim do loop sobre pares

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_out);
    printf("\n=== Fim dos testes ===\n\n");
}

// ── Testa primos de Mersenne conhecidos ───────────────────────────────────────
//
// Usa primos de Mersenne M_p = 2^p - 1 confirmados, todos com > 128 bits.
// Todos têm s=1 (M_p - 1 = 2*(2^(p-1)-1), com 2^(p-1)-1 ímpar).
// Espera-se que todos passem em todos os witnesses.

static void run_known_prime_tests()
{
    // Expoentes de primos de Mersenne conhecidos com > 128 bits
    static const int MERSENNE_EXP[] = {521, 607, 1279};
    static const int N_MERSENNE = (int)(sizeof(MERSENNE_EXP) / sizeof(MERSENNE_EXP[0]));

    printf("\n=== Testes com primos de Mersenne conhecidos ===\n");
    printf("  M_p = 2^p - 1 para p = 521, 607, 1279\n\n");

    // n_limbs baseado no maior (M1279, ~386 dígitos decimais)
    int max_digits = (int)(MERSENNE_EXP[N_MERSENNE - 1] * 0.30103) + 4;
    int n_limbs = limbs_for_digits(max_digits + 4);

    int nb = N_MERSENNE;
    std::vector<uint64_t> N_all((size_t)nb * n_limbs, 0);
    std::vector<uint64_t> Nm1_all((size_t)nb * n_limbs, 0);
    std::vector<uint64_t> d_all((size_t)nb * n_limbs, 0);

    mpz_t M, Mm1, d;
    mpz_inits(M, Mm1, d, nullptr);
    for (int i = 0; i < nb; i++)
    {
        // M = 2^p - 1
        mpz_ui_pow_ui(M, 2, (unsigned long)MERSENNE_EXP[i]);
        mpz_sub_ui(M, M, 1);

        // d = (M-1)/2  (s=1 para todos os Mersenne)
        mpz_sub_ui(Mm1, M, 1);
        mpz_tdiv_q_2exp(d, Mm1, 1);

        auto to_lims = [&](uint64_t *out, const mpz_t x)
        {
            mpz_t tmp;
            mpz_init_set(tmp, x);
            for (int j = 0; j < n_limbs; j++)
            {
                out[j] = mpz_get_ui(tmp) & LIMB_MASK;
                mpz_tdiv_q_2exp(tmp, tmp, LIMB_BITS);
            }
            mpz_clear(tmp);
        };

        to_lims(N_all.data() + i * n_limbs, M);
        to_lims(Nm1_all.data() + i * n_limbs, Mm1);
        to_lims(d_all.data() + i * n_limbs, d);
    }
    mpz_clears(M, Mm1, d, nullptr);

    printf("  n_limbs=%d  NTT padded=%d\n", n_limbs, next_pow2_ntt(2 * n_limbs));

    BatchModCtx mont(N_all, n_limbs, nb);
    auto alive = gpu_miller_rabin_s1(mont, d_all, Nm1_all, nb, DEFAULT_WITNESSES, "Mersenne");

    printf("\n  Resultados:\n");
    int ok = 0, fail = 0;
    for (int i = 0; i < nb; i++)
    {
        bool passed = alive[i];
        printf("  M%-4d (2^%d-1): %s\n",
               MERSENNE_EXP[i], MERSENNE_EXP[i], passed ? "PRIMO OK" : "FALHOU (bug!)");
        if (passed)
            ok++;
        else
            fail++;
    }

    if (fail == 0)
        printf("\n  Todos os %d primos de Mersenne identificados corretamente.\n", ok);
    else
        printf("\n  ERRO: %d primo(s) nao identificado(s) — bug no algoritmo!\n", fail);

    printf("=== Fim dos testes com Mersenne ===\n\n");
}

// ── Testa primos conhecidos com s != 1 ────────────────────────────────────────
//
// Gera primos deterministicamente via GMP a partir de pontos fixos, buscando
// primos com s=2, s=3, e s>=4 (cada um > 512 bits / ~155 dígitos decimais).
// Verifica que run_witnesses_general os identifica corretamente como primos.

static void run_general_s_prime_tests()
{
    printf("\n=== Testes com primos de s != 1 ===\n");
    printf("  Gerando primos com s=2, s=3, s>=4 via GMP (> 512 bits)...\n\n");

    // Ponto de partida fixo acima de 2^512 para cada busca
    // Cada valor é ajustado para garantir residuos mod 8 que favorecem o s alvo,
    // acelerando a busca (mas s é verificado rigorosamente após encontrar o primo).
    //
    //   p ≡ 5 mod 8  →  p-1 ≡ 4 mod 8  →  s=2 exato
    //   p ≡ 9 mod 16 →  p-1 ≡ 8 mod 16 →  s=3 exato
    //   p ≡ 1 mod 16 →  p-1 ≡ 0 mod 16 →  s>=4
    struct Target
    {
        unsigned long start_offset; // 2^512 + start_offset
        int want_s_min, want_s_max;
        const char *desc;
    };
    static const Target TARGETS[] = {
        {4UL, 2, 2, "s=2"},    // 2^512+4 ≡ 4 mod 8, próximo primo impar perto
        {7UL, 3, 3, "s=3"},    // partida próxima de 9 mod 16
        {15UL, 4, 99, "s>=4"}, // partida próxima de 1 mod 16
    };
    static const int N_TARGETS = (int)(sizeof(TARGETS) / sizeof(TARGETS[0]));

    int n_limbs = limbs_for_digits(160 + 4); // ~155 dígitos (> 512 bits)

    printf("  n_limbs=%d  NTT padded=%d\n\n", n_limbs, next_pow2_ntt(2 * n_limbs));

    mpz_t base, p, Nm1, d_tmp;
    mpz_inits(base, p, Nm1, d_tmp, nullptr);
    mpz_ui_pow_ui(base, 2, 512);

    int total_ok = 0, total_fail = 0;

    for (int ti = 0; ti < N_TARGETS; ti++)
    {
        const Target &tgt = TARGETS[ti];

        // Encontra o próximo primo a partir do ponto fixo com s no intervalo desejado
        mpz_add_ui(p, base, tgt.start_offset);
        if (mpz_even_p(p))
            mpz_add_ui(p, p, 1);

        int found_s = -1;
        int attempts = 0;
        while (found_s < tgt.want_s_min || found_s > tgt.want_s_max)
        {
            mpz_nextprime(p, p);
            mpz_sub_ui(Nm1, p, 1);
            mpz_set(d_tmp, Nm1);
            found_s = 0;
            while (mpz_even_p(d_tmp))
            {
                mpz_tdiv_q_2exp(d_tmp, d_tmp, 1);
                found_s++;
            }
            attempts++;
            if (attempts > 2000)
            {
                found_s = -1;
                break;
            }
        }

        if (found_s == -1)
        {
            printf("  [%s] SKIP — nao encontrado em 2000 tentativas\n", tgt.desc);
            continue;
        }

        // Verifica via GMP que p é de fato primo
        if (!mpz_probab_prime_p(p, 25))
        {
            printf("  [%s] ERRO interno: numero encontrado nao eh primo segundo GMP!\n", tgt.desc);
            total_fail++;
            continue;
        }

        // Imprime informação do primo encontrado
        char *p_str = mpz_get_str(nullptr, 10, p);
        int p_digits = (int)strlen(p_str);
        free(p_str);
        printf("  [%s] primo com %d dígitos encontrado em %d tentativas (s=%d)\n",
               tgt.desc, p_digits, attempts, found_s);

        // Constrói NumberCandidate e executa o teste
        NumberCandidate cand;
        cand.build_from_mpz(p, n_limbs);

        BatchModCtx mont(cand.N_lims, n_limbs, 1);
        auto alive = gpu_miller_rabin(mont, cand.d_lims, cand.Nm1_lims,
                                      cand.s, 1, DEFAULT_WITNESSES, tgt.desc);

        bool passed = alive[0];
        printf("  [%s] resultado: %s\n\n", tgt.desc, passed ? "PRIMO OK" : "FALHOU (bug!)");
        if (passed)
            total_ok++;
        else
            total_fail++;
    }

    mpz_clears(base, p, Nm1, d_tmp, nullptr);

    if (total_fail == 0)
        printf("  Todos os %d primos (s!=1) identificados corretamente.\n", total_ok);
    else
        printf("  ERRO: %d primo(s) nao identificado(s) — bug no algoritmo!\n", total_fail);

    printf("=== Fim dos testes com s!=1 ===\n\n");
}

// ── Testa primos com s=1 gerados via mpz_nextprime ────────────────────────────
//
// Busca primos p ≡ 3 (mod 4) acima de 2^512, que garantem s=1 (p-1 = 2*d,
// d ímpar). Verifica via mpz_probab_prime_p e depois via gpu_miller_rabin.

static void run_s1_nextprime_tests()
{
    printf("\n=== Testes com primos s=1 (mpz_nextprime, > 512 bits) ===\n");

    // p ≡ 3 mod 4 → p-1 ≡ 2 mod 4 → s=1 exato
    // Partimos de 2^512 + 3 e avançamos até encontrar 3 primos com s=1.
    static const int N_WANT = 3;
    int n_limbs = limbs_for_digits(160 + 4);

    printf("  n_limbs=%d  NTT padded=%d\n\n", n_limbs, next_pow2_ntt(2 * n_limbs));

    mpz_t base, p, Nm1, d_tmp;
    mpz_inits(base, p, Nm1, d_tmp, nullptr);
    mpz_ui_pow_ui(base, 2, 512);
    mpz_add_ui(p, base, 3); // começa em 2^512+3

    int total_ok = 0, total_fail = 0, found = 0;

    while (found < N_WANT)
    {
        mpz_nextprime(p, p);

        // Verifica s=1
        mpz_sub_ui(Nm1, p, 1);
        mpz_set(d_tmp, Nm1);
        int s = 0;
        while (mpz_even_p(d_tmp))
        {
            mpz_tdiv_q_2exp(d_tmp, d_tmp, 1);
            s++;
        }
        if (s != 1)
            continue;

        found++;

        // Confirmação independente via GMP
        if (!mpz_probab_prime_p(p, 25))
        {
            printf("  [s=1 #%d] ERRO interno: numero nao eh primo segundo GMP!\n", found);
            total_fail++;
            continue;
        }

        char *p_str = mpz_get_str(nullptr, 10, p);
        int p_digits = (int)strlen(p_str);
        free(p_str);
        printf("  [s=1 #%d] primo com %d dígitos (s=1)\n", found, p_digits);

        NumberCandidate cand;
        cand.build_from_mpz(p, n_limbs);

        BatchModCtx mont(cand.N_lims, n_limbs, 1);
        auto alive = gpu_miller_rabin(mont, cand.d_lims, cand.Nm1_lims,
                                      cand.s, 1, DEFAULT_WITNESSES, "s=1");

        bool passed = alive[0];
        printf("  [s=1 #%d] resultado: %s\n\n", found, passed ? "PRIMO OK" : "FALHOU (bug!)");
        if (passed)
            total_ok++;
        else
            total_fail++;
    }

    mpz_clears(base, p, Nm1, d_tmp, nullptr);

    if (total_fail == 0)
        printf("  Todos os %d primos s=1 identificados corretamente.\n", total_ok);
    else
        printf("  ERRO: %d primo(s) nao identificado(s) — bug no algoritmo!\n", total_fail);

    printf("=== Fim dos testes s=1 ===\n\n");
}
