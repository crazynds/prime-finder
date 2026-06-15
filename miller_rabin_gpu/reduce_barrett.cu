// reduce_barrett.cu — Redução de Barrett batched.
//
// Forma de trabalho: resíduo plano (x mod N). Pré-computa μ_i = floor(b^{2k_i}/N_i)
// por candidato (b = 2^LIMB_BITS, k_i = limbs tight de N_i) e reduz via 2 multiplicações
// NTT + finalize. Compilado apenas quando MOD_REDUCTION_ALG == MOD_RED_BARRETT.

#include "batch_mod_ctx.cuh"
#include "gmp_helpers.cuh"
#include "mod_internal.cuh"
#include <vector>
#include <gmp.h>

#if MOD_REDUCTION_ALG == MOD_RED_BARRETT

// ── kernels ───────────────────────────────────────────────────────────────────

// Shift à direita com offset POR-CANDIDATO: dst_i = src_i >> (bar_k[i] + delta) limbs.
__global__ static void shift_right_var_batch(
    Data64 *__restrict__ dst,
    const Data64 *__restrict__ src,
    const int *__restrict__ bark, int delta,
    int n_out, int n_src, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= n_out)
        return;
    int off = bark[cand] + delta;
    int s = j + off;
    dst[cand * n_out + j] = (s >= 0 && s < n_src) ? src[cand * n_src + s] : 0ULL;
}

// ── Finalização paralela (subtração tileada com prefix-scan de borrow) ────────
//
// O finalize do Barrett é r = T − q̂·N seguido de até 2 subtrações condicionais
// de N (pois 0 <= r_true < 3N). Em vez de 1 thread/candidato serial sobre W1
// limbs (gargalo quando W1 ~ milhares), usamos o mesmo esquema tileado G/P/K do
// cond_sub do Montgomery: grid (n_tiles × n_batch), prefix scan O(log CS_TILE).
//
// Os kernels são genéricos (strides separados para a/b/out e largura de b por
// candidato) para servir tanto à subtração incondicional T−qn quanto às
// subtrações condicionais de N:
//   • T−qn: incondicional (r_true >= 0 ⇒ borrow_out = 0), b = qn largura W1.
//   • r−N:  condicional (só se r >= N),                   b = N  largura bar_k.
static constexpr int CS_TILE = MR_SUB_TILE;

// Composição de estados G/P/K de borrow: state = bw0 | (bw1 << 1).
__device__ static int bar_cs_combine(int L, int R)
{
    int c0 = (R >> (L & 1)) & 1;
    int c1 = (R >> ((L >> 1) & 1)) & 1;
    return c0 | (c1 << 1);
}

// Kernel 1: por tile, comparação a vs b e estado G/P/K de borrow do tile.
// bk: largura efetiva de b por candidato (nullptr ⇒ usa W). sa/sb: strides.
__global__ static void bar_sub_phase1(
    const Data64 *__restrict__ a, int sa,
    const Data64 *__restrict__ b, int sb,
    const int *__restrict__ bk, int W,
    int *__restrict__ tile_cmp, int *__restrict__ tile_bstate, int n_batch)
{
    __shared__ Data64 s_a[CS_TILE];
    __shared__ Data64 s_b[CS_TILE];
    __shared__ int s_reduce[CS_TILE];
    __shared__ int s_state[CS_TILE];

    int cand = blockIdx.y, tile = blockIdx.x, tid = threadIdx.x;
    int j = tile * CS_TILE + tid;
    int n_tiles = (W + CS_TILE - 1) / CS_TILE;
    if (cand >= n_batch)
        return;

    int bw = bk ? bk[cand] : W;
    s_a[tid] = (j < W) ? a[(size_t)cand * sa + j] : 0ULL;
    s_b[tid] = (j < W && j < bw) ? b[(size_t)cand * sb + j] : 0ULL;
    __syncthreads();

    // Comparação: encode (j+1)<<2 | (cmp+1) para redução de máximo.
    int enc = 1;
    if (j < W && s_a[tid] != s_b[tid])
        enc = ((j + 1) << 2) | ((s_a[tid] > s_b[tid]) ? 2 : 0);
    s_reduce[tid] = enc;
    __syncthreads();
    for (int stride = CS_TILE >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride && s_reduce[tid + stride] > s_reduce[tid])
            s_reduce[tid] = s_reduce[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        tile_cmp[cand * n_tiles + tile] = (s_reduce[0] == 1) ? 0 : ((s_reduce[0] & 3) - 1);

    int bw0 = (j < W && s_a[tid] < s_b[tid]) ? 1 : 0;
    int bw1 = (j < W && s_a[tid] <= s_b[tid]) ? 1 : 0;
    s_state[tid] = bw0 | (bw1 << 1);
    __syncthreads();
    for (int stride = 1; stride < CS_TILE; stride <<= 1)
    {
        int combined = (tid >= stride) ? bar_cs_combine(s_state[tid - stride], s_state[tid])
                                       : s_state[tid];
        __syncthreads();
        s_state[tid] = combined;
        __syncthreads();
    }
    int last = min(CS_TILE, W - tile * CS_TILE) - 1;
    if (tid == last)
        tile_bstate[cand * n_tiles + tile] = s_state[tid];
}

// Kernel 2 (fundido): por tile, resolve o borrow_in do próprio tile e aplica
// out = a − b. A thread 0 lê os n_tiles (~poucos) resultados de phase1 e:
//   • se !uncond: acha o cmp global (a >= b?); se a < b, marca skip.
//   • replay da cadeia de borrow dos tiles [0..tile) → borrow_in deste tile.
// Isso elimina o kernel resolve de 1-thread/bloco (ocupância 1/CS_TILE).
// uncond != 0 ⇒ subtração sempre aplicada (T−qn), ignorando a comparação.
// so: stride de out.
__global__ static void bar_sub_apply(
    Data64 *__restrict__ out, int so,
    const Data64 *__restrict__ a, int sa,
    const Data64 *__restrict__ b, int sb,
    const int *__restrict__ bk, int W,
    const int *__restrict__ tile_cmp, const int *__restrict__ tile_bstate,
    int uncond, int n_batch)
{
    __shared__ Data64 s_a[CS_TILE];
    __shared__ Data64 s_b[CS_TILE];
    __shared__ int s_state[CS_TILE];
    __shared__ int s_tile_bin; // borrow_in deste tile (-1 ⇒ sem subtração)

    int cand = blockIdx.y, tile = blockIdx.x, tid = threadIdx.x;
    int j = tile * CS_TILE + tid;
    int n_tiles = (W + CS_TILE - 1) / CS_TILE;
    if (cand >= n_batch)
        return;

    if (tid == 0)
    {
        const int *cmp = tile_cmp + cand * n_tiles;
        const int *bstate = tile_bstate + cand * n_tiles;
        int do_sub = 1;
        if (!uncond)
        {
            int gcmp = 0;
            for (int t = n_tiles - 1; t >= 0 && gcmp == 0; t--)
                gcmp = cmp[t];
            if (gcmp < 0)
                do_sub = 0; // a < b
        }
        int bin = -1;
        if (do_sub)
        {
            int cur = 0; // replay do borrow até o tile atual
            for (int t = 0; t < tile; t++)
                cur = (bstate[t] >> cur) & 1;
            bin = cur;
        }
        s_tile_bin = bin;
    }
    __syncthreads();

    int tile_bin_v = s_tile_bin;
    if (tile_bin_v < 0)
        return; // a < b, sem subtração (out já contém a no caso in-place)

    int bw = bk ? bk[cand] : W;
    s_a[tid] = (j < W) ? a[(size_t)cand * sa + j] : 0ULL;
    s_b[tid] = (j < W && j < bw) ? b[(size_t)cand * sb + j] : 0ULL;
    __syncthreads();

    int bw0 = (j < W && s_a[tid] < s_b[tid]) ? 1 : 0;
    int bw1 = (j < W && s_a[tid] <= s_b[tid]) ? 1 : 0;
    s_state[tid] = bw0 | (bw1 << 1);
    __syncthreads();
    for (int stride = 1; stride < CS_TILE; stride <<= 1)
    {
        int combined = (tid >= stride) ? bar_cs_combine(s_state[tid - stride], s_state[tid])
                                       : s_state[tid];
        __syncthreads();
        s_state[tid] = combined;
        __syncthreads();
    }
    int prefix_excl = (tid == 0) ? 2 : s_state[tid - 1];
    int bin = (prefix_excl >> tile_bin_v) & 1;

    if (j < W)
    {
        int64_t d = (int64_t)s_a[tid] - (int64_t)s_b[tid] - bin;
        out[(size_t)cand * so + j] = (Data64)((d < 0) ? d + (1LL << LIMB_BITS) : d);
    }
}

// Copia os out_limbs baixos de r (W limbs) → out (zero-extende se preciso).
__global__ static void bar_copy_out(
    Data64 *__restrict__ out, const Data64 *__restrict__ r,
    int out_limbs, int W, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= out_limbs)
        return;
    out[(size_t)cand * out_limbs + j] = (j < W) ? r[(size_t)cand * W + j] : 0ULL;
}

// ── helpers GMP específicos ───────────────────────────────────────────────────

// μ = floor(b^{2k}/N), base b = 2^LIMB_BITS. Escreve k+1 limbs (little-endian).
static void compute_barrett_mu(uint64_t *mu_out, const uint64_t *N_lims, int k)
{
    mpz_t N, B2k, mu;
    mpz_init(N);
    mpz_init(B2k);
    mpz_init(mu);
    limbs_to_mpz(N, N_lims, k);
    if (mpz_sgn(N) == 0)
        throw std::runtime_error("Barrett: N == 0");
    mpz_ui_pow_ui(B2k, 2, (unsigned long)(LIMB_BITS * 2 * k)); // b^{2k}
    mpz_tdiv_q(mu, B2k, N);                                    // floor(b^{2k}/N)
    mpz_to_limbs(mu_out, k + 1, mu);
    mpz_clear(N);
    mpz_clear(B2k);
    mpz_clear(mu);
}

// Forma de trabalho Barrett = resíduo plano: res = x mod N (ida e volta idênticas).
void mod_residue_forward(mpz_t res, const mpz_t x, const mpz_t N, int)
{
    mpz_mod(res, x, N);
}
void mod_residue_backward(mpz_t res, const mpz_t x, const mpz_t N, int)
{
    mpz_mod(res, x, N);
}

// ── setup/teardown do backend ─────────────────────────────────────────────────

void BatchModCtx::precompute_reduction(const std::vector<uint64_t> &N_all)
{
    const size_t pb = (size_t)n_batch * padded * sizeof(Data64);
    const size_t sb = (size_t)n_batch * n_sum * sizeof(Data64);

    // bar_k[i] = limbs tight de N_i (índice do MSB não-nulo + 1), POR candidato.
    // bar_W1 = max(bar_k) + 1 = largura uniforme dos buffers A1/μ/q̂.
    std::vector<int> bar_k_all(n_batch, 0);
    int kmax = 0;
    for (int i = 0; i < n_batch; i++)
    {
        const uint64_t *Ni = N_all.data() + (size_t)i * n_limbs;
        int tight = 0;
        for (int j = n_limbs - 1; j >= 0; j--)
            if (Ni[j] != 0)
            {
                tight = j + 1;
                break;
            }
        if (tight < 2)
            throw std::runtime_error("Barrett: N muito pequeno (tight < 2 limbs).");
        bar_k_all[i] = tight;
        if (tight > kmax)
            kmax = tight;
    }
    bar_W1 = kmax + 1;
    CU(cudaMalloc(&d_bar_k, (size_t)n_batch * sizeof(int)));
    CU(cudaMemcpy(d_bar_k, bar_k_all.data(), (size_t)n_batch * sizeof(int), cudaMemcpyHostToDevice));

    // μ_i = floor(b^{2·bar_k_i}/N_i) (k_i+1 limbs, zero-pad a bar_W1) → NTT(μ).
    const size_t w1b = (size_t)n_batch * bar_W1 * sizeof(Data64);
    CU(cudaMalloc(&d_ntt_mu, pb));
    CU(cudaMalloc(&d_bar_w1, w1b));
    CU(cudaMalloc(&d_bar_prod, sb));

    std::vector<uint64_t> mu_all((size_t)n_batch * bar_W1, 0);
    for (int i = 0; i < n_batch; i++)
        compute_barrett_mu(mu_all.data() + (size_t)i * bar_W1,
                           N_all.data() + (size_t)i * n_limbs, bar_k_all[i]);
    Data64 *d_mu_tmp = nullptr;
    CU(cudaMalloc(&d_mu_tmp, w1b));
    CU(cudaMemcpy(d_mu_tmp, mu_all.data(), w1b, cudaMemcpyHostToDevice));
    ntt.ntt_A(d_mu_tmp, bar_W1);
    CU(cudaMemcpy(d_ntt_mu, ntt.d_buf_A, pb, cudaMemcpyDeviceToDevice));
    CU(cudaFree(d_mu_tmp));

    // Buffers do finalize tileado (subtração T−qn e cond_sub de N): largura W1.
    // O borrow_in é resolvido dentro do apply (fundido), então não há d_cs_tile_bin.
    n_cs_tiles = (bar_W1 + CS_TILE - 1) / CS_TILE;
    const size_t csb = (size_t)n_batch * n_cs_tiles * sizeof(int);
    CU(cudaMalloc(&d_cs_tile_cmp, csb));
    CU(cudaMalloc(&d_cs_tile_bstate, csb));
}

void BatchModCtx::free_reduction()
{
    cudaFree(d_bar_k);
    cudaFree(d_ntt_mu);
    cudaFree(d_bar_w1);
    cudaFree(d_bar_prod);
    cudaFree(d_cs_tile_cmp);
    cudaFree(d_cs_tile_bstate);
}

// cond_sub_batch não é usado no Barrett (finalize faz a subtração), mas a função
// é declarada no header; fornecemos uma definição vazia para satisfazer o linker
// caso alguém a referencie. O reduce_batch do Barrett não a chama.
void BatchModCtx::cond_sub_batch(Data64 *, cudaStream_t) {}

// ── redução ───────────────────────────────────────────────────────────────────

// Redução de Barrett: out = T mod N, com T = A·B em d_T [n_batch*n_sum].
//   q̂ = floor( floor(T/b^{k-1})·μ / b^{k+1} )   (q̂ ∈ {q, q-1, q-2}), k = bar_k[i].
//   out = T − q̂·N, com até 2 subtrações finais de N.
// As stages de perf reaproveitam os campos do REDC para a árvore de relatório.
void BatchModCtx::reduce_batch(Data64 *d_out, cudaStream_t s)
{
    const int thr = MR_THR_REDUCE;
    unsigned nb = (unsigned)n_batch;
    const int W1 = bar_W1; // max(bar_k) + 1
    unsigned bw1 = (unsigned)(W1 + thr - 1) / thr;

    // Passo 1: A1 = floor(T / b^{k_i-1}) → W1 limbs; q2 = A1 · μ (NTT) → d_bar_prod.
    TSTART();
    shift_right_var_batch<<<dim3(bw1, nb), thr, 0, s>>>(
        d_bar_w1, d_T, d_bar_k, -1, W1, n_sum, n_batch);
    TSTOP(perf_cur->bar_shift);
    TSTART();
    ntt.ntt_A(d_bar_w1, W1, s);
    TSTOP(perf_cur->red_ntt_tlow);
    TSTART();
    ntt.pmul_ext(d_ntt_mu, s);
    TSTOP(perf_cur->red_pmul_np);
    TSTART();
    ntt.intt_A(s);
    TSTOP(perf_cur->red_intt_np);
    TSTART();
    ntt.carry_to_limbs(d_bar_prod, n_sum, s);
    TSTOP(perf_cur->red_carry_m);

    // Passo 2: q̂ = floor(q2 / b^{k_i+1}) → W1 limbs; qn = q̂ · N (NTT) → d_bar_prod.
    TSTART();
    shift_right_var_batch<<<dim3(bw1, nb), thr, 0, s>>>(
        d_bar_w1, d_bar_prod, d_bar_k, +1, W1, n_sum, n_batch);
    TSTOP(perf_cur->bar_shift);
    TSTART();
    ntt.ntt_A(d_bar_w1, W1, s);
    TSTOP(perf_cur->red_ntt_m);
    TSTART();
    ntt.pmul_ext(d_ntt_N, s);
    TSTOP(perf_cur->red_pmul_n);
    TSTART();
    ntt.intt_A(s);
    TSTOP(perf_cur->red_intt_n);
    // O finalize só usa os W1 limbs baixos de qn (os altos cancelam em T−qn), e o
    // carry propaga só de baixo p/ cima ⇒ basta normalizar W1 limbs, não n_sum.
    TSTART();
    ntt.carry_to_limbs(d_bar_prod, W1, s);
    TSTOP(perf_cur->red_carry_add);

    // Passo 3: out = (T − qn) mod N — finalize tileado paralelo (ver kernels acima).
    // Sub-passos cronometrados separadamente (campos dedicados em PerfInner):
    //   (a) r = T − qn        → bar_sub     (incondicional, W1 limbs → d_bar_w1)
    //   (b) r −= N, até 2×     → bar_condsub (condicional; r_true < 3N, N largura bar_k)
    //   (c) copia r[0..n_limbs) → bar_copy
    dim3 grid((unsigned)n_cs_tiles, nb);

    // (a) r = T − qn (qn já normalizado; limbs altos cancelam, borrow_out = 0).
    //     apply funde a resolução do borrow_in (uncond=1 ⇒ sempre subtrai).
    TSTART();
    bar_sub_phase1<<<grid, CS_TILE, 0, s>>>(
        d_T, n_sum, d_bar_prod, n_sum, nullptr, W1,
        d_cs_tile_cmp, d_cs_tile_bstate, n_batch);
    bar_sub_apply<<<grid, CS_TILE, 0, s>>>(
        d_bar_w1, W1, d_T, n_sum, d_bar_prod, n_sum, nullptr, W1,
        d_cs_tile_cmp, d_cs_tile_bstate, /*uncond=*/1, n_batch);
    TSTOP(perf_cur->bar_sub);

    // (b) até 2 subtrações condicionais de N (in-place em r = d_bar_w1).
    TSTART();
    for (int it = 0; it < 2; it++)
    {
        bar_sub_phase1<<<grid, CS_TILE, 0, s>>>(
            d_bar_w1, W1, d_N, n_limbs, d_bar_k, W1,
            d_cs_tile_cmp, d_cs_tile_bstate, n_batch);
        bar_sub_apply<<<grid, CS_TILE, 0, s>>>(
            d_bar_w1, W1, d_bar_w1, W1, d_N, n_limbs, d_bar_k, W1,
            d_cs_tile_cmp, d_cs_tile_bstate, /*uncond=*/0, n_batch);
    }
    TSTOP(perf_cur->bar_condsub);

    // (c) out = r[0..n_limbs).
    TSTART();
    {
        const int cthr = MR_THR_COPY;
        dim3 cgrid((unsigned)(n_limbs + cthr - 1) / cthr, nb);
        bar_copy_out<<<cgrid, cthr, 0, s>>>(d_out, d_bar_w1, n_limbs, W1, n_batch);
    }
    TSTOP(perf_cur->bar_copy);
}

#endif // MOD_RED_BARRETT
