// reduce_montgomery.cu — Redução de Montgomery (REDC) batched + subtração condicional.
//
// Forma de trabalho: x·R mod N, R = 2^(LIMB_BITS·n_limbs). Todo o conteúdo é
// compilado apenas quando MOD_REDUCTION_ALG == MOD_RED_MONTGOMERY (params.cmake).

#include "batch_mod_ctx.cuh"
#include "gmp_helpers.cuh"
#include "mod_internal.cuh"
#include <vector>
#include <gmp.h>

#if MOD_REDUCTION_ALG == MOD_RED_MONTGOMERY

// ── kernels do REDC ───────────────────────────────────────────────────────────

// Extrai os n_low limbs de d_T [n_batch * n_sum] para d_dst [n_batch * padded],
// zerando o padding. Cada bloco Y trata um candidato.
__global__ static void extract_low_batch(Data64 *__restrict__ dst,
                                         const Data64 *__restrict__ src,
                                         int n_low, int padded, int n_sum, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= padded)
        return;
    dst[cand * padded + j] = (j < n_low) ? src[cand * n_sum + j] : 0ULL;
}

// Desloca d_src [n_batch * n_src] offset posições para a direita → d_dst [n_batch * n_out].
__global__ static void shift_right_batch(Data64 *__restrict__ dst,
                                         const Data64 *__restrict__ src,
                                         int offset, int n_out, int n_src, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= n_out)
        return;
    int s = j + offset;
    dst[cand * n_out + j] = (s < n_src) ? src[cand * n_src + s] : 0ULL;
}

// ── cond_sub tileado ──────────────────────────────────────────────────────────
//
// 3 kernels com grid (n_tiles × n_batch) para alta ocupância.
// Cada bloco processa CS_TILE elementos de um candidato:
//  - Leitura coalescida (todos os threads do warp leem elementos consecutivos)
//  - 1 thread por elemento → prefix scan paralelo em O(log CS_TILE)
//
// Kernel 1 (cs_phase1): compara tile vs N, computa estado G/P/K de borrow do tile.
// Kernel 2 (cs_resolve): 1 thread por candidato, resolve cmp global + borrow_in por tile.
// Kernel 3 (cs_apply):   aplica subtração tile a tile com borrow_in correto.

static constexpr int CS_TILE = MR_CS_TILE;

// Composição de estados G/P/K para borrow:
//   state = bw0 | (bw1 << 1)   (bw0 = borrow_out dado borrow_in=0, bw1 dado borrow_in=1)
// combined(L, R)(b) = R(L(b))
__device__ static int cs_combine(int L, int R)
{
    int c0 = (R >> (L & 1)) & 1;
    int c1 = (R >> ((L >> 1) & 1)) & 1;
    return c0 | (c1 << 1);
}

// Kernel 1: por tile, leitura coalescida, comparação + estado G/P/K de borrow.
__global__ static void cs_phase1(
    const Data64 *__restrict__ a_all,
    const Data64 *__restrict__ b_all,
    int *__restrict__ d_tile_cmp,    // [n_batch * n_tiles]
    int *__restrict__ d_tile_bstate, // [n_batch * n_tiles]
    int n, int n_batch)
{
    __shared__ Data64 s_a[CS_TILE];
    __shared__ Data64 s_b[CS_TILE];
    __shared__ int s_reduce[CS_TILE]; // redução de comparação
    __shared__ int s_state[CS_TILE];  // G/P/K por elemento → prefix scan

    int cand = blockIdx.y, tile = blockIdx.x, tid = threadIdx.x;
    int j = tile * CS_TILE + tid;
    int n_tiles = (n + CS_TILE - 1) / CS_TILE;
    if (cand >= n_batch)
        return;

    // Carga coalescida
    s_a[tid] = (j < n) ? a_all[cand * n + j] : 0ULL;
    s_b[tid] = (j < n) ? b_all[cand * n + j] : 0ULL;
    __syncthreads();

    // Comparação: encode (j+1) << 2 | (cmp+1) para redução de máximo
    int enc = 1;
    if (j < n && s_a[tid] != s_b[tid])
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
        d_tile_cmp[cand * n_tiles + tile] = (s_reduce[0] == 1) ? 0 : ((s_reduce[0] & 3) - 1);

    // Estado G/P/K por elemento (1 thread por elemento → sem loops):
    //   G: a < b  → gera borrow
    //   P: a == b → propaga
    //   K: a > b  → cancela
    int bw0 = (j < n && s_a[tid] < s_b[tid]) ? 1 : 0;
    int bw1 = (j < n && s_a[tid] <= s_b[tid]) ? 1 : 0;
    s_state[tid] = bw0 | (bw1 << 1);
    __syncthreads();

    // Prefix scan inclusivo (Kogge-Stone) para combinar estados [0..tid]
    for (int stride = 1; stride < CS_TILE; stride <<= 1)
    {
        int combined = (tid >= stride) ? cs_combine(s_state[tid - stride], s_state[tid])
                                       : s_state[tid];
        __syncthreads();
        s_state[tid] = combined;
        __syncthreads();
    }
    // s_state[CS_TILE-1] = estado composto do tile inteiro
    int last = min(CS_TILE, n - tile * CS_TILE) - 1;
    if (tid == last)
        d_tile_bstate[cand * n_tiles + tile] = s_state[tid];
}

// Kernel 2: 1 thread por candidato — resolve cmp global e borrow_in por tile.
__global__ static void cs_resolve(
    const int *__restrict__ d_tile_cmp,
    const int *__restrict__ d_tile_bstate,
    int *__restrict__ d_tile_bin, // borrow_in por tile (primeiro elemento)
    int n_tiles, int n_batch)
{
    int cand = blockIdx.x;
    if (cand >= n_batch || threadIdx.x != 0)
        return;

    const int *cmp = d_tile_cmp + cand * n_tiles;
    const int *bstate = d_tile_bstate + cand * n_tiles;
    int *bin = d_tile_bin + cand * n_tiles;

    // Encontra cmp global: tile mais significativo com cmp != 0
    int gcmp = 0;
    for (int t = n_tiles - 1; t >= 0 && gcmp == 0; t--)
        gcmp = cmp[t];

    if (gcmp < 0)
    {
        // a < N: marca como "sem subtração" usando borrow_in inválido
        for (int t = 0; t < n_tiles; t++)
            bin[t] = -1;
        return;
    }

    // a >= N: prefix scan sobre estados de tile para resolver borrow_in por tile
    int cur = 0;
    for (int t = 0; t < n_tiles; t++)
    {
        bin[t] = cur;
        cur = (bstate[t] >> cur) & 1;
    }
}

// Kernel 3: por tile, aplica subtração com borrow_in correto.
__global__ static void cs_apply(
    Data64 *__restrict__ a_all,
    const Data64 *__restrict__ b_all,
    const int *__restrict__ d_tile_bin,
    int n, int n_batch)
{
    __shared__ Data64 s_a[CS_TILE];
    __shared__ Data64 s_b[CS_TILE];
    __shared__ int s_state[CS_TILE]; // G/P/K por elemento
    __shared__ int s_bin[CS_TILE];   // borrow_in por elemento

    int cand = blockIdx.y, tile = blockIdx.x, tid = threadIdx.x;
    int j = tile * CS_TILE + tid;
    int n_tiles = (n + CS_TILE - 1) / CS_TILE;
    if (cand >= n_batch)
        return;

    int tile_bin = d_tile_bin[cand * n_tiles + tile];
    if (tile_bin < 0)
        return; // a < N, sem subtração

    s_a[tid] = (j < n) ? a_all[cand * n + j] : 0ULL;
    s_b[tid] = (j < n) ? b_all[cand * n + j] : 0ULL;
    __syncthreads();

    // Estado G/P/K por elemento
    int bw0 = (j < n && s_a[tid] < s_b[tid]) ? 1 : 0;
    int bw1 = (j < n && s_a[tid] <= s_b[tid]) ? 1 : 0;
    s_state[tid] = bw0 | (bw1 << 1);
    __syncthreads();

    // Prefix scan exclusivo: s_bin[tid] = estado composto de [0..tid-1]
    s_state[tid] = bw0 | (bw1 << 1); // recarrega
    __syncthreads();
    for (int stride = 1; stride < CS_TILE; stride <<= 1)
    {
        int combined = (tid >= stride) ? cs_combine(s_state[tid - stride], s_state[tid])
                                       : s_state[tid];
        __syncthreads();
        s_state[tid] = combined;
        __syncthreads();
    }
    // borrow entrando em tid = (prefixo exclusivo de [0..tid-1])(tile_bin)
    int prefix_excl = (tid == 0) ? 2 : s_state[tid - 1];
    s_bin[tid] = (prefix_excl >> tile_bin) & 1;
    __syncthreads();

    // Aplica subtração
    if (j < n)
    {
        int64_t d = (int64_t)s_a[tid] - (int64_t)s_b[tid] - s_bin[tid];
        a_all[cand * n + j] = (Data64)((d < 0) ? d + (1LL << LIMB_BITS) : d);
    }
}

// ── helpers GMP específicos ───────────────────────────────────────────────────

// N' = R - N^{-1} mod R (fator de correção do REDC). R = 2^(LIMB_BITS·n).
static void compute_Nprime(uint64_t *Np_out, const uint64_t *N_lims, int n)
{
    mpz_t N, R, Np;
    mpz_init(N);
    mpz_init(R);
    mpz_init(Np);
    limbs_to_mpz(N, N_lims, n);
    mpz_ui_pow_ui(R, 2, (unsigned long)(LIMB_BITS * n));
    if (!mpz_invert(Np, N, R))
        throw std::runtime_error("N nao tem inverso mod R");
    mpz_sub(Np, R, Np);
    mpz_to_limbs(Np_out, n, Np);
    mpz_clear(N);
    mpz_clear(R);
    mpz_clear(Np);
}

// Forma de trabalho Montgomery: res = x·R mod N.
void mod_residue_forward(mpz_t res, const mpz_t x, const mpz_t N, int n_limbs)
{
    mpz_t R;
    mpz_init(R);
    mpz_ui_pow_ui(R, 2, (unsigned long)(LIMB_BITS * n_limbs));
    mpz_mul(res, x, R);
    mpz_mod(res, res, N);
    mpz_clear(R);
}

// Saída da forma Montgomery: res = x·R^{-1} mod N.
void mod_residue_backward(mpz_t res, const mpz_t x, const mpz_t N, int n_limbs)
{
    mpz_t R, Rinv;
    mpz_init(R);
    mpz_init(Rinv);
    mpz_ui_pow_ui(R, 2, (unsigned long)(LIMB_BITS * n_limbs));
    mpz_invert(Rinv, R, N);
    mpz_mul(res, x, Rinv);
    mpz_mod(res, res, N);
    mpz_clear(R);
    mpz_clear(Rinv);
}

// ── setup/teardown do backend ─────────────────────────────────────────────────

void BatchModCtx::precompute_reduction(const std::vector<uint64_t> &N_all)
{
    const size_t nb = (size_t)n_batch * n_limbs * sizeof(Data64);
    const size_t pb = (size_t)n_batch * padded * sizeof(Data64);

    // Tiles de cond_sub.
    n_cs_tiles = (n_limbs + CS_TILE - 1) / CS_TILE;
    const size_t csb = (size_t)n_batch * n_cs_tiles * sizeof(int);
    CU(cudaMalloc(&d_cs_tile_cmp, csb));
    CU(cudaMalloc(&d_cs_tile_bstate, csb));
    CU(cudaMalloc(&d_cs_tile_bin, csb));

    // N' por candidato → GPU; NTT(N') pré-computado.
    CU(cudaMalloc(&d_Nprime, nb));
    CU(cudaMalloc(&d_ntt_Nprime, pb));
    std::vector<uint64_t> Np_all((size_t)n_batch * n_limbs, 0);
    for (int i = 0; i < n_batch; i++)
        compute_Nprime(Np_all.data() + (size_t)i * n_limbs, N_all.data() + (size_t)i * n_limbs, n_limbs);
    CU(cudaMemcpy(d_Nprime, Np_all.data(), nb, cudaMemcpyHostToDevice));
    ntt.ntt_A(d_Nprime, n_limbs);
    CU(cudaMemcpy(d_ntt_Nprime, ntt.d_buf_A, pb, cudaMemcpyDeviceToDevice));
}

void BatchModCtx::free_reduction()
{
    cudaFree(d_Nprime);
    cudaFree(d_ntt_Nprime);
    cudaFree(d_cs_tile_cmp);
    cudaFree(d_cs_tile_bstate);
    cudaFree(d_cs_tile_bin);
}

// ── redução ───────────────────────────────────────────────────────────────────

void BatchModCtx::cond_sub_batch(Data64 *d_x, cudaStream_t s)
{
    dim3 grid(n_cs_tiles, n_batch);
    cs_phase1<<<grid, CS_TILE, 0, s>>>(
        d_x, d_N, d_cs_tile_cmp, d_cs_tile_bstate, n_limbs, n_batch);
    cs_resolve<<<n_batch, 1, 0, s>>>(
        d_cs_tile_cmp, d_cs_tile_bstate, d_cs_tile_bin, n_cs_tiles, n_batch);
    cs_apply<<<grid, CS_TILE, 0, s>>>(
        d_x, d_N, d_cs_tile_bin, n_limbs, n_batch);
}

// Redução Montgomery: dado T = A·B (ou A²) em d_T [n_batch * n_sum],
// calcula out = T · R^{-1} mod N para cada candidato.
//
// Algoritmo (REDC):
//   m  = (T mod R) · N' mod R      -- fator de correção
//   t  = (T + m·N) / R             -- elimina os n_limbs limbs baixos
//   if t >= N: out = t - N         -- redução final (cond_sub)
// As multiplicações de T_low·N' e m·N são feitas via NTT.
void BatchModCtx::reduce_batch(Data64 *d_out, cudaStream_t s)
{
    const int thr = MR_THR_REDUCE;
    unsigned bp = (unsigned)(padded + thr - 1) / thr;
    unsigned bl = (unsigned)(n_limbs + thr - 1) / thr;
    unsigned nb = (unsigned)n_batch;

    // Passo 1: m = (T mod R) · N' mod R. T_low = primeiros n_limbs limbs de T.
    TSTART();
    extract_low_batch<<<dim3(bp, nb), thr, 0, s>>>(
        ntt.d_buf_A, d_T, n_limbs, padded, n_sum, n_batch);
    ntt.fwd_A(s);
    TSTOP(perf_cur->red_ntt_tlow);

    TSTART();
    ntt.pmul_ext(d_ntt_Nprime, s);
    TSTOP(perf_cur->red_pmul_np);
    TSTART();
    ntt.intt_A(s);
    TSTOP(perf_cur->red_intt_np);

    TSTART();
    ntt.carry_to_limbs(d_m, n_limbs, s);
    TSTOP(perf_cur->red_carry_m);

    // Passo 2: mN = m · N (NTT(m) × NTT(N) pré-computado).
    TSTART();
    ntt.ntt_A(d_m, n_limbs, s);
    TSTOP(perf_cur->red_ntt_m);

    TSTART();
    ntt.pmul_ext(d_ntt_N, s);
    TSTOP(perf_cur->red_pmul_n);
    TSTART();
    ntt.intt_A(s);
    TSTOP(perf_cur->red_intt_n);

    // Passo 3: T += mN, normaliza carries. (T + mN) é divisível por R.
#if CARRY_NORM_ALG == CARRY_ALG_SEQUENTIAL
    TSTART();
    ntt.add_raw_buf_and_carry(d_T, n_sum, s);
    TSTOP(perf_cur->red_carry_add);
#else
    TSTART();
    ntt.vadd_raw_buf(d_T, n_sum, s);
    TSTOP(perf_cur->red_vadd);
    TSTART();
    ntt.carry_after_vadd(d_T, n_sum, s);
    TSTOP(perf_cur->red_carry_add);
#endif

    // Passo 4: out = (T + mN) / R = shift direito por n_limbs posições.
    TSTART();
    shift_right_batch<<<dim3(bl, nb), thr, 0, s>>>(
        d_out, d_T, n_limbs, n_limbs, n_sum, n_batch);
    TSTOP(perf_cur->red_shift);

    // Passo 5: subtração condicional — garante out < N (out ∈ [0, 2N)).
    TSTART();
    cond_sub_batch(d_out, s);
    TSTOP(perf_cur->cond_sub);
}

#endif // MOD_RED_MONTGOMERY
