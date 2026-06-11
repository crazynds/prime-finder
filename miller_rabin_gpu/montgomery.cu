// montgomery.cu
#include "montgomery.cuh"
#include <stdexcept>
#include <string>

#define CU(expr)                                                                                  \
    do                                                                                            \
    {                                                                                             \
        cudaError_t _e = (expr);                                                                  \
        if (_e != cudaSuccess)                                                                    \
            throw std::runtime_error(std::string("[CUDA] " #expr ": ") + cudaGetErrorString(_e)); \
    } while (0)

// ── kernels ───────────────────────────────────────────────────────────────────

// Um bloco por candidato. Verifica se r_mont == ref_a OU r_mont == ref_b.
// Usa reducao em shared memory para evitar race conditions.
__global__ static void check_passed_kernel(
    const Data64 *__restrict__ r,
    const Data64 *__restrict__ ref_a, // 1_mont por candidato
    const Data64 *__restrict__ ref_b, // (N-1)_mont por candidato
    uint8_t *__restrict__ passed,
    int n_limbs)
{
    int t = blockIdx.x;
    __shared__ int match_a, match_b;
    if (threadIdx.x == 0)
    {
        match_a = 1;
        match_b = 1;
    }
    __syncthreads();

    const Data64 *rv = r + (size_t)t * n_limbs;
    const Data64 *ra = ref_a + (size_t)t * n_limbs;
    const Data64 *rb = ref_b + (size_t)t * n_limbs;

    for (int j = (int)threadIdx.x; j < n_limbs; j += (int)blockDim.x)
    {
        if (rv[j] != ra[j])
            atomicAnd(&match_a, 0);
        if (rv[j] != rb[j])
            atomicAnd(&match_b, 0);
    }
    __syncthreads();

    if (threadIdx.x == 0)
        passed[t] = (uint8_t)(match_a | match_b);
}

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

// Desloca d_src [n_batch * n_src] n_offset posicoes para a direita ->
// d_dst [n_batch * n_out]. Usado para shift por n_limbs no final do reduce.
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
// Cada bloco processa CS_TILE=256 elementos de um candidato:
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
    // Inicializa com identidade (P=2) e usa scan inclusivo deslocado
    // Scan inclusivo completo, depois deslocar
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
    // s_state[tid] = estado de [0..tid] (inclusivo)
    // borrow entrando em tid = cs_combine(estado [0..tid-1], tile_bin)
    // = (prefixo exclusivo)(tile_bin)
    // prefixo exclusivo de [0..tid-1]:
    //   tid=0 → identidade (P=2), aplicado a tile_bin → tile_bin
    //   tid>0 → s_state[tid-1] aplicado a tile_bin
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

// ── helpers CPU (GMP) ─────────────────────────────────────────────────────────

static void limbs_to_mpz(mpz_t out, const uint64_t *lims, int n)
{
    mpz_set_ui(out, 0);
    for (int i = n - 1; i >= 0; i--)
    {
        mpz_mul_2exp(out, out, 16);
        mpz_add_ui(out, out, (unsigned long)lims[i]);
    }
}

static void mpz_to_limbs(uint64_t *out, int n, const mpz_t x)
{
    mpz_t tmp;
    mpz_init_set(tmp, x);
    for (int i = 0; i < n; i++)
    {
        out[i] = mpz_get_ui(tmp) & 0xFFFF;
        mpz_tdiv_q_2exp(tmp, tmp, 16);
    }
    mpz_clear(tmp);
}

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

// ── BatchMontCtx ─────────────────────────────────────────────────────────────

// Helpers para o construtor vector<mpz_t*> — usados na delegação.

static int mpz_compute_n_limbs(const std::vector<mpz_t *> &numbers)
{
    int max_digits = 0;
    for (auto *p : numbers)
    {
        int d = (int)mpz_sizeinbase(*p, 10);
        if (d > max_digits)
            max_digits = d;
    }
    return limbs_for_digits(max_digits + 4);
}

static std::vector<uint64_t> mpz_build_N_all(const std::vector<mpz_t *> &numbers, int nl)
{
    int nb = (int)numbers.size();
    std::vector<uint64_t> N_all((size_t)nb * nl, 0);
    for (int i = 0; i < nb; i++)
        mpz_to_limbs(N_all.data() + i * nl, nl, *numbers[i]);
    return N_all;
}

BatchMontCtx::BatchMontCtx(const std::vector<mpz_t *> &numbers, int device_id_)
    : BatchMontCtx(mpz_build_N_all(numbers, mpz_compute_n_limbs(numbers)),
                   mpz_compute_n_limbs(numbers),
                   (int)numbers.size(),
                   device_id_)
{
}

BatchMontCtx::BatchMontCtx(const std::vector<uint64_t> &N_all, int n_limbs_, int n_batch_,
                           int device_id_)
    : n_limbs(n_limbs_), n_batch(n_batch_), device_id(device_id_), padded(next_pow2_ntt(2 * n_limbs_)), n_sum(2 * n_limbs_ + 16), ntt(n_limbs_, n_batch_)
{
    CU(cudaSetDevice(device_id_));
    const size_t nb = (size_t)n_batch * n_limbs * sizeof(Data64);
    const size_t pb = (size_t)n_batch * padded * sizeof(Data64);
    const size_t sb = (size_t)n_batch * n_sum * sizeof(Data64);

    n_cs_tiles = (n_limbs_ + CS_TILE - 1) / CS_TILE;
    const size_t csb = (size_t)n_batch * n_cs_tiles * sizeof(int);

    CU(cudaMalloc(&d_N, nb));
    CU(cudaMalloc(&d_Nprime, nb));
    CU(cudaMalloc(&d_ntt_N, pb));
    CU(cudaMalloc(&d_ntt_Nprime, pb));
    CU(cudaMalloc(&d_T, sb));
    CU(cudaMalloc(&d_m, pb));
    CU(cudaMalloc(&d_one_mont, nb));
    CU(cudaMalloc(&d_Nm1_mont, nb));
    CU(cudaMalloc(&d_cs_tile_cmp, csb));
    CU(cudaMalloc(&d_cs_tile_bstate, csb));
    CU(cudaMalloc(&d_cs_tile_bin, csb));

    CU(cudaMemcpy(d_N, N_all.data(), nb, cudaMemcpyHostToDevice));

    // Calcula N' para cada candidato e sobe para GPU
    std::vector<uint64_t> Np_all(n_batch * n_limbs, 0);
    for (int i = 0; i < n_batch; i++)
        compute_Nprime(Np_all.data() + i * n_limbs, N_all.data() + i * n_limbs, n_limbs);
    CU(cudaMemcpy(d_Nprime, Np_all.data(), nb, cudaMemcpyHostToDevice));

    // Pre-computa NTT(N) e NTT(N')
    ntt.ntt_A(d_N, n_limbs);
    CU(cudaMemcpy(d_ntt_N, ntt.d_buf_A, pb, cudaMemcpyDeviceToDevice));
    ntt.ntt_A(d_Nprime, n_limbs);
    CU(cudaMemcpy(d_ntt_Nprime, ntt.d_buf_A, pb, cudaMemcpyDeviceToDevice));
    // Não precisa de cudaDeviceSynchronize aqui: to_mont_batch() começa com um
    // cudaMemcpy(D2H) no default stream, que garante ordering implicitamente.

    // Computa 1_mont e (N-1)_mont para cada candidato (via GMP, feito so uma vez)
    std::vector<uint64_t> one_lims(n_batch * n_limbs, 0);
    for (int i = 0; i < n_batch; i++)
        one_lims[i * n_limbs] = 1;

    std::vector<uint64_t> Nm1_lims(n_batch * n_limbs, 0);
    for (int i = 0; i < n_batch; i++)
    {
        // Nm1 = N - 1: subtrai 1 do primeiro limb
        const uint64_t *Ni = N_all.data() + i * n_limbs;
        uint64_t *out = Nm1_lims.data() + i * n_limbs;
        std::copy(Ni, Ni + n_limbs, out);
        // subtrai 1 com borrow
        for (int j = 0; j < n_limbs; j++)
        {
            if (out[j] > 0)
            {
                out[j]--;
                break;
            }
            out[j] = 0xFFFF; // borrow
        }
    }

    std::vector<uint64_t> one_mont_h, Nm1_mont_h;
    to_mont_batch(one_lims, one_mont_h);
    to_mont_batch(Nm1_lims, Nm1_mont_h);
    CU(cudaMemcpy(d_one_mont, one_mont_h.data(), nb, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_Nm1_mont, Nm1_mont_h.data(), nb, cudaMemcpyHostToDevice));

    for (int i = 0; i <= PERF_RING; i++)
        CU(cudaEventCreate(&ev_ring[i]));
}

BatchMontCtx::~BatchMontCtx()
{
    cudaFree(d_N);
    cudaFree(d_Nprime);
    cudaFree(d_ntt_N);
    cudaFree(d_ntt_Nprime);
    cudaFree(d_T);
    cudaFree(d_m);
    cudaFree(d_one_mont);
    cudaFree(d_Nm1_mont);
    for (int i = 0; i <= PERF_RING; i++)
        if (ev_ring[i])
            cudaEventDestroy(ev_ring[i]);
}

// Sincroniza o último evento gravado e acumula todos os tempos pendentes.
// Chamada UMA VEZ por mont_mul_batch / mont_sq_batch.
void BatchMontCtx::perf_flush(cudaStream_t s)
{
    if (!perf_enabled || ring_cur == 0)
        return;
    CU(cudaEventSynchronize(ev_ring[ring_cur]));
    for (int i = 0; i < ring_cur; i++)
    {
        float ms = 0;
        CU(cudaEventElapsedTime(&ms, ev_ring[i], ev_ring[i + 1]));
        *acc_ring[i] += ms;
    }
    ring_cur = 0;
}

void BatchMontCtx::to_mont_batch(const std::vector<uint64_t> &x_all,
                                 std::vector<uint64_t> &out_all) const
{
    out_all.resize((size_t)n_batch * n_limbs, 0);
    std::vector<uint64_t> N_h(n_batch * n_limbs);
    CU(cudaMemcpy(N_h.data(), d_N, N_h.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    mpz_t xm, N, R, res;
    mpz_init(xm);
    mpz_init(N);
    mpz_init(R);
    mpz_init(res);
    for (int i = 0; i < n_batch; i++)
    {
        limbs_to_mpz(xm, x_all.data() + i * n_limbs, n_limbs);
        limbs_to_mpz(N, N_h.data() + i * n_limbs, n_limbs);
        mpz_ui_pow_ui(R, 2, (unsigned long)(LIMB_BITS * n_limbs));
        mpz_mul(res, xm, R);
        mpz_mod(res, res, N);
        mpz_to_limbs(out_all.data() + i * n_limbs, n_limbs, res);
    }
    mpz_clear(xm);
    mpz_clear(N);
    mpz_clear(R);
    mpz_clear(res);
}

void BatchMontCtx::from_mont_batch(const Data64 *d_x, std::vector<uint64_t> &out_all) const
{
    out_all.resize((size_t)n_batch * n_limbs, 0);

    std::vector<uint64_t> x_h((size_t)n_batch * n_limbs);
    std::vector<uint64_t> N_h((size_t)n_batch * n_limbs);
    CU(cudaMemcpy(x_h.data(), d_x, x_h.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(N_h.data(), d_N, N_h.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    mpz_t xm, N, R, Rinv, res;
    mpz_init(xm);
    mpz_init(N);
    mpz_init(R);
    mpz_init(Rinv);
    mpz_init(res);
    for (int i = 0; i < n_batch; i++)
    {
        limbs_to_mpz(xm, x_h.data() + i * n_limbs, n_limbs);
        limbs_to_mpz(N, N_h.data() + i * n_limbs, n_limbs);
        mpz_ui_pow_ui(R, 2, (unsigned long)(LIMB_BITS * n_limbs));
        mpz_invert(Rinv, R, N);
        mpz_mul(res, xm, Rinv);
        mpz_mod(res, res, N);
        mpz_to_limbs(out_all.data() + i * n_limbs, n_limbs, res);
    }
    mpz_clear(xm);
    mpz_clear(N);
    mpz_clear(R);
    mpz_clear(Rinv);
    mpz_clear(res);
}

void BatchMontCtx::check_passed(const Data64 *d_r_mont, uint8_t *d_passed,
                                cudaStream_t s) const
{
    // Um bloco por candidato, MR_THR_CHECK threads para varrer n_limbs em paralelo
    check_passed_kernel<<<n_batch, MR_THR_CHECK, 0, s>>>(
        d_r_mont, d_one_mont, d_Nm1_mont, d_passed, n_limbs);
}

// Grava o marco de início da seção no ring (sem sync).
// No-op quando perf_enabled == false.
#define TSTART()                                       \
    do                                                 \
    {                                                  \
        if (perf_enabled)                              \
            CU(cudaEventRecord(ev_ring[ring_cur], s)); \
    } while (0)

// Grava o marco de fim, registra o acumulador — sem sync.
// perf_flush() no final da função pública sincroniza uma única vez.
// No-op quando perf_enabled == false.
#define TSTOP(acc)                                         \
    do                                                     \
    {                                                      \
        if (perf_enabled)                                  \
        {                                                  \
            CU(cudaEventRecord(ev_ring[ring_cur + 1], s)); \
            acc_ring[ring_cur] = &(acc);                   \
            ring_cur++;                                    \
        }                                                  \
    } while (0)

// Redução Montgomery: dado T = A*B (ou A²) em d_T [n_batch * n_sum],
// calcula out = T * R^{-1} mod N para cada candidato.
//
// Algoritmo (REDC):
//   m  = (T mod R) * N' mod R      -- fator de correção
//   t  = (T + m*N) / R             -- elimina os n_limbs limbs baixos
//   if t >= N: out = t - N         -- redução final (cond_sub, feita pelo chamador)
//   else:      out = t
//
// Como T e m*N são polinômios grandes, a multiplicação é via NTT.
// R = 2^(LIMB_BITS * n_limbs).
void BatchMontCtx::reduce_batch(Data64 *d_out, cudaStream_t s)
{
    const int thr = MR_THR_REDUCE;
    unsigned bp = (unsigned)(padded + thr - 1) / thr;
    unsigned bl = (unsigned)(n_limbs + thr - 1) / thr;
    unsigned nb = (unsigned)n_batch;

    // Passo 1: m = (T mod R) * N' mod R
    //   T mod R = T_low = primeiros n_limbs limbs de T.
    //   extract_low copia T_low para buf_A (zerado até padded).
    //   fwd_A aplica NTT forward em buf_A, preparando para pmul.
    TSTART();
    extract_low_batch<<<dim3(bp, nb), thr, 0, s>>>(
        ntt.d_buf_A, d_T, n_limbs, padded, n_sum, n_batch);
    ntt.fwd_A(s);
    TSTOP(perf.red_ntt_tlow_ms);

    // Passo 2: buf_A = INTT(NTT(T_low) * NTT(N')) = T_low * N' (convolução polinomial)
    //   Resultado raw (sem carry) fica em buf_A.
    TSTART();
    ntt.pmul_ext(d_ntt_Nprime, s);
    TSTOP(perf.red_pmul_np_ms);
    TSTART();
    ntt.intt_A(s);
    TSTOP(perf.red_intt_np_ms);

    // Passo 3: normaliza m — propaga carries nos coeficientes brutos do INTT.
    //   Resultado normalizado (limbs de 16 bits) vai para d_m [n_batch * n_limbs].
    TSTART();
    ntt.carry_to_limbs(d_m, n_limbs, s);
    TSTOP(perf.red_carry_m_ms);

    // Passo 4: mN = m * N
    //   Transforma m para o domínio NTT para multiplicar por NTT(N) pré-computado.
    TSTART();
    ntt.ntt_A(d_m, n_limbs, s);
    TSTOP(perf.red_ntt_m_ms);

    // Passo 5: buf_A = INTT(NTT(m) * NTT(N)) = m * N (convolução polinomial)
    TSTART();
    ntt.pmul_ext(d_ntt_N, s);
    TSTOP(perf.red_pmul_n_ms);
    TSTART();
    ntt.intt_A(s);
    TSTOP(perf.red_intt_n_ms);

    // Passo 6: T += mN, normaliza carries.
    //   (T + mN) é divisível por R por construção; o resultado normalizado
    //   fica em d_T [n_batch * n_sum] pronto para o shift.
#if CARRY_NORM_ALG == CARRY_ALG_SEQUENTIAL
    TSTART();
    ntt.add_raw_buf_and_carry(d_T, n_sum, s);
    TSTOP(perf.red_add_carry_ms);
#else
    TSTART();
    ntt.vadd_raw_buf(d_T, n_sum, s);
    TSTOP(perf.red_vadd_ms);
    TSTART();
    ntt.carry_after_vadd(d_T, n_sum, s);
    TSTOP(perf.red_add_carry_ms);
#endif

    // Passo 7: out = (T + mN) / R = shift direito por n_limbs posições.
    //   shift_right copia d_T[n_limbs .. n_limbs+n_limbs-1] para d_out.
    TSTART();
    shift_right_batch<<<dim3(bl, nb), thr, 0, s>>>(
        d_out, d_T, n_limbs, n_limbs, n_sum, n_batch);
    TSTOP(perf.red_shift_ms);
}

void BatchMontCtx::cond_sub_batch(Data64 *d_x, cudaStream_t s)
{
    dim3 grid(n_cs_tiles, n_batch);
    cs_phase1<<<grid, CS_TILE, 0, s>>>(
        d_x, d_N, d_cs_tile_cmp, d_cs_tile_bstate, n_limbs, n_batch);
    cs_resolve<<<n_batch, 1, 0, s>>>(
        d_cs_tile_cmp, d_cs_tile_bstate, d_cs_tile_bin, n_cs_tiles, n_batch);
    cs_apply<<<grid, CS_TILE, 0, s>>>(
        d_x, d_N, d_cs_tile_bin, n_limbs, n_batch);
}

// Multiplicação Montgomery: out = A * B * R^{-1} mod N para cada candidato.
//
// R = 2^(LIMB_BITS * n_limbs). Entrada e saída estão em forma Montgomery
// (A_mont = A*R mod N), então mont_mul(A_mont, B_mont) = A*B*R mod N.
void BatchMontCtx::mont_mul_batch(const Data64 *d_A, const Data64 *d_B, Data64 *d_out,
                                  cudaStream_t s)
{
    // Passo 1 — produto polinomial A * B → d_buf_A (raw, sem carry).
    //   NTT: transforma A e B, multiplica pontualmente, INTT de volta.
    //   SCHOOLBOOK: convolução direta O(n²), escreve coefs brutos em d_buf_A.
    TSTART();
#if MONT_MUL_ALG == MONT_MUL_ALG_NTT
    ntt.ntt_AB(d_A, d_B, n_limbs, s);
#elif MONT_MUL_ALG == MONT_MUL_ALG_SCHOOLBOOK
    ntt.schoolbook_mul(d_A, d_B, n_limbs, s);
#endif
    TSTOP(perf.ntt_input_ms);

    // Passo 2 — (apenas NTT) multiplicação pontual + INTT.
    //   Para SCHOOLBOOK, d_buf_A já tem os coefs brutos do passo anterior.
#if MONT_MUL_ALG == MONT_MUL_ALG_NTT
    TSTART();
    ntt.pmul(s);
    TSTOP(perf.pmul_ms);
    TSTART();
    ntt.intt_A(s);
    TSTOP(perf.intt_product_ms);
#endif

    // Passo 3: normaliza T — propaga carries dos coeficientes brutos do INTT.
    //   Resultado vai para d_T [n_batch * n_sum] com limbs de 16 bits.
    TSTART();
    ntt.carry_to_limbs(d_T, n_sum, s);
    TSTOP(perf.carry_product_ms);

    // Passo 4: redução Montgomery — out = T * R^{-1} mod N.
    reduce_batch(d_out, s);

    // Passo 5: subtração condicional — garante out < N.
    //   Se out >= N após a redução, subtrai N (raro mas possível).
    TSTART();
    cond_sub_batch(d_out, s);
    TSTOP(perf.cond_sub_ms);
    perf_flush(s); // único sync por chamada
}

// Quadrado Montgomery: out = A² * R^{-1} mod N para cada candidato.
//
// Equivalente a mont_mul_batch(A, A, out) mas usa psq (pointwise square)
// no domínio NTT em vez de pmul — economiza NTT(B) e metade das multiplicações.
void BatchMontCtx::mont_sq_batch(const Data64 *d_A, Data64 *d_out, cudaStream_t s)
{
    // Passo 1 — quadrado polinomial A² → d_buf_A (raw, sem carry).
    TSTART();
#if MONT_MUL_ALG == MONT_MUL_ALG_NTT
    ntt.ntt_A(d_A, n_limbs, s);
#elif MONT_MUL_ALG == MONT_MUL_ALG_SCHOOLBOOK
    ntt.schoolbook_sq(d_A, n_limbs, s);
#endif
    TSTOP(perf.ntt_input_ms);

    // Passo 2 — (apenas NTT) PSQ pontual + INTT.
#if MONT_MUL_ALG == MONT_MUL_ALG_NTT
    TSTART();
    ntt.psq(s);
    TSTOP(perf.pmul_ms);
    TSTART();
    ntt.intt_A(s);
    TSTOP(perf.intt_product_ms);
#endif

    // Passo 3: normaliza T — propaga carries dos coeficientes brutos do INTT.
    //   Resultado vai para d_T [n_batch * n_sum] com limbs de 16 bits.
    TSTART();
    ntt.carry_to_limbs(d_T, n_sum, s);
    TSTOP(perf.carry_product_ms);

    // Passo 4: redução Montgomery — out = T * R^{-1} mod N.
    reduce_batch(d_out, s);

    // Passo 5: subtração condicional — garante out < N.
    //   Se out >= N após a redução, subtrai N (raro mas possível).
    TSTART();
    cond_sub_batch(d_out, s);
    TSTOP(perf.cond_sub_ms);
    perf_flush(s); // único sync por chamada
}

// Apenas NTT(A)*NTT(B) + INTT + carry — sem REDC nem cond_sub.
// Usado para benchmark de mul pura, sem redução modular.
void BatchMontCtx::mul_no_redc_batch(const Data64 *d_A, const Data64 *d_B,
                                     Data64 *d_out, cudaStream_t s)
{
    ntt.ntt_AB(d_A, d_B, n_limbs, s);
    ntt.pmul_and_intt(s);
    ntt.carry_to_limbs(d_out, n_sum, s);
    cudaStreamSynchronize(s);
}

// Apenas NTT(A)^2 + INTT + carry — sem REDC nem cond_sub.
void BatchMontCtx::sq_no_redc_batch(const Data64 *d_A, Data64 *d_out, cudaStream_t s)
{
    ntt.ntt_A(d_A, n_limbs, s);
    ntt.psq_and_intt(s);
    ntt.carry_to_limbs(d_out, n_sum, s);
    cudaStreamSynchronize(s);
}
