// carry_norm.cu — Normalização de carry (propagação) dos coeficientes brutos do INTT
// e somas de big-integer associadas. Separado de bigint_ntt.cu por reunir os 4
// algoritmos de carry selecionáveis (CARRY_NORM_ALG) e os métodos de BigIntNTTBatch
// que os lançam. Os kernels de transformada NTT propriamente ditos ficam em bigint_ntt.cu.

#include "config.h"
#include "ops/mul/ntt_merge.cuh"
#include <cstdio>

// ── kernels de soma (suporte ao carry) ────────────────────────────────────────

__global__ static void vadd_batch(
    Data64 *__restrict__ d_c,
    const Data64 *__restrict__ d_a,
    const Data64 *__restrict__ d_b,
    int n, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= n)
        return;
    d_c[cand * n + j] = d_a[cand * n + j] + d_b[cand * n + j];
}

#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE || CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE || CARRY_NORM_ALG == CARRY_ALG_PREFIX_SCAN
// Soma d_buf_A (stride=padded, raw INTT) em d_dst (stride=n_dst) element-wise.
__global__ static void vadd_from_raw_batch(
    Data64 *__restrict__ d_dst,
    const Data64 *__restrict__ d_raw,
    int n_dst, int padded, int n_batch)
{
    int cand = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch || j >= n_dst)
        return;
    if (j < padded)
        d_dst[cand * n_dst + j] += d_raw[cand * padded + j];
}
#endif

// ── Algoritmos de carry normalização ─────────────────────────────────────────
//
// Selecione via CARRY_NORM_ALG em config.h:
//   CARRY_ALG_SINGLE_TILE — 1 bloco/candidato, CARRY_TILE threads, shared-mem carry
//   CARRY_ALG_MULTI_TILE  — intra-tile paralelo + inter-tile sequencial (2 kernels)
//   CARRY_ALG_SEQUENTIAL  — 1 thread/candidato, loop sequencial puro
//   CARRY_ALG_PREFIX_SCAN — 1 bloco/candidato, PSCAN_TILE threads, carry-lookahead

static constexpr int CARRY_TILE = MR_CARRY_TILE;

// ── CARRY_ALG_SINGLE_TILE ────────────────────────────────────────────────────
// CARRY_TILE deve ser exatamente 32 (um warp) para usar __ballot_sync /
// __shfl_up_sync e eliminar has_carry de shared memory.
#if CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE

static_assert(CARRY_TILE == 32, "CARRY_ALG_SINGLE_TILE requer CARRY_TILE == 32 (um warp)");

#ifdef MR_ADVANCED_MONITOR
__device__ unsigned long long g_for_count = 0;
__device__ unsigned long long g_dowhile_count = 0;
#endif

__global__ static void carry_16bits(
    Data64 *d_src,
    Data64 *d_dst,
    int n, int src_stride, int n_batch)
{
    int tid = threadIdx.x;
    int cand = blockIdx.x;
    if (cand >= n_batch)
        return;
    int src_offset = cand * src_stride;
    int dst_offset = cand * n;

    Data64 tile_carry = 0;
#ifdef MR_ADVANCED_MONITOR
    unsigned long long local_for = 0, local_dowhile = 0;
#endif

    for (int tile = tid; tile < n; tile += CARRY_TILE)
    {
#ifdef MR_ADVANCED_MONITOR
        if (tid == 0)
            local_for++;
#endif
        Data64 currVal = d_src[src_offset + tile];
        Data64 c = (tid == 0) ? tile_carry : 0ULL;
        Data64 escape = 0;

        unsigned ballot;
        do
        {
#ifdef MR_ADVANCED_MONITOR
            if (tid == 0)
                local_dowhile++;
#endif
            c += currVal;
            currVal = c & LIMB_MASK;
            c >>= LIMB_BITS;

            escape += __shfl_sync(0xFFFFFFFFu, c, CARRY_TILE - 1);
            c = (tid == CARRY_TILE - 1) ? 0 : c;

            Data64 from_left = __shfl_up_sync(0xFFFFFFFFu, c, 1);
            c = (tid > 0) ? from_left : 0ULL;

            ballot = __ballot_sync(0xFFFFFFFFu, c > 0);
        } while (ballot);

        tile_carry = escape;

        d_dst[dst_offset + tile] = currVal;
    }

#ifdef MR_ADVANCED_MONITOR
    if (tid == 0)
    {
        atomicAdd(&g_for_count, local_for);
        atomicAdd(&g_dowhile_count, local_dowhile);
    }
#endif
}

// ── CARRY_ALG_MULTI_TILE ─────────────────────────────────────────────────────
#elif CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE

// Fase 1 — copia src→dst e normaliza carries intra-tile em paralelo.
// Cada bloco (tile, cand) processa CARRY_TILE elementos independentemente.
// O carry que escapa do tile é salvo em d_tile_carry[cand*n_tiles + tile].
__global__ static void carry_intra_copy(
    Data64 *__restrict__ d_dst,
    const Data64 *__restrict__ d_src,
    Data64 *__restrict__ d_tile_carry,
    int n_dst, int n_src, int n_batch)
{
    int cand = blockIdx.y, tile = blockIdx.x, tid = threadIdx.x;
    if (cand >= n_batch)
        return;

    int n_tiles = (n_dst + CARRY_TILE - 1) / CARRY_TILE;
    int j_start = tile * CARRY_TILE;

    int j = j_start + tid;
    __shared__ Data64 carry[CARRY_TILE + 1];
    __shared__ int has_carry[2];
    if (tid == 0)
    {
        carry[CARRY_TILE] = 0;
        has_carry[0] = false;
        has_carry[1] = false;
    }
    carry[tid] = ((j < n_src) ? d_src[cand * n_src + j] : 0ULL);
    __syncthreads();
    Data64 currVal = 0;
    int currIter = 0;
    do
    {
        Data64 v = carry[tid];
        currIter = currIter ^ 1;
        carry[tid] = 0;
        if (tid == 0)
        {
            has_carry[currIter] = false;
        }
        __syncthreads();
        v += currVal;
        currVal = v & LIMB_MASK;
        v >>= LIMB_BITS;
        if (v > 0)
            has_carry[currIter] = true;
        carry[tid + 1] += v;
        __syncthreads();
    } while (has_carry[currIter]);

    if (j < n_dst)
        d_dst[cand * n_dst + j] = currVal;
    if (tid == 0)
        d_tile_carry[cand * n_tiles + tile] = carry[CARRY_TILE];
}

// Fase 2 — propaga carries entre tiles sequencialmente (1 thread por candidato).
__global__ static void carry_inter_tiles(
    Data64 *__restrict__ d_dst,
    Data64 *__restrict__ d_tile_carry,
    int n, int n_batch)
{
    int cand = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch)
        return;

    int n_tiles = (n + CARRY_TILE - 1) / CARRY_TILE;
    for (int t = 0; t < n_tiles - 1; t++)
    {
        Data64 c = d_tile_carry[cand * n_tiles + t];
        if (c == 0)
            continue;
        int j_start = (t + 1) * CARRY_TILE;
        int j_end = min(j_start + CARRY_TILE, n);
        for (int j = j_start; c > 0 && j < j_end; j++)
        {
            Data64 v = d_dst[cand * n + j] + c;
            d_dst[cand * n + j] = v & LIMB_MASK;
            c = v >> LIMB_BITS;
        }
        // Carry que escapa do tile seguinte — raro mas possível
        if (c > 0 && j_end < n)
            d_tile_carry[cand * n_tiles + t + 1] += c;
    }
}

// ── CARRY_ALG_SEQUENTIAL ─────────────────────────────────────────────────────
#elif CARRY_NORM_ALG == CARRY_ALG_SEQUENTIAL

// 1 thread por candidato — copia d_src (stride=n_src) → d_dst (stride=n_dst)
// normalizando carries sequencialmente.
__global__ static void carry_sequential(
    Data64 *__restrict__ d_dst,
    const Data64 *__restrict__ d_src,
    int n_dst, int n_src, int n_batch)
{
    int cand = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch)
        return;

    Data64 carry = 0;
    for (int j = 0; j < n_dst; j++)
    {
        Data64 v = (j < n_src ? d_src[cand * n_src + j] : 0ULL) + carry;
        d_dst[cand * n_dst + j] = v & LIMB_MASK;
        carry = v >> LIMB_BITS;
    }
}

// Versão fundida para add_raw_buf_and_carry: soma d_raw (raw INTT, stride=padded)
// em d_dst e normaliza carries em uma única passagem sequencial por candidato.
__global__ static void vadd_carry_sequential(
    Data64 *__restrict__ d_dst,
    const Data64 *__restrict__ d_raw,
    int n_dst, int padded, int n_batch)
{
    int cand = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= n_batch)
        return;

    Data64 carry = 0;
    for (int j = 0; j < n_dst; j++)
    {
        Data64 raw = (j < padded ? d_raw[cand * padded + j] : 0ULL);
        Data64 v = d_dst[cand * n_dst + j] + raw + carry;
        d_dst[cand * n_dst + j] = v & LIMB_MASK;
        carry = v >> LIMB_BITS;
    }
}

// ── CARRY_ALG_PREFIX_SCAN ────────────────────────────────────────────────────
// Carry-lookahead adder (Kogge-Stone). 1 bloco por candidato, PSCAN_TILE threads.
//
// Ideia: um limb bruto de 64 bits é a soma de 4 "planos" de dígitos de 16 bits,
// onde o plano k contribui no dígito (i+k):
//     v[i] = c0(i) + c1(i)·2^16 + c2(i)·2^32 + c3(i)·2^48
//   ⇒ valor = Σ c_k(i)·2^(16(i+k))   ⇒  plano k vale c_k(i) no dígito i+k.
// Normalizar = somar os 4 planos deslocados. Fazemos pela árvore
//     (p0 + p1) + (p2 + p3)
// e cada soma é de dois números JÁ normalizados em base 2^16, logo o carry de
// cada dígito é no máximo 1 (binário). Isso habilita a álgebra generate/propagate:
//     gen[j]  = (A[j]+B[j]) >= 2^16            (a posição gera carry)
//     prop[j] = (A[j]+B[j]) == 2^16 - 1        (a posição propaga um carry que chegar)
// O carry que entra em cada posição sai de um prefix-scan (Kogge-Stone) do operador
// associativo  (G,P) = (G_hi | (P_hi & G_lo),  P_hi & P_lo),  em log(T) passos —
// sem passe sequencial de propagação.
#elif CARRY_NORM_ALG == CARRY_ALG_PREFIX_SCAN

static constexpr int PSCAN_TILE = MR_CARRY_TILE;
static_assert(PSCAN_TILE >= 32 && PSCAN_TILE <= 1024 && (PSCAN_TILE % 32) == 0,
              "MR_PSCAN_TILE deve ser múltiplo de 32 entre 32 e 1024");

static constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

#if PSCAN_TILE == 32
// ───── Caminho 1 warp (PSCAN_TILE == 32): só shuffle, sem shared, sem barreira ──

// Soma carry-lookahead de dois dígitos já normalizados, dentro de um único warp.
// `s` = A[lane] + B[lane] (≤ 2·LIMB_MASK, carry binário). `cin` é o carry escalar
// que entra no tile (idêntico em todas as lanes). Retorna o dígito normalizado da
// lane; escreve em *tile_cout o carry que sai do tile (idêntico em todas as lanes).
// Tudo via __shfl — sem __syncthreads e sem shared memory.
__device__ static inline Data64 cla_warp(Data64 s, int cin, int *tile_cout)
{
    int lane = threadIdx.x;                                // 0..31
    unsigned g = (unsigned)((s >> LIMB_BITS) & 1ULL);      // generate
    unsigned p = ((s & LIMB_MASK) == LIMB_MASK) ? 1u : 0u; // propagate

    // Kogge-Stone via shuffle: scan inclusivo do operador (G,P), low→high.
#pragma unroll
    for (int d = 1; d < 32; d <<= 1)
    {
        unsigned gl = __shfl_up_sync(FULL_MASK, g, d);
        unsigned pl = __shfl_up_sync(FULL_MASK, p, d);
        if (lane >= d)
        {
            g = g | (p & gl);
            p = p & pl;
        }
    }
    // (g,p) = agregado inclusivo de [0..lane]. Carry que ENTRA na lane = agregado
    // EXCLUSIVo [0..lane-1] (shuffle ↑1) aplicado a cin; prefixo vazio propaga cin.
    unsigned eg = __shfl_up_sync(FULL_MASK, g, 1);
    unsigned ep = __shfl_up_sync(FULL_MASK, p, 1);
    if (lane == 0)
    {
        eg = 0u;
        ep = 1u;
    }
    int carry_in = (int)(eg | (ep & (unsigned)cin));

    // Carry que SAI do tile = agregado inclusivo do warp inteiro (lane 31) com cin.
    unsigned Gt = __shfl_sync(FULL_MASK, g, 31);
    unsigned Pt = __shfl_sync(FULL_MASK, p, 31);
    *tile_cout = (int)(Gt | (Pt & (unsigned)cin));

    return ((s & LIMB_MASK) + (Data64)carry_in) & LIMB_MASK;
}

// Normaliza d_src (limbs brutos de 64 bits, stride=src_stride) → d_dst (stride=n_dst)
// em base 2^16. 1 warp por candidato. Seguro in-place (src == dst, src_stride ==
// n_dst): o halo de 3 limbs do tile anterior fica em registradores (pr1/pr2/pr3),
// e cada lane só escreve sua própria posição depois de já ter lido o limb bruto.
__global__ static void pscan_normalize(
    Data64 *__restrict__ d_dst,
    const Data64 *__restrict__ d_src,
    int n_dst, int src_stride, int n_batch)
{
    int cand = blockIdx.x;
    if (cand >= n_batch)
        return;
    int lane = threadIdx.x;
    const Data64 *src = d_src + (size_t)cand * src_stride;
    Data64 *dst = d_dst + (size_t)cand * n_dst;

    int cin1 = 0, cin2 = 0, cin3 = 0; // carries escalares entre tiles
    // Halo: limbs brutos globais base-1, base-2, base-3 (0 antes do primeiro tile).
    Data64 pr1 = 0, pr2 = 0, pr3 = 0;

#ifdef MR_ADVANCED_MONITOR
    unsigned long long local_for = 0, local_dowhile = 0;
#endif

    for (int base = 0; base < n_dst; base += PSCAN_TILE)
    {
        int j = base + lane;
        Data64 raw0 = (j < n_dst) ? src[j] : 0ULL; // posição j
#ifdef MR_ADVANCED_MONITOR
        if (lane == 0)
        {
            local_for++;
            local_dowhile += 3;
        }
#endif
        // Halo dentro do tile via shuffle; fora do tile (lane < k) vem de pr1/pr2/pr3.
        Data64 sh1 = __shfl_up_sync(FULL_MASK, raw0, 1);
        Data64 sh2 = __shfl_up_sync(FULL_MASK, raw0, 2);
        Data64 sh3 = __shfl_up_sync(FULL_MASK, raw0, 3);
        Data64 raw1 = (lane >= 1) ? sh1 : pr1;                                         // posição j-1
        Data64 raw2 = (lane >= 2) ? sh2 : (lane == 1 ? pr1 : pr2);                     // posição j-2
        Data64 raw3 = (lane >= 3) ? sh3 : (lane == 2 ? pr1 : (lane == 1 ? pr2 : pr3)); // j-3

        Data64 c0 = raw0 & LIMB_MASK;         // plano 0, dígito j
        Data64 c1 = (raw1 >> 16) & LIMB_MASK; // plano 1, deslocado +1
        Data64 c2 = (raw2 >> 32) & LIMB_MASK; // plano 2, deslocado +2
        Data64 c3 = (raw3 >> 48) & LIMB_MASK; // plano 3, deslocado +3

        int cout;
        Data64 r1 = cla_warp(c0 + c1, cin1, &cout);
        cin1 = cout;
        Data64 r2 = cla_warp(c2 + c3, cin2, &cout);
        cin2 = cout;
        Data64 digit = cla_warp(r1 + r2, cin3, &cout);
        cin3 = cout;

        if (j < n_dst)
            dst[j] = digit;

        // Halo do próximo tile = limbs brutos das lanes 31,30,29 deste tile.
        pr1 = __shfl_sync(FULL_MASK, raw0, 31);
        pr2 = __shfl_sync(FULL_MASK, raw0, 30);
        pr3 = __shfl_sync(FULL_MASK, raw0, 29);
    }

#ifdef MR_ADVANCED_MONITOR
    if (lane == 0)
    {
        atomicAdd(&g_for_count, local_for);
        atomicAdd(&g_dowhile_count, local_dowhile);
    }
#endif
}

#else
// ───── Caminho block-wide (PSCAN_TILE > 32): scan hierárquico warp→bloco ────────
// Scan inclusivo dentro de cada warp via shuffle (sem barreira), depois combina os
// NWARPS totais de warp em shared memory (1 __syncthreads) e aplica o offset.
// Vizinhos do halo cruzam warps, então os limbs brutos do tile vão para shared.

static constexpr int PSCAN_NWARPS = PSCAN_TILE / 32;

// Soma carry-lookahead de dois dígitos já normalizados, cooperativa no bloco inteiro.
// wG/wP são buffers de shared memory de tamanho PSCAN_NWARPS (totais por warp).
__device__ static Data64 cla_block(Data64 s, int cin,
                                   unsigned *wG, unsigned *wP, int *tile_cout)
{
    int t = threadIdx.x;
    int lane = t & 31;
    int warp = t >> 5;
    unsigned g = (unsigned)((s >> LIMB_BITS) & 1ULL);      // generate
    unsigned p = ((s & LIMB_MASK) == LIMB_MASK) ? 1u : 0u; // propagate

    // 1) scan inclusivo (G,P) dentro do warp, via shuffle.
#pragma unroll
    for (int d = 1; d < 32; d <<= 1)
    {
        unsigned gl = __shfl_up_sync(FULL_MASK, g, d);
        unsigned pl = __shfl_up_sync(FULL_MASK, p, d);
        if (lane >= d)
        {
            g = g | (p & gl);
            p = p & pl;
        }
    }
    // exclusivo intra-warp (agregado de [warpbase .. t-1]) via shuffle ↑1.
    unsigned eg = __shfl_up_sync(FULL_MASK, g, 1);
    unsigned ep = __shfl_up_sync(FULL_MASK, p, 1);
    if (lane == 0)
    {
        eg = 0u;
        ep = 1u;
    }

    // 2) publica o total de cada warp (lane 31 = agregado inclusivo do warp).
    if (lane == 31)
    {
        wG[warp] = g;
        wP[warp] = p;
    }
    __syncthreads();

    // 3) prefixo EXCLUSIVo entre warps [0..warp-1] (fold em ordem low→high) e total.
    unsigned Gpre = 0u, Ppre = 1u; // agregado dos warps menos significativos que `warp`
    unsigned Gtot = 0u, Ptot = 1u; // agregado de todos os warps
#pragma unroll
    for (int w = 0; w < PSCAN_NWARPS; w++)
    {
        unsigned gw = wG[w], pw = wP[w];
        if (w < warp)
        {
            Gpre = gw | (pw & Gpre);
            Ppre = pw & Ppre;
        }
        Gtot = gw | (pw & Gtot);
        Ptot = pw & Ptot;
    }

    // 4) carry que ENTRA na posição t = combine(prefixo-de-warps, exclusivo-intra-warp)
    //    aplicado a cin. low = prefixo de warps; high = exclusivo intra-warp.
    unsigned Cg = eg | (ep & Gpre);
    unsigned Cp = ep & Ppre;
    int carry_in = (int)(Cg | (Cp & (unsigned)cin));

    // carry que SAI do tile = agregado do bloco inteiro aplicado a cin.
    *tile_cout = (int)(Gtot | (Ptot & (unsigned)cin));

    __syncthreads(); // wG/wP serão reusados na próxima chamada
    return ((s & LIMB_MASK) + (Data64)carry_in) & LIMB_MASK;
}

__global__ static void pscan_normalize(
    Data64 *__restrict__ d_dst,
    const Data64 *__restrict__ d_src,
    int n_dst, int src_stride, int n_batch)
{
    int cand = blockIdx.x;
    if (cand >= n_batch)
        return;
    int t = threadIdx.x;
    const Data64 *src = d_src + (size_t)cand * src_stride;
    Data64 *dst = d_dst + (size_t)cand * n_dst;

    __shared__ Data64 sraw[PSCAN_TILE];
    __shared__ unsigned wG[PSCAN_NWARPS];
    __shared__ unsigned wP[PSCAN_NWARPS];
    __shared__ Data64 sprev[3]; // limbs brutos das posições base-1, base-2, base-3

    if (t < 3)
        sprev[t] = 0ULL;
    int cin1 = 0, cin2 = 0, cin3 = 0; // carries escalares entre tiles

    for (int base = 0; base < n_dst; base += PSCAN_TILE)
    {
        int j = base + t;
        sraw[t] = (j < n_dst) ? src[j] : 0ULL;
        __syncthreads();
        // Halo com vizinhos no tile (shared) ou no tile anterior (sprev).
        Data64 raw0 = sraw[t];                                          // posição j
        Data64 raw1 = (t >= 1) ? sraw[t - 1] : sprev[0];                // posição j-1
        Data64 raw2 = (t >= 2) ? sraw[t - 2] : sprev[(t == 0) ? 1 : 0]; // j-2
        Data64 raw3 = (t >= 3) ? sraw[t - 3] : sprev[2 - t];            // j-3

        Data64 c0 = raw0 & LIMB_MASK;         // plano 0, dígito j
        Data64 c1 = (raw1 >> 16) & LIMB_MASK; // plano 1, deslocado +1
        Data64 c2 = (raw2 >> 32) & LIMB_MASK; // plano 2, deslocado +2
        Data64 c3 = (raw3 >> 48) & LIMB_MASK; // plano 3, deslocado +3

        int cout;
        Data64 r1 = cla_block(c0 + c1, cin1, wG, wP, &cout);
        cin1 = cout;
        Data64 r2 = cla_block(c2 + c3, cin2, wG, wP, &cout);
        cin2 = cout;
        Data64 digit = cla_block(r1 + r2, cin3, wG, wP, &cout);
        cin3 = cout;

        if (j < n_dst)
            dst[j] = digit;

        // Preserva os 3 últimos limbs brutos do tile para o halo do próximo.
        __syncthreads();
        if (t == 0)
        {
            sprev[0] = sraw[PSCAN_TILE - 1];
            sprev[1] = sraw[PSCAN_TILE - 2];
            sprev[2] = sraw[PSCAN_TILE - 3];
        }
        __syncthreads();
    }
}

#endif // PSCAN_TILE == 32

#else
#error "CARRY_NORM_ALG deve ser CARRY_ALG_SINGLE_TILE, CARRY_ALG_MULTI_TILE, CARRY_ALG_SEQUENTIAL ou CARRY_ALG_PREFIX_SCAN"
#endif

// ── Monitor avançado: stats globais de carry ─────────────────────────────────

void carry_stats_print_and_reset()
{
#ifdef MR_ADVANCED_MONITOR
#if CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE
    unsigned long long h_for, h_dowhile;
    cudaMemcpyFromSymbol(&h_for, g_for_count, sizeof(h_for));
    cudaMemcpyFromSymbol(&h_dowhile, g_dowhile_count, sizeof(h_dowhile));
    if (h_for > 0)
        printf("[carry_16bits] for=%llu  do-while=%llu  media=%.3f iter/tile\n",
               h_for, h_dowhile, (double)h_dowhile / (double)h_for);
    unsigned long long zero = 0;
    cudaMemcpyToSymbol(g_for_count, &zero, sizeof(zero));
    cudaMemcpyToSymbol(g_dowhile_count, &zero, sizeof(zero));
#endif
#endif
}

// ── Métodos de BigIntNTTBatch que lançam os kernels de carry ──────────────────

void BigIntNTTBatch::vadd_raw_buf(Data64 *d_dst, int n_dst, cudaStream_t s)
{
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    int n_tiles = (n_dst + CARRY_TILE - 1) / CARRY_TILE;
    vadd_from_raw_batch<<<dim3(n_tiles, n_batch), CARRY_TILE, 0, s>>>(
        d_dst, d_buf_A, n_dst, padded, n_batch);
#elif CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE || CARRY_NORM_ALG == CARRY_ALG_PREFIX_SCAN
    constexpr int THR = MR_THR_REDUCE;
    unsigned bp = (unsigned)(n_dst + THR - 1) / THR;
    vadd_from_raw_batch<<<dim3(bp, (unsigned)n_batch), THR, 0, s>>>(
        d_dst, d_buf_A, n_dst, padded, n_batch);
#endif
    // SEQUENTIAL: não tem vadd separado — usar add_raw_buf_and_carry diretamente
}

void BigIntNTTBatch::carry_to_limbs(Data64 *d_out, int n_out, cudaStream_t s)
{
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    constexpr int THR = MR_CARRY_INTER_THR;
    int n_tiles = (n_out + CARRY_TILE - 1) / CARRY_TILE;
    int inter_blk = (n_batch + THR - 1) / THR;
    carry_intra_copy<<<dim3(n_tiles, n_batch), CARRY_TILE, 0, s>>>(
        d_out, d_buf_A, d_tile_carry, n_out, padded, n_batch);
    carry_inter_tiles<<<inter_blk, THR, 0, s>>>(
        d_out, d_tile_carry, n_out, n_batch);

#elif CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE
    carry_16bits<<<n_batch, CARRY_TILE, 0, s>>>(d_buf_A, d_out, n_out, padded, n_batch);

#elif CARRY_NORM_ALG == CARRY_ALG_SEQUENTIAL
    int blk = (n_batch + 31) / 32;
    carry_sequential<<<blk, 32, 0, s>>>(d_out, d_buf_A, n_out, padded, n_batch);

#elif CARRY_NORM_ALG == CARRY_ALG_PREFIX_SCAN
    pscan_normalize<<<n_batch, PSCAN_TILE, 0, s>>>(d_out, d_buf_A, n_out, padded, n_batch);
#endif
}

void BigIntNTTBatch::carry_after_vadd(Data64 *d_dst, int n_dst, cudaStream_t s)
{
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    constexpr int THR = MR_CARRY_INTER_THR;
    int n_tiles = (n_dst + CARRY_TILE - 1) / CARRY_TILE;
    int inter_blk = (n_batch + THR - 1) / THR;
    carry_intra_copy<<<dim3(n_tiles, n_batch), CARRY_TILE, 0, s>>>(
        d_dst, d_dst, d_tile_carry, n_dst, n_dst, n_batch);
    carry_inter_tiles<<<inter_blk, THR, 0, s>>>(
        d_dst, d_tile_carry, n_dst, n_batch);
#elif CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE
    carry_16bits<<<n_batch, CARRY_TILE, 0, s>>>(d_dst, d_dst, n_dst, n_dst, n_batch);
#elif CARRY_NORM_ALG == CARRY_ALG_PREFIX_SCAN
    pscan_normalize<<<n_batch, PSCAN_TILE, 0, s>>>(d_dst, d_dst, n_dst, n_dst, n_batch);
#endif
    // SEQUENTIAL: no-op — carry já foi feito em add_raw_buf_and_carry
}

void BigIntNTTBatch::add_raw_buf_and_carry(Data64 *d_dst, int n_dst,
                                           cudaStream_t s)
{
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE || CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE || CARRY_NORM_ALG == CARRY_ALG_PREFIX_SCAN
    vadd_raw_buf(d_dst, n_dst, s);
    carry_after_vadd(d_dst, n_dst, s);
#elif CARRY_NORM_ALG == CARRY_ALG_SEQUENTIAL
    int blk = (n_batch + 31) / 32;
    vadd_carry_sequential<<<blk, 32, 0, s>>>(d_dst, d_buf_A, n_dst, padded, n_batch);
#endif
}

void BigIntNTTBatch::add_and_carry(Data64 *d_a, const Data64 *d_b, int n, int n_passes,
                                   cudaStream_t s)
{
#if CARRY_NORM_ALG == CARRY_ALG_MULTI_TILE
    constexpr int THR = MR_CARRY_INTER_THR;
    int n_tiles = (n + CARRY_TILE - 1) / CARRY_TILE;
    int inter_blk = (n_batch + THR - 1) / THR;
    vadd_batch<<<dim3(n_tiles, n_batch), CARRY_TILE, 0, s>>>(
        d_a, d_a, d_b, n, n_batch);
    carry_intra_copy<<<dim3(n_tiles, n_batch), CARRY_TILE, 0, s>>>(
        d_a, d_a, d_tile_carry, n, n, n_batch, n_passes);
    carry_inter_tiles<<<inter_blk, THR, 0, s>>>(
        d_a, d_tile_carry, n, n_batch);

#elif CARRY_NORM_ALG == CARRY_ALG_SINGLE_TILE
    constexpr int THR = MR_THR_REDUCE;
    unsigned bp = (unsigned)(n + THR - 1) / THR;
    vadd_batch<<<dim3(bp, (unsigned)n_batch), THR, 0, s>>>(
        d_a, d_a, d_b, n, n_batch);
    carry_16bits<<<n_batch, CARRY_TILE, 0, s>>>(d_a, d_a, n, n, n_batch);

#elif CARRY_NORM_ALG == CARRY_ALG_SEQUENTIAL
    // carry_sequential lê d_src e escreve d_dst; em-place é seguro (j cresce)
    // Mas primeiro precisamos somar d_b em d_a
    constexpr int THR = MR_THR_REDUCE;
    unsigned bp = (unsigned)(n + THR - 1) / THR;
    vadd_batch<<<dim3(bp, (unsigned)n_batch), THR, 0, s>>>(
        d_a, d_a, d_b, n, n_batch);
    int blk = (n_batch + 31) / 32;
    carry_sequential<<<blk, 32, 0, s>>>(d_a, d_a, n, n, n_batch);

#elif CARRY_NORM_ALG == CARRY_ALG_PREFIX_SCAN
    constexpr int THR = MR_THR_REDUCE;
    unsigned bp = (unsigned)(n + THR - 1) / THR;
    vadd_batch<<<dim3(bp, (unsigned)n_batch), THR, 0, s>>>(
        d_a, d_a, d_b, n, n_batch);
    pscan_normalize<<<n_batch, PSCAN_TILE, 0, s>>>(d_a, d_a, n, n, n_batch);
#endif
}
