#pragma once
// config.cuh — Parâmetros configuráveis do Miller-Rabin GPU.
//
// Modifique aqui para ajustar performance e comportamento sem tocar no código interno.

// ── Exponenciação por janela deslizante ──────────────────────────────────────
// Bits por janela. Maior valor = menos multiplicações, mas mais memória para a tabela.
// Tabela consome: 2^MR_WINDOW_BITS * n_total * n_limbs * 8 bytes.
// Valores razoáveis: 4–10. Padrão: 8.
#ifndef MR_WINDOW_BITS
#  define MR_WINDOW_BITS 8
#endif
#define MR_WINDOW_SIZE (1 << MR_WINDOW_BITS)

// ── Tamanho do batch ──────────────────────────────────────────────────────────
// Candidatos processados por chamada ao GPU. Afeta uso de VRAM.
// Reduzir se ficar sem memória; aumentar em GPUs com muita VRAM.
#ifndef MR_BATCH_SIZE
#  define MR_BATCH_SIZE 128
#endif

// ── Threads por bloco (kernels internos) ─────────────────────────────────────
// Usado em select_window_kernel, check_equals_kernel e cs_apply.
// Deve ser múltiplo de 32 (warp size). Padrão: 256.
#ifndef MR_BLOCK_THREADS
#  define MR_BLOCK_THREADS 256
#endif

// ── Tile de cond_sub (montgomery.cu) ─────────────────────────────────────────
// Elementos por tile na subtração condicional tileada.
// Deve ser potência de 2 e múltiplo de MR_BLOCK_THREADS.
#ifndef MR_CS_TILE
#  define MR_CS_TILE 256
#endif

// ── Ring de eventos CUDA (profiling interno) ──────────────────────────────────
// Profundidade do ring de cudaEvent_t usado para medir tempo dos kernels.
// Precisa ser >= número de seções cronometradas por mont_mul_batch (atualmente 11).
#ifndef MR_PERF_RING
#  define MR_PERF_RING 12
#endif

// ── Intervalo da barra de progresso ──────────────────────────────────────────
// Tempo mínimo entre atualizações da barra de progresso, em milissegundos.
#ifndef MR_PROGRESS_INTERVAL_MS
#  define MR_PROGRESS_INTERVAL_MS 2000
#endif
