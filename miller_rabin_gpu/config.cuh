#pragma once
// config.cuh — Parâmetros configuráveis do Miller-Rabin GPU.

/** --------------------------------------------------
                        Algoritmo
------------------------------------------------------*/

// Janela de exponenciação. Range: 4–10.
#define MR_WINDOW_BITS 8

// Candidatos por chamada GPU. Quanto maior, mais VRAM usada.
#define MR_BATCH_SIZE 32

/** --------------------------------------------------
                    Threads por bloco
         Todos devem ser múltiplo de 32 (warp size).
------------------------------------------------------*/

// load_padded_batch: copia e zero-pad limbs de entrada para o buffer NTT
#define MR_THR_LOAD 256
// pmul_batch / psq_batch: multiplicação/quadrado pointwise no domínio NTT
#define MR_THR_PMUL 256
// extract_low / shift_right: kernels de redução Montgomery (reduce_batch)
#define MR_THR_REDUCE 256
// select_window_kernel: lê a entrada correta da tabela de potências para a janela atual
#define MR_THR_SELECT_WIN 256
// check_passed_kernel / check_equals_kernel: compara resultado MR com 1 ou N-1
#define MR_THR_CHECK 256
// carry_inter_tiles: propaga carry entre tiles, 1 thread por candidato
#define MR_CARRY_INTER_THR 32

/** --------------------------------------------------
                   Carry normalização
------------------------------------------------------*/

// carry_intra_copy / carry_16bits: tamanho do tile (= threads por bloco). Reduzir se smem estourar.
#define MR_CARRY_TILE 32

// Descomente para usar o algoritmo 2-fase (carry_intra_copy + carry_inter_tiles).
// Por padrão usa carry_16bits (1 kernel, 1 bloco por candidato).
// #define CARRY_MULTI_TILE

// Descomente para carry_intra_copy usar 1 thread sequencial por tile (só com CARRY_MULTI_TILE).
// #define CARRY_INTRA_SEQUENTIAL

/** --------------------------------------------------
              Subtração condicional (cond_sub)
------------------------------------------------------*/

// cs_phase1 / cs_apply: tamanho do tile da subtração condicional (cond_sub_batch)
#define MR_CS_TILE 256

/** --------------------------------------------------
                       Monitoring
------------------------------------------------------*/

// Intervalo mínimo entre atualizações da barra de progresso (ms).
#define MR_PROGRESS_INTERVAL_MS 2000
