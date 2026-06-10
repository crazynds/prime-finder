#pragma once
#include "constants.h"
// config.h — Parâmetros configuráveis do Miller-Rabin GPU.

/** --------------------------------------------------
                        Algoritmo
------------------------------------------------------*/

// Janela de exponenciação. Range: 4–10.
#define MR_WINDOW_BITS 8

// Candidatos por chamada GPU. Quanto maior, mais VRAM usada.
#define MR_BATCH_SIZE 256

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
// carry_inter_tiles (CARRY_ALG_MULTI_TILE): threads por bloco na fase inter-tile
#define MR_CARRY_INTER_THR 32

/** --------------------------------------------------
             Carry normalização — algoritmos disponíveis

  CARRY_ALG_SINGLE_TILE — 1 bloco por candidato, MR_CARRY_TILE threads.
    Todos os tiles de um candidato processados por um único bloco com
    shared-memory carry entre tiles. Menor ocupação de GPU mas simples.

  CARRY_ALG_MULTI_TILE  — 2 fases separadas:
    Fase 1 (intra): n_tiles × n_batch blocos em paralelo, cada um
      normaliza MR_CARRY_TILE elementos com carry de saída em d_tile_carry.
    Fase 2 (inter): 1 thread por candidato, propaga carries entre tiles
      sequencialmente. Melhor ocupação na fase intra.

  CARRY_ALG_SEQUENTIAL  — 1 thread por candidato, loop sequencial puro
    sobre todos os elementos. Mínimo de shared memory, máxima serialização.
    Útil como baseline ou quando outros algoritmos têm race conditions.

------------------------------------------------------*/

// Selecione o algoritmo desejado:
// #define CARRY_NORM_ALG CARRY_ALG_MULTI_TILE
// #define CARRY_NORM_ALG CARRY_ALG_SINGLE_TILE
#define CARRY_NORM_ALG CARRY_ALG_SEQUENTIAL

// Tamanho da janeal de carry (threads por bloco).
// Usado pelos algoritmos de TILE
#define MR_CARRY_TILE 256

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

/** --------------------------------------------------
      Algoritmo de multiplicação big-integer
      (mont_mul_batch / mont_sq_batch)
------------------------------------------------------*/

// #define MONT_MUL_ALG MONT_MUL_ALG_NTT
#define MONT_MUL_ALG MONT_MUL_ALG_SCHOOLBOOK
