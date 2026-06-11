# params.cmake — Configurações do build. Edite aqui, depois rode cmake.

# ── Miller-Rabin GPU ──────────────────────────────────────────────────────────

# Janela de exponenciação (range: 4–10)
set(MR_WINDOW_BITS 8)

# Candidatos por chamada GPU. Quanto maior, mais VRAM usada.
set(MR_BATCH_SIZE 256)

# Threads por bloco de cada kernel (múltiplo de 32)
set(MR_THR_LOAD        256)
set(MR_THR_PMUL        256)
set(MR_THR_REDUCE      256)
set(MR_THR_SELECT_WIN  256)
set(MR_THR_CHECK       256)
set(MR_CARRY_INTER_THR  32)

# Algoritmo de carry: CARRY_ALG_SEQUENTIAL | CARRY_ALG_SINGLE_TILE | CARRY_ALG_MULTI_TILE
set(CARRY_NORM_ALG CARRY_ALG_MULTI_TILE)

# Tamanho do tile de carry (threads por bloco, usado pelos algoritmos TILE)
set(MR_CARRY_TILE 256)

# Tile da subtração condicional
set(MR_CS_TILE 256)

# Intervalo mínimo entre updates da barra de progresso (ms)
set(MR_PROGRESS_INTERVAL_MS 2000)

# Algoritmo de multiplicação big-integer: MONT_MUL_ALG_NTT | MONT_MUL_ALG_SCHOOLBOOK
set(MONT_MUL_ALG MONT_MUL_ALG_NTT)

# ── GPU-NTT (biblioteca Alisah-Ozcan/GPU-NTT) ─────────────────────────────────

# Ativa tabelas de kernel otimizadas para RTX 4090 (Compute Capability 8.9).
# ON  → usa configs CC_89 para n_power 27 e 28 (shared_mem 1024×T, grids maiores)
# OFF → usa configs genéricas para essas faixas
set(GPUNTT_CC89 OFF)

# Layout da NTT no batch:
#   PerPolynomial  — NTT por linha (por candidato). Padrão recomendado.
#   PerCoefficient — NTT por coluna (por índice de coeficiente entre candidatos).
set(GPUNTT_NTT_LAYOUT PerPolynomial)
