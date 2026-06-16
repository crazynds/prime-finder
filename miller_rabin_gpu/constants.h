// ── Carry normalização ────────────────────────────────────────────────────────
#define CARRY_ALG_SINGLE_TILE 1
#define CARRY_ALG_MULTI_TILE 2
#define CARRY_ALG_SEQUENTIAL 3
#define CARRY_ALG_PREFIX_SCAN 4

// ── Multiplicação de big-integer (mont_mul / mont_sq) ─────────────────────────
//
// Algoritmo ÚNICO selecionado por MUL_ALG (params.cmake). Define tanto o produto
// quanto, quando for NTT, qual backend a classe Multiplier resolve (ver
// ops/mul/multiplier.cuh).
//
// MUL_SCHOOLBOOK — O(n²) por convolução direta. Só para n_limbs pequeno; baseline.
//                  (A redução ainda usa NTT "merge" internamente.)
// MUL_NTT_MERGE  — GPU-NTT "merge", O(n log n). Produção (padrão).
// MUL_NTT_4STEP  — GPU-NTT "4step" (radix; transposes). Exige logn ∈ [12,24].
#define MUL_SCHOOLBOOK 1
#define MUL_NTT_MERGE 2
#define MUL_NTT_4STEP 3

// ── Redução modular (modmul / modsq) ──────────────────────────────────────────
//
// MOD_RED_MONTGOMERY       — REDC clássico. Forma de trabalho = Montgomery (x·R mod N).
// MOD_RED_BARRETT          — Redução de Barrett. Forma de trabalho = resíduo plano
//                            (x mod N). Pré-computa μ = floor(b^{2k}/N), reusa NTT.
// MOD_RED_BURNIKEL_ZIEGLER — Divisão D&C de Burnikel-Ziegler (não implementado).
#define MOD_RED_MONTGOMERY 1
#define MOD_RED_BARRETT 2
#define MOD_RED_BURNIKEL_ZIEGLER 3
