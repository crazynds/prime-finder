// ── Carry normalização ────────────────────────────────────────────────────────
#define CARRY_ALG_SINGLE_TILE 1
#define CARRY_ALG_MULTI_TILE 2
#define CARRY_ALG_SEQUENTIAL 3
#define CARRY_ALG_PREFIX_SCAN 4

// ── Multiplicação de big-integer (mont_mul / mont_sq) ─────────────────────────
//
// MONT_MUL_ALG_NTT        — NTT-based, O(n log n). Produção.
// MONT_MUL_ALG_SCHOOLBOOK — O(n²) por convolução direta. Apenas para n_limbs
//                           pequeno (< ~512). Útil como baseline de correctness.
#define MONT_MUL_ALG_NTT 1
#define MONT_MUL_ALG_SCHOOLBOOK 2
