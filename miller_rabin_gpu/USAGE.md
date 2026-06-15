# Miller-Rabin GPU — Documentação da API

Teste de primalidade de Miller-Rabin em GPU para números grandes (centenas de dígitos decimais), usando aritmética de Montgomery com multiplicação via NTT na GPU.

---

## Dependências

- CUDA Toolkit
- GMP (`libgmp`)
- [GPU-NTT](https://github.com/Alisah-Ozcan/GPU-NTT) (baixado automaticamente pelo CMake via FetchContent)

---

## Incluindo no seu projeto

Adicione ao seu `CMakeLists.txt` os arquivos fonte e o include path:

```cmake
target_sources(seu_target PRIVATE
    miller_rabin_gpu/miller_rabin_runner.cu
    miller_rabin_gpu/bigint_ntt.cu
    miller_rabin_gpu/carry_norm.cu
    miller_rabin_gpu/batch_mod_ctx.cu
    miller_rabin_gpu/mod_perf.cu
    miller_rabin_gpu/reduce_montgomery.cu
    miller_rabin_gpu/reduce_barrett.cu
)
target_include_directories(seu_target PRIVATE miller_rabin_gpu/)
target_link_libraries(seu_target PRIVATE GPUNTT::ntt CUDA::cudart ${GMP_LIB})
set_target_properties(seu_target PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
```

Nos seus arquivos, inclua:

```cpp
#include "batch_mod_ctx.cuh"
#include "miller_rabin_runner.cuh"
```

---

## Conceitos fundamentais

### Representação dos números — limbs de 16 bits

Todos os números são representados como arrays de `uint64_t` no formato **little-endian com limbs de 16 bits**: cada elemento armazena 16 bits do número, do menos significativo para o mais significativo.

```cpp
// Converte um mpz_t para o formato de limbs
void mpz_to_limbs(uint64_t* out, int n_limbs, const mpz_t x) {
    mpz_t tmp; mpz_init_set(tmp, x);
    for (int i = 0; i < n_limbs; i++) {
        out[i] = mpz_get_ui(tmp) & 0xFFFF;
        mpz_tdiv_q_2exp(tmp, tmp, 16);
    }
    mpz_clear(tmp);
}
```

Para calcular quantos limbs são necessários dado o número de dígitos decimais:

```cpp
int n_limbs = limbs_for_digits(numero_de_digitos + 4);  // +4 de margem
```

### Decomposição N-1 = 2^s · d

O teste de Miller-Rabin requer decompor `N-1 = 2^s · d` onde `d` é ímpar. O valor de `s` define qual função usar:

- **s = 1**: use `miller_rabin_s1` (versão otimizada)
- **s > 1**: use `miller_rabin`

---

## `BatchModCtx` — Contexto de redução modular

Encapsula toda a aritmética modular para um **batch** de candidatos processados simultaneamente na GPU. Todos os candidatos do batch devem ter o mesmo `n_limbs`.

O **algoritmo de redução** é escolhido em tempo de compilação via `MOD_REDUCTION_ALG` (em `params.cmake`):

| Valor                      | Forma de trabalho        | Observações                                                            |
| -------------------------- | ------------------------ | --------------------------------------------------------------------- |
| `MOD_RED_MONTGOMERY`       | Montgomery (`x·R mod N`) | REDC clássico via NTT. Padrão.                                         |
| `MOD_RED_BARRETT`          | resíduo plano (`x mod N`)| μ = ⌊b^{2k}/N⌋ pré-computado; 2 multiplicações via NTT + finalize. Exige que todos os candidatos do batch tenham a **mesma magnitude** (mesmo nº de limbs tight de N — vale para candidatos esparsos). |
| `MOD_RED_BURNIKEL_ZIEGLER` | —                        | **Não implementado** (erro de compilação claro).                      |

A interface abaixo é a mesma para qualquer algoritmo: a "forma de trabalho" muda conforme o backend, mas `to_residue_batch`/`from_residue_batch` sempre convertem de/para o inteiro normal, e `d_one_res`/`d_Nm1_res` guardam `1` e `N-1` na forma de trabalho.

### Construção

Dois construtores disponíveis:

```cpp
// Construtor 1: a partir de limbs pré-computados
BatchModCtx mont(N_all, n_limbs, n_batch, device_id = 0);
```

| Parâmetro   | Tipo                      | Descrição                                                               |
| ----------- | ------------------------- | ----------------------------------------------------------------------- |
| `N_all`     | `const vector<uint64_t>&` | Array flat `[n_batch × n_limbs]` com os módulos N, little-endian 16-bit |
| `n_limbs`   | `int`                     | Limbs de 16 bits por candidato                                          |
| `n_batch`   | `int`                     | Número de candidatos no batch                                           |
| `device_id` | `int`                     | Índice da GPU (padrão: 0; use `cudaGetDeviceCount` para listar)         |

```cpp
// Construtor 2: diretamente de mpz_t — calcula n_limbs automaticamente
BatchModCtx mont(numbers, device_id = 0);
```

| Parâmetro   | Tipo                    | Descrição                              |
| ----------- | ----------------------- | -------------------------------------- |
| `numbers`   | `const vector<mpz_t*>&` | Ponteiros para os módulos N como mpz_t |
| `device_id` | `int`                   | Índice da GPU (padrão: 0)              |

### Métodos

```cpp
// Converte valores do host para a forma de trabalho (também no host)
mont.to_residue_batch(x_all, out_all);

// Converte de volta: GPU (forma de trabalho) → host (normal)
mont.from_residue_batch(d_x, out_all);

// d_out = d_A * d_B mod N  (GPU, forma de trabalho)
mont.modmul_batch(d_A, d_B, d_out);

// d_out = d_A^2 mod N  (GPU, forma de trabalho)
mont.modsq_batch(d_A, d_out);

// d_passed[t] = 1 se d_r[t] == 1 ou d_r[t] == N-1  (na forma de trabalho)
mont.check_passed(d_r, d_passed);
```

### Campos públicos úteis

```cpp
mont.n_limbs     // limbs por candidato
mont.n_batch     // número de candidatos
mont.d_N         // módulos N na GPU  [n_batch × n_limbs]
mont.d_one_res  // 1 na forma de trabalho por candidato  [n_batch × n_limbs]
mont.d_Nm1_res  // N-1 na forma de trabalho por candidato  [n_batch × n_limbs]
mont.perf        // contadores de tempo interno; .print() para exibir
```

---

## Funções de teste — `miller_rabin_runner.cuh`

A constante `DEFAULT_WITNESSES` contém os 16 primeiros primos `{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}` e pode ser usada como valor conveniente para o parâmetro `witnesses`.

### `gpu_miller_rabin_s1`

Versão otimizada para `s = 1` (N ≡ 3 mod 4, ou seja, N-1 = 2·d).  
Computa `a^d mod N` e verifica se o resultado é `±1 mod N`.

```cpp
std::vector<bool> gpu_miller_rabin_s1(
    BatchModCtx& mont,
    const std::vector<uint64_t>& d_all,          // d = (N-1)/2,  flat [n_total × n_limbs]
    const std::vector<uint64_t>& Nm1_all,        // N-1,          flat [n_total × n_limbs]
    int n_total,
    const std::vector<uint32_t>& witnesses,      // testemunhas a usar; e.g. DEFAULT_WITNESSES
    const char* label,
    bool show_report   = false,   // exibe relatório de performance
    bool show_progress = false    // exibe barra de progresso
);
```

### `gpu_miller_rabin`

Versão geral para qualquer `s ≥ 1`. Delega para `gpu_miller_rabin_s1` quando `s == 1`.  
Após computar `a^d`, realiza até `s-1` squarings extras verificando N-1 a cada passo.

```cpp
std::vector<bool> gpu_miller_rabin(
    BatchModCtx& mont,
    const std::vector<uint64_t>& d_all,          // d ímpar onde N-1 = 2^s·d, flat [n_total × n_limbs]
    const std::vector<uint64_t>& Nm1_all,        // N-1,                       flat [n_total × n_limbs]
    int s,
    int n_total,
    const std::vector<uint32_t>& witnesses,      // testemunhas a usar; e.g. DEFAULT_WITNESSES
    const char* label,
    bool show_report   = false,
    bool show_progress = false
);
```

**Retorno:** `vector<bool>` de tamanho `n_total` — `true` = provavelmente primo, `false` = composto.

---

## Exemplo completo

```cpp
#include "batch_mod_ctx.cuh"
#include "miller_rabin_runner.cuh"
#include <gmp.h>
#include <vector>
#include <cstdio>

// Retorna true se N (em limbs) é provavelmente primo
bool is_probably_prime(const mpz_t N) {
    int digits  = (int)mpz_sizeinbase(N, 10);
    int n_limbs = limbs_for_digits(digits + 4);

    // Monta arrays N, N-1, d onde N-1 = 2^s * d
    std::vector<uint64_t> N_lims(n_limbs, 0);
    std::vector<uint64_t> Nm1_lims(n_limbs, 0);
    std::vector<uint64_t> d_lims(n_limbs, 0);

    mpz_t Nm1, d; mpz_inits(Nm1, d, nullptr);
    mpz_sub_ui(Nm1, N, 1);
    mpz_set(d, Nm1);

    int s = 0;
    while (mpz_even_p(d)) { mpz_tdiv_q_2exp(d, d, 1); s++; }

    // Converte para limbs (little-endian, 16 bits por limb)
    auto to_lims = [&](std::vector<uint64_t>& v, const mpz_t x) {
        mpz_t tmp; mpz_init_set(tmp, x);
        for (int i = 0; i < n_limbs; i++) {
            v[i] = mpz_get_ui(tmp) & 0xFFFF;
            mpz_tdiv_q_2exp(tmp, tmp, 16);
        }
        mpz_clear(tmp);
    };
    to_lims(N_lims,   N);
    to_lims(Nm1_lims, Nm1);
    to_lims(d_lims,   d);
    mpz_clears(Nm1, d, nullptr);

    // Cria contexto e executa
    BatchModCtx mont(N_lims, n_limbs, 1);

    std::vector<bool> result;
    if (s == 1)
        result = gpu_miller_rabin_s1(mont, d_lims, Nm1_lims, 1, DEFAULT_WITNESSES, "N");
    else
        result = gpu_miller_rabin(mont, d_lims, Nm1_lims, s, 1, DEFAULT_WITNESSES, "N");

    return result[0];
}

int main() {
    mpz_t N; mpz_init(N);
    // M521 = 2^521 - 1  (primo de Mersenne conhecido)
    mpz_ui_pow_ui(N, 2, 521);
    mpz_sub_ui(N, N, 1);

    if (is_probably_prime(N))
        printf("Provavelmente primo\n");
    else
        printf("Composto\n");

    mpz_clear(N);
}
```

### Processando múltiplos números em batch

Para melhor performance, agrupe candidatos com o mesmo `n_limbs` em um único `BatchModCtx`:

```cpp
int n_batch = (int)candidatos.size();
int n_limbs = limbs_for_digits(max_digitos + 4);

// Monta arrays flat [n_batch × n_limbs]
std::vector<uint64_t> N_all  (n_batch * n_limbs, 0);
std::vector<uint64_t> Nm1_all(n_batch * n_limbs, 0);
std::vector<uint64_t> d_all  (n_batch * n_limbs, 0);
// ... preenche cada candidato em N_all[i*n_limbs .. (i+1)*n_limbs]

BatchModCtx mont(N_all, n_limbs, n_batch);
auto alive = gpu_miller_rabin_s1(mont, d_all, Nm1_all, n_batch, DEFAULT_WITNESSES, "batch");

for (int i = 0; i < n_batch; i++)
    printf("candidato %d: %s\n", i, alive[i] ? "primo" : "composto");
```

---

## Parâmetros configuráveis

Todos os parâmetros de tuning estão centralizados em `config.h`. Redefina-os antes de incluir qualquer header (via `-D` no compilador ou `#define` antes dos `#include`) para sobrescrever os padrões.

| `#define`                 | Padrão | Descrição                                                                                                    |
| ------------------------- | ------ | ------------------------------------------------------------------------------------------------------------ |
| `MR_WINDOW_BITS`          | `8`    | Bits por janela na exponenciação sliding-window. Mais bits = menos muls, mais VRAM (tabela de 2^k entradas). |
| `MR_BATCH_SIZE`           | `128`  | Candidatos por batch de GPU. Reduza se a VRAM for insuficiente.                                              |
| `MR_BLOCK_THREADS`        | `256`  | Threads por bloco CUDA (kernels internos). Deve ser múltiplo de 32.                                          |
| `MR_CS_TILE`              | `256`  | Tamanho do tile na subtração condicional de Montgomery.                                                      |
| `MR_PERF_RING`            | `12`   | Profundidade do ring de eventos CUDA para profiling interno.                                                 |
| `MR_PROGRESS_INTERVAL_MS` | `2000` | Intervalo mínimo (ms) entre atualizações da barra de progresso.                                              |

`miller_rabin_runner.cuh`:

| Constante           | Padrão       | Descrição                                                        |
| ------------------- | ------------ | ---------------------------------------------------------------- |
| `DEFAULT_WITNESSES` | 2,3,5,...,53 | Vetor padrão de 16 testemunhas; passe como argumento `witnesses` |

`bigint_ntt.cuh`:

| Constante   | Padrão | Descrição                                              |
| ----------- | ------ | ------------------------------------------------------ |
| `LIMB_BITS` | `16`   | Bits por limb (redefina antes de incluir para usar 32) |

---

## Arquivos da biblioteca

```
config.h                — Parâmetros configuráveis (WINDOW_BITS, BATCH_SIZE, etc.)
miller_rabin_runner.cuh   — API pública: gpu_miller_rabin_s1, gpu_miller_rabin
miller_rabin_runner.cu    — Implementação dos kernels de exponenciação
batch_mod_ctx.cuh         — API pública: BatchModCtx (struct + interface)
batch_mod_ctx.cu          — Núcleo comum: ctor/dtor, to/from_residue, modmul/modsq
mod_perf.cu               — Impressão da árvore de perfil (BatchModCtx::print_perf)
reduce_montgomery.cu      — Redução de Montgomery (REDC) + cond_sub  [MOD_RED_MONTGOMERY]
reduce_barrett.cu         — Redução de Barrett (μ pré-NTT + finalize) [MOD_RED_BARRETT]
gmp_helpers.cuh           — Conversões host limbs↔mpz_t
mod_internal.cuh          — Macros internas (CU, TSTART/TSTOP)
bigint_ntt.cuh            — BigIntNTTBatch (multiplicação NTT, usado internamente)
bigint_ntt.cu             — Kernels NTT + schoolbook
carry_norm.cu             — Algoritmos de normalização de carry (CARRY_NORM_ALG)
correctness_tests.cuh     — run_correctness_tests(), run_known_prime_tests(),
                            run_general_s_prime_tests()  (testes opcionais)
```
