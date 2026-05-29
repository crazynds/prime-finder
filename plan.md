# Prime Hunter — Plano de Implementação

## Objetivo

Encontrar um primo de 50k–100k dígitos da forma:

```
N = 10^a - k * 10^b - j * 2^c - 1
```

Restrições: `a >= 50000`, `a > b > c >= 100`, `1 <= k,j <= 100`

---

## Arquitetura Geral

```
MPI rank 0  →  master (só coordena, não computa)
MPI rank 1..N  →  workers, cada rank é 1 thread de trabalho real

  Tipos de worker:
    GPU worker  : responsável por 1 GPU, faz fase 0 + fase 1
    CPU worker  : faz fase 2 (Miller-Rabin)

  GPU worker pode roubar tarefas de CPU quando não há tasks de GPU.
  CPU worker nunca faz tarefas de GPU.
```

### Exemplo de layout num cluster

```
Máquina A: 2 GPUs, 8 cores
  rank 1  → GPU worker (GPU 0)
  rank 2  → GPU worker (GPU 1)
  rank 3  → CPU worker
  rank 4  → CPU worker
  rank 5  → CPU worker
  rank 6  → CPU worker   (outros cores livres para MR)

Máquina B: 1 GPU, 4 cores
  rank 7  → GPU worker (GPU 0)
  rank 8  → CPU worker
  rank 9  → CPU worker
```

### Como cada rank descobre seu papel

Cada rank chama `cudaGetDeviceCount()` e verifica o local rank via `OMPI_COMM_WORLD_LOCAL_RANK`:

```c
local_rank = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
cudaGetDeviceCount(&ngpus);

if (local_rank < ngpus):
    sou GPU worker → cudaSetDevice(local_rank)
else:
    sou CPU worker
```

Ao iniciar, cada worker envia ao master uma mensagem `REGISTER{type=GPU|CPU}`.
O master mantém duas listas de ranks disponíveis: `gpu_free[]` e `cpu_free[]`.

---

## Fluxo de Tarefas (dois tipos)

```
gpu_task_t  { long long a, b, c }
  → enviado pelo master para um GPU worker
  → GPU worker faz fase 0 + fase 1
  → devolve ao master: lista de sobreviventes (a,b,c,k,j)
  → master empurra cada sobrevivente como cpu_task

cpu_task_t  { long long a, b, c; int k, j }
  → enviado pelo master para um CPU worker (ou GPU worker ocioso)
  → worker faz fase 2 (Miller-Rabin)
  → devolve resultado ao master
  → master grava em arquivo se for provável primo
```

---

## Master: dois níveis de fila

```
gpu_pending[]  : lista de (a,b,c) esperando um GPU worker livre
cpu_pending[]  : lista de (a,b,c,k,j) esperando um CPU worker livre

loop do master:
  MPI_Recv(qualquer fonte)

  if msg == REGISTER:
    adiciona rank na lista correta (gpu_free ou cpu_free)

  if msg == GPU_RESULT:
    para cada sobrevivente recebido: push → cpu_pending
    mark rank como livre → gpu_free
    if gpu_pending não vazio: despacha próximo gpu_task

  if msg == CPU_RESULT:
    if provável primo: grava arquivo
    mark rank como livre → cpu_free (ou gpu_free se era GPU worker)
    if cpu_pending não vazio: despacha próximo cpu_task
    elif gpu_pending não vazio e é GPU worker: despacha gpu_task

  Geração de (a,b,c): produz mais tasks conforme workers ficam livres
```

---

## Arquivos

| Arquivo            | Responsabilidade                                          |
|--------------------|-----------------------------------------------------------|
| `gen_primes.c`     | Gera arquivo binário de primos (roda uma vez)             |
| `prime_list.h/c`   | Carrega arquivo binário de primos                         |
| `sieve_table.h/c`  | Pré-computa tabelas periódicas de fase 0                  |
| `sieve.h/c`        | Fase 0 runtime: lookup O(1) na tabela                     |
| `trial_div.h/cu`   | Fase 1: kernel CUDA                                       |
| `miller_rabin.h/c` | Fase 2: GMP                                               |
| `master.h/c`       | Lógica do rank 0: filas, despacho, arquivo                |
| `worker.h/c`       | Lógica dos ranks 1..N: GPU worker e CPU worker            |
| `messages.h`       | Definição de todos os tipos de mensagem MPI               |
| `main.c`           | Inicialização MPI, decide papel do rank                   |
| `CMakeLists.txt`   | Build: MPI + CUDA + GMP + pthreads                        |

---

## Fase 0 — Sieve Simbólico com Tabela Pré-Computada

### Fundamento

Para um primo `p`, `N mod p` é periódico em (a,b,c):
- período de `a,b` = `ord_p(10)`  (divide `p-1`)
- período de `c`   = `ord_p(2)`   (divide `p-1`)

### Pré-computação (inicialização do worker)

```
Para cada primo p até SMALL_PRIME_BOUND:
  L10 = ord_p(10),  L2 = ord_p(2)
  Se L10² × L2 > MAX_TABLE_SIZE: pular

  Para cada (ra, rb, rc) em [0,L10) × [0,L10) × [0,L2):
    bitmask = 0  // 10000 bits para os 100×100 pares (k,j)
    Para k=1..100:
      j0 = ((ra - k*rb - 1) mod p) * inv(rc,p) mod p
      Para j = j0, j0+p, ... ≤ 100: setar bit (k-1)*100+(j-1)
    table[p][ra][rb][rc] = bitmask
```

### Runtime

```
bitmask = OR de table[p][a%L10][b%L10][c%L2]  para cada p tabelado
survivors = bits em 0 no bitmask
Se survivors == 0: descarta (a,b,c) sem chamar GPU
```

---

## Fase 1 — GPU Trial Division (CUDA)

1 thread CUDA por primo da lista completa (`primes_1e8.bin`).

```
ra = 10^a % p,  rb = 10^b % p,  rc = 2^c % p
inv_rc = rc^(p-2) % p

Para k=1..100:
  j0 = ((ra - k*rb - 1 + 2p) % p) * inv_rc % p
  Para j = j0..100 step p:
    atomicOr(&bitset[bit/32], 1u << (bit%32))
```

Output: bitset de 10000 bits. Sobreviventes (bit=0) → cpu_tasks enviadas ao master.

---

## Fase 2 — Miller-Rabin (GMP)

```c
N = 10^a - k*10^b - j*2^c - 1   (GMP)
r = mpz_probab_prime_p(N, 25)
→ r > 0: envia resultado ao master
```

---

## Mensagens MPI

```c
// worker → master ao iniciar
typedef struct { int type; } msg_register_t;  // type: WORKER_GPU ou WORKER_CPU

// master → worker GPU
typedef struct { long long a, b, c; } gpu_task_t;

// worker GPU → master
typedef struct {
    long long a, b, c;
    int n_survivors;
    int ks[10000], js[10000];
} gpu_result_t;

// master → worker CPU
typedef struct { long long a, b, c; int k, j; } cpu_task_t;

// worker CPU → master
typedef struct {
    long long a, b, c; int k, j;
    int result;  // 0=composto, 1=provável primo, 2=primo
} cpu_result_t;

// master → worker (terminação)
#define MSG_TERMINATE  99
```

Tags MPI: `TAG_REGISTER=1`, `TAG_GPU_TASK=2`, `TAG_GPU_RESULT=3`,
           `TAG_CPU_TASK=4`, `TAG_CPU_RESULT=5`, `TAG_TERMINATE=6`

---

## Saída em Arquivo

Apenas o master escreve.

```
phase1_survivors.txt   →  a=N b=N c=N k=N j=N
phase2_primes.txt      →  a=N b=N c=N k=N j=N PROBABLE_PRIME | PRIME
```

---

## Iteração de (a,b,c)

```
a: a_min → a_max  (externo)
b: a-1   → c_min+1
c: c_min → b-1   (interno)
```

---

## Geração dos Arquivos de Primos

```bash
./gen_primes 10000      small_primes.bin   # fase 0
./gen_primes 100000000  primes_1e8.bin     # fase 1
```

---

## Dependências

| Biblioteca    | Uso                          |
|---------------|------------------------------|
| MPI (OpenMPI) | Distribuição entre nós       |
| CUDA          | Fase 1 trial division        |
| GMP           | Fase 2 Miller-Rabin          |
| pthreads      | Não necessário (1 rank = 1 thread) |

---

## Análise de FLOPs

### Fase 0 — CPU sieve lookup

- Por primo tabelado: 3 mod + 1 OR = ~4 ops
- ~1000 primos tabelados → **~4k ops por (a,b,c)**
- Negligível, roda em < 1µs

### Fase 1 — GPU trial division

Cada thread CUDA processa 1 primo `p`:

| Operação | Custo |
|---|---|
| 3× `powmod(base, exp≈50000, p)` | 3 × 16 iters × 2 muls = **96 muls** |
| 1× `modinv` (powmod p-2) | **32 muls** |
| k-loop (100 iters) | 100 × 5 ops = **500 ops** |
| **Total por primo** | **~628 INT64 ops** |

> **Nota:** como `p < 10^8`, temos `a,b < p < 10^8`, logo `a*b < 10^16 < UINT64_MAX`.
> O `__uint128_t` no kernel **não é necessário** — plain `uint64_t` basta.
> Isso é uma otimização importante para o throughput do kernel.

Total por (a,b,c):
```
5.7M primos × 628 ops = 3.6 × 10⁹ INT64 ops
```

Throughput INT64 em GPU moderna (ex: RTX 3090): ~17 TOPS  
→ **~0.2ms por (a,b,c)** (ideal, sem contar latência de lançamento e atomics)

### Fase 2 — Miller-Rabin (GMP)

N tem ~50.000 dígitos decimais ≈ **166.000 bits** ≈ **2.600 palavras de 64 bits**

GMP usa multiplicação via FFT para números grandes:
- Custo por multiplicação de 2600-words: `O(n log n)` ≈ 2600 × 12 ≈ **31k ops**
- `modexp(N)` com expoente de 166k bits: 166k × 31k ≈ **5.1 × 10⁹ ops**
- Miller-Rabin com 25 rounds, 2 modexps cada: 50 × 5.1×10⁹ = **~2.5 × 10¹¹ ops**
- Em 1 core (~10 GOPS com SIMD GMP): **~2–5s por candidato**

### Sobreviventes esperados por (a,b,c)

Pelo teorema de Mertens, a fração de números sem fator ≤ B é ~`1 / (e^γ × ln B)`:
```
B = 10^8 → ln(10^8) ≈ 18.4 → fração ≈ 1/32 ≈ 3%
```
De 10.000 pares (k,j): **~300 sobreviventes por (a,b,c)** chegam à fase 2.

### Gargalo

```
Fase 1 GPU : ~0.2ms por (a,b,c)
Fase 2 CPU : ~300 candidatos × 2–5s = 600–1500s por (a,b,c) com 1 core
             Com 100 CPU workers: 6–15s por (a,b,c)
```

**Conclusão: fase 2 é o gargalo absoluto.** Maximizar o número de CPU workers
é mais importante que ter mais GPU workers para esta workload.

---

## Plano de Testes

### 1. Testes unitários de funções matemáticas

| Teste | O que valida |
|---|---|
| `test_powmod` | `powmod(a, e, m)` == resultado esperado para valores conhecidos |
| `test_modinv` | `modinv(x, p) * x % p == 1` para vários x,p |
| `test_ord_p` | `ord_p(10, p)`: verifica que `10^ord ≡ 1 (mod p)` e é o menor tal |
| `test_powmod_cuda` | Resultado do kernel CUDA bate com CPU para os mesmos inputs |

### 2. Validação da Fase 0

| Teste | O que valida |
|---|---|
| `test_sieve_table_vs_direct` | Para (a,b,c) pequenos: lookup na tabela == cálculo direto |
| `test_eliminated_are_composite` | Todo (k,j) marcado como composto é realmente divisível por algum primo da lista |
| `test_survivors_not_eliminated` | Todo (k,j) sobrevivente não é divisível por nenhum primo da lista da fase 0 |
| `test_periodicity` | `sieve(a,b,c)` == `sieve(a + L10, b, c)` para o período correto |

### 3. Validação da Fase 1 (GPU)

| Teste | O que valida |
|---|---|
| `test_gpu_vs_cpu_brute` | Para (a,b,c) pequenos: GPU bitset == varredura CPU ingênua (N%p para todos os primos) |
| `test_gpu_no_false_composites` | Nenhum (k,j) marcado é realmente primo (falso composto é proibido) |
| `test_gpu_consistency` | Rodar o mesmo (a,b,c) duas vezes dá o mesmo bitset |

### 4. Validação da Fase 2

| Teste | O que valida |
|---|---|
| `test_mr_known_primes` | Miller-Rabin retorna 1 ou 2 para primos conhecidos (ex: primos de Mersenne pequenos) |
| `test_mr_known_composites` | Retorna 0 para compostos conhecidos |
| `test_mr_formula_small` | Para a=10, b=5, c=3, k=1, j=1: N é primo? Verificar com tabela de primos |
| `test_mr_vs_trial_div` | Para N pequeno: MR concorda com trial division completo |

### 5. Validação end-to-end com valores pequenos

Usar `a=15, b=8, c=3`, varrer todos os (k,j):
1. Calcular N para cada par com GMP
2. Verificar divisibilidade por todos os primos até 10^6 manualmente (ground truth)
3. Rodar fase 0 + fase 1 + fase 2 e comparar com ground truth
4. Verificar que os arquivos `phase1_survivors.txt` e `phase2_primes.txt` contêm exatamente o esperado

### 6. Testes de distribuição MPI

| Teste | O que valida |
|---|---|
| `test_mpi_register` | Todos os ranks registram corretamente como GPU ou CPU |
| `test_mpi_task_types` | GPU workers só recebem `gpu_task_t`, CPU workers só recebem `cpu_task_t` |
| `test_mpi_gpu_steal` | GPU worker ocioso recebe `cpu_task_t` quando `gpu_queue` está vazia |
| `test_mpi_termination` | Sentinel `-1` chega a todos os workers e eles terminam limpo |
| `test_mpi_no_duplicate_tasks` | Mesmo (a,b,c) nunca é enviado duas vezes |
| `test_mpi_output_files` | Arquivo final contém exatamente os resultados esperados do teste end-to-end |

### 7. Testes de stress / regressão

| Teste | O que valida |
|---|---|
| `test_stress_phase1` | Roda 1000 (a,b,c) aleatórios e verifica que GPU não produz resultados absurdos |
| `test_stress_mr` | 1000 chamadas paralelas de Miller-Rabin sem crash/race condition |
| `test_mpi_2_workers` | `mpirun -np 3` (1 master + 1 GPU + 1 CPU) termina corretamente |
| `test_mpi_no_gpu` | Roda sem GPU disponível (fallback CPU para fase 1) |

---

## Ordem de Implementação

1. `messages.h`
2. `gen_primes.c` + `prime_list.h/c`
3. `sieve_table.h/c` + `sieve.h/c`
4. `trial_div.h/cu`
5. `miller_rabin.h/c`
6. `worker.h/c`
7. `master.h/c`
8. `main.c` + `CMakeLists.txt`
9. Testes unitários (passos 2–5)
10. Teste end-to-end + MPI
11. `mpirun -np 4 ...` no cluster
