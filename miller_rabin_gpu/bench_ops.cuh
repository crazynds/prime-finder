#pragma once
// Executa benchmark de operações Montgomery GPU vs GMP e imprime tabela.
// Ativado com --bench-ops (até 65536 bits) ou --bench-ops-long (até 131072 bits).
void run_bench_ops(bool long_run = false);
