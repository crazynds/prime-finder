#pragma once

/*
 * Entry point for rank 0 (master).
 *
 * a_min/a_max: range of exponent a to search
 * c_min:       minimum value for c (>= 100)
 * nworkers:    total MPI world size - 1
 */
/* cpu_phase2_only: if non-zero, CPU workers only run Phase 2 (never steal Phase 1) */
void master_run(long long a_min, long long a_max, long long c_min, int nworkers,
                int cpu_phase2_only);
