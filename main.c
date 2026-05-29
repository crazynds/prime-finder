#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "messages.h"
#include "prime_list.h"
#include "sieve_table.h"
#include "worker.h"
#include "master.h"

/* Try to detect CUDA device count without hard-linking CUDA at compile time
 * when no GPU is available. The actual CUDA call is in trial_div.cu. */
#include "trial_div.h"

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s <small_primes.bin> <gpu_primes.bin> <a_min> <a_max> [c_min] [--cpu-phase2-only]\n"
        "  small_primes.bin   : primes for sieve table (phase 0)\n"
        "  gpu_primes.bin     : primes for GPU trial division (phase 1)\n"
        "  a_min, a_max       : exponent range (>= 50000)\n"
        "  c_min              : minimum c (default 100)\n"
        "  --cpu-phase2-only  : CPU workers only run Phase 2 (Miller-Rabin), never steal Phase 1\n",
        prog);
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 5) {
        if (world_rank == 0) usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *small_path    = argv[1];
    const char *gpu_path      = argv[2];
    long long   a_min         = atoll(argv[3]);
    long long   a_max         = atoll(argv[4]);
    long long   c_min         = 100;
    int         cpu_p2_only   = 0;

    for (int i = 5; i < argc; i++) {
        if (strcmp(argv[i], "--cpu-phase2-only") == 0)
            cpu_p2_only = 1;
        else
            c_min = atoll(argv[i]);
    }

    /* All ranks participate: rank 0 broadcasts its hostname so workers can
     * detect co-location and adjust their GPU index accordingly. */
    char my_host[MPI_MAX_PROCESSOR_NAME];
    char master_host[MPI_MAX_PROCESSOR_NAME];
    {
        int hlen;
        MPI_Get_processor_name(my_host, &hlen);
        if (world_rank == 0)
            strncpy(master_host, my_host, MPI_MAX_PROCESSOR_NAME);
        MPI_Bcast(master_host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    /* MPI_Comm_split_type is collective — ALL ranks must call it */
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                        world_rank, MPI_INFO_NULL, &node_comm);

    if (world_rank == 0) {
        MPI_Comm_free(&node_comm);  /* master doesn't use node_comm */
        fprintf(stderr, "[master] world_size=%d a=[%lld,%lld] c_min=%lld cpu_phase2_only=%d\n",
                world_size, a_min, a_max, c_min, cpu_p2_only);
        master_run(a_min, a_max, c_min, world_size - 1, cpu_p2_only);
    } else {
        int local_rank = 0;
        const char *lr = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
        if (lr) local_rank = atoi(lr);

        /* If master is co-located on this node it takes local_rank=0;
         * subtract 1 so the first worker gets GPU index 0. */
        int master_colocated = (strncmp(my_host, master_host, MPI_MAX_PROCESSOR_NAME) == 0);
        int gpu_index = master_colocated ? (local_rank - 1) : local_rank;

        int ngpus = cuda_get_device_count();
        int is_gpu = (gpu_index >= 0 && gpu_index < ngpus);
        if (is_gpu) cuda_set_device(gpu_index);

        fprintf(stderr, "[rank %d] local_rank=%d ngpus=%d gpu_index=%d role=%s\n",
                world_rank, local_rank, ngpus, gpu_index, is_gpu ? "GPU" : "CPU");

        /* Load prime lists */
        prime_list_t small_primes = {0};
        prime_list_t gpu_primes   = {0};

        fprintf(stderr, "[rank %d] carregando %s...\n", world_rank, small_path);
        if (prime_list_load(&small_primes, small_path) != 0) {
            fprintf(stderr, "[rank %d] failed to load %s\n", world_rank, small_path);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(stderr, "[rank %d] %s carregado: %u primos\n", world_rank, small_path, small_primes.count);

        /* All workers load gpu_primes — CPU workers use it for Phase 1 CPU fallback */
        fprintf(stderr, "[rank %d] carregando %s...\n", world_rank, gpu_path);
        if (prime_list_load(&gpu_primes, gpu_path) != 0) {
            fprintf(stderr, "[rank %d] failed to load %s\n", world_rank, gpu_path);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(stderr, "[rank %d] %s carregado: %u primos\n", world_rank, gpu_path, gpu_primes.count);

        /* First worker on the node = local_rank 1 if master is co-located,
         * or local_rank 0 otherwise. Only this worker prints progress. */
        int is_first_on_node = master_colocated ? (local_rank == 1) : (local_rank == 0);

        sieve_table_t st = {0};
        if (is_first_on_node) {
            size_t flat_size = sieve_table_flat_size(&small_primes);
            fprintf(stderr, "[rank %d] *** INICIANDO BUILD DA TABELA SIEVE (%.2f MB) ***\n",
                    world_rank, flat_size / (1024.0 * 1024.0));
        }
        if (sieve_table_build(&st, is_first_on_node, &small_primes) != 0) {
            fprintf(stderr, "[rank %d] sieve_table_build failed\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (is_first_on_node)
            fprintf(stderr, "[rank %d] sieve table pronto: %d entradas\n",
                    world_rank, st.n);

        worker_run(&gpu_primes, &gpu_primes, &st, is_gpu, gpu_index);

        sieve_table_free(&st);
        MPI_Comm_free(&node_comm);
        prime_list_free(&small_primes);
        if (is_gpu) prime_list_free(&gpu_primes);
    }

    MPI_Finalize();
    return 0;
}
