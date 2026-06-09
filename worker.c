#include "worker.h"
#include "messages.h"
#include "trial_div.h"
#include "trial_div_cpu.h"
#include "miller_rabin.h"

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

static int world_rank;

static void do_gpu_task(const prime_list_t *gpu_primes,
                        const prime_list_t *cpu_p1_primes,
                        int use_gpu,
                        const gpu_task_t *task,
                        gpu_result_t *result)
{
    result->a = task->a;
    result->b = task->b;
    result->c = task->c;
    result->n_survivors = 0;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    fprintf(stderr, "[rank %d] Phase 1 - Starting %s task a=%lld b=%lld c=%lld\n",
            world_rank, use_gpu ? "GPU" : "CPU", task->a, task->b, task->c);

    /* sieve_bits pre-computed by master (Phase 0) */
    uint32_t bits[KJ_WORDS];
    memcpy(bits, task->sieve_bits, sizeof(bits));

    gpu_timing_t gtiming = {0};
    if (!sieve_all_eliminated(bits)) {
        if (use_gpu) {
            if (trial_div_gpu(gpu_primes, task->a, task->b, task->c, bits, &gtiming) != 0)
                fprintf(stderr, "[rank %d] GPU trial division failed\n", world_rank);
        } else {
            trial_div_cpu(cpu_p1_primes, task->a, task->b, task->c, bits);
        }
    }

    for (int k = 1; k <= 100; k++) {
        for (int j = 1; j <= 100; j++) {
            int bit = (k-1)*100 + (j-1);
            if (!(bits[bit >> 5] & (1u << (bit & 31)))) {
                result->ks[result->n_survivors] = (uint8_t)k;
                result->js[result->n_survivors] = (uint8_t)j;
                result->n_survivors++;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed1 = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    fprintf(stderr, "[rank %d] End Phase 1 - %s a=%lld b=%lld c=%lld survivors=%d time=%.3fs\n",
            world_rank, use_gpu ? "GPU" : "CPU",
            task->a, task->b, task->c, result->n_survivors, elapsed1);
    if (use_gpu && (gtiming.upload + gtiming.kernel + gtiming.download) > 0.0) {
        fprintf(stderr,
                "  ├─ upload  (H→D): %8.3fs\n"
                "  ├─ kernel       : %8.3fs\n"
                "  └─ download(D→H): %8.3fs\n",
                gtiming.upload, gtiming.kernel, gtiming.download);
    }
}

static void do_cpu_task(const cpu_task_t *task, cpu_result_t *result)
{
    result->a = task->a;
    result->b = task->b;
    result->c = task->c;
    result->k = task->k;
    result->j = task->j;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    fprintf(stderr, "[rank %d] Phase 2 - Starting CPU task a=%lld b=%lld c=%lld k=%d j=%d\n",
            world_rank, task->a, task->b, task->c, task->k, task->j);

    mr_timing_t mrt = {0};
    result->result = (uint8_t)miller_rabin_test(task->a, task->b, task->c, task->k, task->j, &mrt);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    fprintf(stderr, "[rank %d] End Phase 2 - CPU a=%lld b=%lld c=%lld k=%d j=%d result=%d time=%.3fs\n",
            world_rank, task->a, task->b, task->c, task->k, task->j, result->result, elapsed);
    if (result->result == 0) {
        fprintf(stderr,
                "  ├─ build N      : %8.6fs\n"
                "  └─ Miller-Rabin N: %7.3fs  → composite\n",
                mrt.build_n, mrt.mr_n);
    } else {
        fprintf(stderr,
                "  ├─ build N      : %8.6fs\n"
                "  ├─ Miller-Rabin N: %7.3fs  → probable prime\n"
                "  ├─ build rev(N) : %8.6fs\n"
                "  └─ Miller-Rabin R: %7.3fs  → %s\n",
                mrt.build_n, mrt.mr_n, mrt.build_revn, mrt.mr_revn,
                (result->result == 2) ? "probable prime ★" : "composite");
    }
}

void worker_run(const prime_list_t *gpu_primes,
                const prime_list_t *cpu_p1_primes,
                int is_gpu_worker,
                int gpu_index)
{
    (void)gpu_index;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    msg_register_t reg;
    reg.type      = is_gpu_worker ? WORKER_GPU : WORKER_CPU;
    reg.gpu_index = gpu_index;
    {
        int hlen;
        MPI_Get_processor_name(reg.hostname, &hlen);
        reg.hostname[sizeof(reg.hostname)-1] = '\0';
    }
    MPI_Send(&reg, sizeof(reg), MPI_BYTE, 0, TAG_REGISTER, MPI_COMM_WORLD);

    for (;;) {
        MPI_Status status;
        char buf[sizeof(gpu_result_t) > sizeof(cpu_task_t)
                 ? sizeof(gpu_result_t) : sizeof(cpu_task_t)];

        MPI_Recv(buf, sizeof(buf), MPI_BYTE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == TAG_TERMINATE) break;

        if (status.MPI_TAG == TAG_GPU_TASK) {
            gpu_task_t   task;
            gpu_result_t result;
            memcpy(&task, buf, sizeof(task));
            /* GPU workers use CUDA; CPU workers use CPU fallback */
            do_gpu_task(gpu_primes, cpu_p1_primes, is_gpu_worker, &task, &result);
            MPI_Send(&result, sizeof(result), MPI_BYTE, 0, TAG_GPU_RESULT, MPI_COMM_WORLD);
        } else if (status.MPI_TAG == TAG_CPU_TASK) {
            cpu_task_t   task;
            cpu_result_t result;
            memcpy(&task, buf, sizeof(task));
            do_cpu_task(&task, &result);
            MPI_Send(&result, sizeof(result), MPI_BYTE, 0, TAG_CPU_RESULT, MPI_COMM_WORLD);
        }
    }
}
