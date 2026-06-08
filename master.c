#include "master.h"
#include "messages.h"
#include "prime_list.h"
#include "sieve_table.h"
#include "sieve.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_WORKERS      4096
#define MAX_PENDING_CPU  500000 /* individual (a,b,c,k,j) pairs (~12 MB: 500k × 24 B each) */
#define MAX_PENDING_GPU  1024   /* gpu tasks with sieve_bits (~1.3 MB: 1024 × ~1.3 KB each) */
#define MONITOR_INTERVAL_SEC 1

/* ------- simple circular queue ------- */
typedef struct {
    void  *data;
    int    elem_size;
    int    head, tail, cap;
} queue_t;

static int queue_init(queue_t *q, int cap, int elem_size) {
    q->data      = malloc((size_t)cap * elem_size);
    q->elem_size = elem_size;
    q->head = q->tail = 0;
    q->cap   = cap;
    return q->data ? 0 : -1;
}
static int  queue_empty(queue_t *q) { return q->head == q->tail; }
static int  queue_full(queue_t *q)  { return (q->tail + 1) % q->cap == q->head; }
static int  queue_size(queue_t *q)  { return (q->tail - q->head + q->cap) % q->cap; }
static void queue_push(queue_t *q, const void *item) {
    memcpy((char*)q->data + (size_t)q->tail * q->elem_size, item, q->elem_size);
    q->tail = (q->tail + 1) % q->cap;
}
static void queue_pop(queue_t *q, void *item) {
    memcpy(item, (char*)q->data + (size_t)q->head * q->elem_size, q->elem_size);
    q->head = (q->head + 1) % q->cap;
}

/* ------- task descriptions for monitor ------- */
#define TASK_IDLE     0
#define TASK_GPU      1
#define TASK_CPU      2

typedef struct {
    int       rank;
    int       type;        /* WORKER_GPU or WORKER_CPU */
    int       gpu_index;
    char      hostname[64];
    /* current task */
    int       task_type;   /* TASK_IDLE / TASK_GPU / TASK_CPU */
    long long ta, tb, tc;
    int       tk, tj;
    time_t    task_start;
} worker_info_t;

/* ------- abc iterator ------- */
typedef struct {
    long long a, b, c;
    long long a_min, a_max, c_min;
    int done;
} abc_iter_t;

static long long b_max_for_a(long long a) { return a * 2 / 3; }

static void abc_iter_init(abc_iter_t *it,
                          long long a_min, long long a_max, long long c_min)
{
    it->a_min = a_min; it->a_max = a_max; it->c_min = c_min;
    it->a = a_min;
    it->b = b_max_for_a(a_min);
    it->c = c_min;
    it->done = (a_min > a_max);
}

static int abc_iter_next(abc_iter_t *it, long long *a, long long *b, long long *c)
{
    if (it->done) return 0;
    *a = it->a; *b = it->b; *c = it->c;

    it->c++;
    if (it->c >= it->b) {
        it->c = it->c_min;
        it->b--;
        if (it->b <= it->c_min) {
            it->a++;
            if (it->a > it->a_max) { it->done = 1; return 1; }
            it->b = b_max_for_a(it->a);
        }
    }
    return 1;
}

/* ------- ring buffers for last-N results ------- */
#define HISTORY_SIZE 20

typedef struct {
    long long a, b, c;
    int n_survivors;
} p1_hist_t;

typedef struct {
    long long a, b, c;
    int n_pairs, n_pp, n_both;
} p2_hist_t;

typedef struct {
    long long a, b, c;
    int k, j;
} found_hist_t;

typedef struct {
    p1_hist_t    p1[HISTORY_SIZE];
    p2_hist_t    p2[HISTORY_SIZE];
    found_hist_t found[HISTORY_SIZE];
    int p1_head, p1_count;
    int p2_head, p2_count;
    int found_head, found_count;
    long long total_p1, total_p2, total_found;
} history_t;

static void hist_push_p1(history_t *h, long long a, long long b, long long c, int surv) {
    h->p1[h->p1_head] = (p1_hist_t){a, b, c, surv};
    h->p1_head = (h->p1_head + 1) % HISTORY_SIZE;
    if (h->p1_count < HISTORY_SIZE) h->p1_count++;
    h->total_p1++;
}
static void hist_push_p2(history_t *h, long long a, long long b, long long c, int pairs, int pp, int both) {
    h->p2[h->p2_head] = (p2_hist_t){a, b, c, pairs, pp, both};
    h->p2_head = (h->p2_head + 1) % HISTORY_SIZE;
    if (h->p2_count < HISTORY_SIZE) h->p2_count++;
    h->total_p2++;
}
static void hist_push_found(history_t *h, long long a, long long b, long long c, int k, int j) {
    h->found[h->found_head] = (found_hist_t){a, b, c, k, j};
    h->found_head = (h->found_head + 1) % HISTORY_SIZE;
    if (h->found_count < HISTORY_SIZE) h->found_count++;
    h->total_found++;
}

/* ------- monitor writer ------- */
static void write_monitor(const char *path,
                          worker_info_t *workers, int nw,
                          int phase1_q_size, int phase2_q_size,
                          long long a_min, long long a_max, long long c_min,
                          int iter_done,
                          const history_t *hist)
{
    FILE *f = fopen(path, "w");
    if (!f) return;

    time_t now = time(NULL);
    char tbuf[32];
    strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", localtime(&now));

    fprintf(f, "=== prime_hunter monitor  %s ===\n\n", tbuf);
    fprintf(f, "Search range: a=[%lld,%lld]  c_min=%lld  iterator=%s\n\n",
            a_min, a_max, c_min, iter_done ? "DONE" : "running");
    fprintf(f, "Queue status:  Phase 1 pending=%-6d  Phase 2 pending=%-6d\n",
            phase1_q_size, phase2_q_size);
    fprintf(f, "Totals:        Phase 1 done=%-8lld  Phase 2 done=%-8lld  Found=%-6lld\n\n",
            hist->total_p1, hist->total_p2, hist->total_found);

    /* ---- workers ---- */
    char seen[MAX_WORKERS][64];
    int  n_seen = 0;

    for (int i = 0; i < nw; i++) {
        const char *h = workers[i].hostname;
        int found = 0;
        for (int s = 0; s < n_seen; s++)
            if (strcmp(seen[s], h) == 0) { found = 1; break; }
        if (!found) strncpy(seen[n_seen++], h, 63);
    }

    for (int s = 0; s < n_seen; s++) {
        fprintf(f, "Machine: %s\n", seen[s]);
        fprintf(f, "  %-6s %-4s %-6s %-10s %-8s  task\n",
                "rank", "type", "device", "status", "elapsed");
        fprintf(f, "  %s\n", "------------------------------------------------------");

        for (int i = 0; i < nw; i++) {
            worker_info_t *w = &workers[i];
            if (strcmp(w->hostname, seen[s]) != 0) continue;

            const char *type_str = (w->type == WORKER_GPU) ? "GPU" : "CPU";
            char dev_str[16];
            if (w->type == WORKER_GPU)
                snprintf(dev_str, sizeof(dev_str), "gpu%d", w->gpu_index);
            else
                strcpy(dev_str, "-");

            char elapsed_str[16];
            if (w->task_type == TASK_IDLE) {
                strcpy(elapsed_str, "-");
            } else {
                long secs = (long)(now - w->task_start);
                snprintf(elapsed_str, sizeof(elapsed_str), "%lds", secs);
            }

            char task_str[128];
            if (w->task_type == TASK_IDLE) {
                strcpy(task_str, "idle");
            } else if (w->task_type == TASK_GPU) {
                snprintf(task_str, sizeof(task_str),
                         "Phase1  a=%lld b=%lld c=%lld",
                         w->ta, w->tb, w->tc);
            } else {
                snprintf(task_str, sizeof(task_str),
                         "Phase2  a=%lld b=%lld c=%lld  k=%d j=%d",
                         w->ta, w->tb, w->tc, w->tk, w->tj);
            }

            fprintf(f, "  %-6d %-4s %-6s %-10s %-8s  %s\n",
                    w->rank, type_str, dev_str,
                    w->task_type == TASK_IDLE ? "idle" : "busy",
                    elapsed_str, task_str);
        }
        fprintf(f, "\n");
    }

    /* ---- three-column history ---- */
#define C1W 44
#define C2W 54
#define C3W 40

    fprintf(f, "%-*s  %-*s  %-*s\n",
            C1W, "--- Phase 1 (last 20) ---",
            C2W, "--- Phase 2 (last 20) ---",
            C3W, "--- Candidates found ---");
    fprintf(f, "%-*s  %-*s  %-*s\n",
            C1W, "a  b  c  survivors",
            C2W, "a  b  c  pairs  pp  both",
            C3W, "a  b  c  k  j");

    int rows = HISTORY_SIZE;
    for (int r = 0; r < rows; r++) {
        char c1[C1W+1], c2[C2W+1], c3[C3W+1];
        c1[0] = c2[0] = c3[0] = '\0';

        /* newest first: index = (head - 1 - r + SIZE) % SIZE */
        int i1 = (hist->p1_head - 1 - r + HISTORY_SIZE) % HISTORY_SIZE;
        if (r < hist->p1_count) {
            const p1_hist_t *e = &hist->p1[i1];
            snprintf(c1, sizeof(c1), "a=%-6lld b=%-6lld c=%-6lld surv=%-5d",
                     e->a, e->b, e->c, e->n_survivors);
        }

        int i2 = (hist->p2_head - 1 - r + HISTORY_SIZE) % HISTORY_SIZE;
        if (r < hist->p2_count) {
            const p2_hist_t *e = &hist->p2[i2];
            snprintf(c2, sizeof(c2), "a=%-6lld b=%-6lld c=%-6lld p=%-5d pp=%-4d b=%-3d",
                     e->a, e->b, e->c, e->n_pairs, e->n_pp, e->n_both);
        }

        int i3 = (hist->found_head - 1 - r + HISTORY_SIZE) % HISTORY_SIZE;
        if (r < hist->found_count) {
            const found_hist_t *e = &hist->found[i3];
            snprintf(c3, sizeof(c3), "a=%-6lld b=%-6lld c=%-6lld k=%-3d j=%-3d",
                     e->a, e->b, e->c, e->k, e->j);
        }

        fprintf(f, "%-*s  %-*s  %-*s\n", C1W, c1, C2W, c2, C3W, c3);
    }

    fclose(f);
}

/* ------- checkpoint ------- */
#define CHECKPOINT_FILE       "checkpoint.txt"
#define CHECKPOINT_PHASE2_FILE "checkpoint_phase2.txt"
#define CHECKPOINT_INTERVAL_SEC 30

static void checkpoint_save(long long a_min, long long a_max, long long c_min,
                             long long last_a, long long last_b, long long last_c)
{
    FILE *f = fopen(CHECKPOINT_FILE, "w");
    if (!f) return;
    fprintf(f, "a_min=%lld a_max=%lld c_min=%lld\n", a_min, a_max, c_min);
    fprintf(f, "last=%lld %lld %lld\n", last_a, last_b, last_c);
    fclose(f);
}

/* Dump all items in phase2_q to disk without consuming the queue */
static void checkpoint_save_phase2(const queue_t *q)
{
    FILE *f = fopen(CHECKPOINT_PHASE2_FILE, "w");
    if (!f) return;
    int n = queue_size((queue_t *)q);
    fprintf(f, "n=%d\n", n);
    for (int i = 0; i < n; i++) {
        int idx = (q->head + i) % q->cap;
        const cpu_task_t *t = (const cpu_task_t *)((const char *)q->data + (size_t)idx * q->elem_size);
        fprintf(f, "%lld %lld %lld %d %d\n", t->a, t->b, t->c, (int)t->k, (int)t->j);
    }
    fclose(f);
}

/* Restore phase2 pending items into q; returns number of items loaded */
static int checkpoint_load_phase2(queue_t *q)
{
    FILE *f = fopen(CHECKPOINT_PHASE2_FILE, "r");
    if (!f) return 0;

    int n = 0;
    if (fscanf(f, "n=%d\n", &n) != 1) { fclose(f); return 0; }

    int loaded = 0;
    for (int i = 0; i < n; i++) {
        long long a, b, c;
        int k, j;
        if (fscanf(f, "%lld %lld %lld %d %d\n", &a, &b, &c, &k, &j) != 5) break;
        if (queue_full(q)) break;
        cpu_task_t t = { a, b, c, (uint8_t)k, (uint8_t)j };
        queue_push(q, &t);
        loaded++;
    }
    fclose(f);

    if (loaded > 0)
        fprintf(stderr, "[master] fase 2 restaurada: %d pares pendentes\n", loaded);
    return loaded;
}

/* Returns 1 if checkpoint loaded and valid, sets *ra/*rb/*rc to resume after that point */
static int checkpoint_load(long long a_min, long long a_max, long long c_min,
                            long long *ra, long long *rb, long long *rc)
{
    FILE *f = fopen(CHECKPOINT_FILE, "r");
    if (!f) return 0;

    long long fa_min, fa_max, fc_min, la, lb, lc;
    if (fscanf(f, "a_min=%lld a_max=%lld c_min=%lld\n", &fa_min, &fa_max, &fc_min) != 3 ||
        fscanf(f, "last=%lld %lld %lld\n", &la, &lb, &lc) != 3) {
        fclose(f); return 0;
    }
    fclose(f);

    if (fa_min != a_min || fa_max != a_max || fc_min != c_min) {
        fprintf(stderr, "[master] checkpoint ignorado: parametros diferentes\n");
        return 0;
    }

    *ra = la; *rb = lb; *rc = lc;
    return 1;
}

/* Fast-forward iterator past (skip_a, skip_b, skip_c) */
static void abc_iter_skip_to(abc_iter_t *it,
                              long long skip_a, long long skip_b, long long skip_c)
{
    long long a, b, c;
    while (!it->done) {
        if (it->a > skip_a) break;
        if (it->a == skip_a && it->b < skip_b) break;
        if (it->a == skip_a && it->b == skip_b && it->c > skip_c) break;
        abc_iter_next(it, &a, &b, &c);
    }
}

/* ------- send helpers ------- */
static worker_info_t *find_worker(worker_info_t *workers, int nw, int rank)
{
    for (int i = 0; i < nw; i++)
        if (workers[i].rank == rank) return &workers[i];
    return NULL;
}

static void send_gpu_task(int rank, long long a, long long b, long long c,
                          worker_info_t *workers, int nw)
{
    fprintf(stderr, "[master] → rank %d GPU  a=%lld b=%lld c=%lld\n", rank, a, b, c);
    gpu_task_t t = { a, b, c };
    MPI_Send(&t, sizeof(t), MPI_BYTE, rank, TAG_GPU_TASK, MPI_COMM_WORLD);

    worker_info_t *w = find_worker(workers, nw, rank);
    if (w) {
        w->task_type = TASK_GPU;
        w->ta = a; w->tb = b; w->tc = c;
        w->tk = 0; w->tj = 0;
        w->task_start = time(NULL);
    }
}

static void send_cpu_task(int rank, const cpu_task_t *t,
                          worker_info_t *workers, int nw)
{
    MPI_Send(t, sizeof(*t), MPI_BYTE, rank, TAG_CPU_TASK, MPI_COMM_WORLD);

    worker_info_t *w = find_worker(workers, nw, rank);
    if (w) {
        w->task_type = TASK_CPU;
        w->ta = t->a; w->tb = t->b; w->tc = t->c;
        w->tk = t->k; w->tj = t->j;
        w->task_start = time(NULL);
    }
}

static void send_terminate(int rank)
{
    char dummy = 0;
    MPI_Send(&dummy, 1, MPI_BYTE, rank, TAG_TERMINATE, MPI_COMM_WORLD);
}

/* ------- master main loop ------- */
void master_run(const char *small_primes_path,
                long long a_min, long long a_max, long long c_min, int nworkers,
                int cpu_phase2_only)
{
    FILE *f1 = fopen("phase1_survivors.txt", "a");
    FILE *f2 = fopen("phase2_primes.txt", "a");
    if (!f1 || !f2) { perror("fopen output"); return; }

    /* Master owns the sieve table — workers receive pre-applied bits in each task */
    prime_list_t small_primes = {0};
    if (prime_list_load(&small_primes, small_primes_path) != 0) {
        fprintf(stderr, "[master] failed to load %s\n", small_primes_path);
        return;
    }
    sieve_table_t st = {0};
    fprintf(stderr, "[master] building sieve table (%u primes)...\n", small_primes.count);
    if (sieve_table_build(&st, 0, &small_primes) != 0) {
        fprintf(stderr, "[master] sieve_table_build failed\n");
        prime_list_free(&small_primes);
        return;
    }
    fprintf(stderr, "[master] sieve table ready: %d entries\n", st.n);
    prime_list_free(&small_primes);

    worker_info_t workers[MAX_WORKERS];
    int n_registered = 0;

    history_t hist;
    memset(&hist, 0, sizeof(hist));

    queue_t phase1_q, phase2_q;
    queue_init(&phase1_q, MAX_PENDING_GPU, sizeof(gpu_task_t));
    queue_init(&phase2_q, MAX_PENDING_CPU, sizeof(cpu_task_t));

    abc_iter_t it;
    abc_iter_init(&it, a_min, a_max, c_min);

    /* Resume from checkpoint if available */
    {
        long long ck_a, ck_b, ck_c;
        if (checkpoint_load(a_min, a_max, c_min, &ck_a, &ck_b, &ck_c)) {
            fprintf(stderr, "[master] retomando do checkpoint: a=%lld b=%lld c=%lld\n",
                    ck_a, ck_b, ck_c);
            abc_iter_skip_to(&it, ck_a, ck_b, ck_c);
            checkpoint_load_phase2(&phase2_q);
        }
    }

    long long last_ckpt_a = -1, last_ckpt_b = -1, last_ckpt_c = -1;
    time_t last_checkpoint = 0;

    int gpu_free[MAX_WORKERS], n_gpu_free = 0;
    int cpu_free[MAX_WORKERS], n_cpu_free = 0;
    int terminated = 0;

    time_t last_monitor = 0;

    char rbuf[sizeof(cpu_result_t) > sizeof(gpu_result_t)
              ? sizeof(cpu_result_t) : sizeof(gpu_result_t)];

    while (terminated < nworkers) {
        /* Non-blocking check so monitor can update even when no messages arrive */
        MPI_Status status;
        int flag = 0;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);

        if (flag) {
            MPI_Recv(rbuf, sizeof(rbuf), MPI_BYTE, status.MPI_SOURCE, status.MPI_TAG,
                     MPI_COMM_WORLD, &status);
            int src = status.MPI_SOURCE;

            if (status.MPI_TAG == TAG_REGISTER) {
                msg_register_t *reg = (msg_register_t *)rbuf;
                worker_info_t *w = &workers[n_registered++];
                w->rank      = src;
                w->type      = reg->type;
                w->gpu_index = reg->gpu_index;
                strncpy(w->hostname, reg->hostname, sizeof(w->hostname)-1);
                w->task_type = TASK_IDLE;

                if (reg->type == WORKER_GPU)
                    fprintf(stderr, "[master] registered rank %d (GPU) host=%s gpu=%d\n",
                            src, reg->hostname, reg->gpu_index);
                else
                    fprintf(stderr, "[master] registered rank %d (CPU) host=%s\n",
                            src, reg->hostname);

                if (reg->type == WORKER_GPU) gpu_free[n_gpu_free++] = src;
                else                          cpu_free[n_cpu_free++] = src;

            } else if (status.MPI_TAG == TAG_GPU_RESULT) {
                gpu_result_t *res = (gpu_result_t *)rbuf;
                /*
                fprintf(stderr, "[master] rank %d done GPU a=%lld b=%lld c=%lld → %d survivors\n",
                        src, res->a, res->b, res->c, res->n_survivors);
                **/
                worker_info_t *w = find_worker(workers, n_registered, src);
                if (w) w->task_type = TASK_IDLE;

                hist_push_p1(&hist, res->a, res->b, res->c, res->n_survivors);
                last_ckpt_a = res->a; last_ckpt_b = res->b; last_ckpt_c = res->c;

                if (res->n_survivors > 0) {
                    for (int i = 0; i < res->n_survivors; i++)
                        fprintf(f1, "a=%lld b=%lld c=%lld k=%d j=%d\n",
                                res->a, res->b, res->c, res->ks[i], res->js[i]);
                    fflush(f1);

                    for (int i = 0; i < res->n_survivors; i++) {
                        if (queue_full(&phase2_q)) break;
                        cpu_task_t ct;
                        ct.a = res->a; ct.b = res->b; ct.c = res->c;
                        ct.k = res->ks[i];
                        ct.j = res->js[i];
                        queue_push(&phase2_q, &ct);
                    }
                }
                /* Return sender to correct free list based on worker type */
                {
                    int wtype = WORKER_GPU;
                    for (int i = 0; i < n_registered; i++)
                        if (workers[i].rank == src) { wtype = workers[i].type; break; }
                    if (wtype == WORKER_GPU) gpu_free[n_gpu_free++] = src;
                    else                      cpu_free[n_cpu_free++] = src;
                }

            } else if (status.MPI_TAG == TAG_CPU_RESULT) {
                cpu_result_t *res = (cpu_result_t *)rbuf;
                worker_info_t *w = find_worker(workers, n_registered, src);
                if (w) w->task_type = TASK_IDLE;

                if (res->result >= 1) {
                    int both = (res->result == 2) ? 1 : 0;
                    hist_push_p2(&hist, res->a, res->b, res->c, 1, 1, both);
                }

                if (res->result == 2) {
                    fprintf(f2, "a=%lld b=%lld c=%lld k=%d j=%d PROBABLE_PRIME_BOTH\n",
                            res->a, res->b, res->c, res->k, res->j);
                    hist_push_found(&hist, res->a, res->b, res->c, res->k, res->j);
                    fflush(f2);
                } else if (res->result == 3) {
                    fprintf(f2, "a=%lld b=%lld c=%lld k=%d j=%d PRIME\n",
                            res->a, res->b, res->c, res->k, res->j);
                    hist_push_found(&hist, res->a, res->b, res->c, res->k, res->j);
                    fflush(f2);
                }

                int wtype = WORKER_CPU;
                for (int i = 0; i < n_registered; i++)
                    if (workers[i].rank == src) { wtype = workers[i].type; break; }

                if (wtype == WORKER_GPU) gpu_free[n_gpu_free++] = src;
                else                      cpu_free[n_cpu_free++] = src;
            }
        }

        /* Refill phase1_q from iterator */
        if (!queue_full(&phase1_q) && !it.done) {
            long long a, b, c;
            while (!queue_full(&phase1_q) && abc_iter_next(&it, &a, &b, &c)) {
                gpu_task_t t;
                t.a = a; t.b = b; t.c = c;
                memset(t.sieve_bits, 0, sizeof(t.sieve_bits));
                sieve_apply(&st, a, b, c, t.sieve_bits);
                queue_push(&phase1_q, &t);
            }
        }

        /* --- GPU workers: prefer Phase 1, steal Phase 2 if Phase 1 empty --- */
        while (n_gpu_free > 0) {
            if (!queue_empty(&phase1_q)) {
                gpu_task_t t;
                queue_pop(&phase1_q, &t);
                send_gpu_task(gpu_free[--n_gpu_free], t.a, t.b, t.c, workers, n_registered);
            } else if (!queue_empty(&phase2_q)) {
                cpu_task_t t;
                queue_pop(&phase2_q, &t);
                send_cpu_task(gpu_free[--n_gpu_free], &t, workers, n_registered);
            } else {
                break;
            }
        }

        /* --- CPU workers: prefer Phase 2, steal Phase 1 if Phase 2 empty (unless restricted) --- */
        while (n_cpu_free > 0) {
            if (!queue_empty(&phase2_q)) {
                cpu_task_t t;
                queue_pop(&phase2_q, &t);
                send_cpu_task(cpu_free[--n_cpu_free], &t, workers, n_registered);
            } else if (!cpu_phase2_only && !queue_empty(&phase1_q)) {
                gpu_task_t t;
                queue_pop(&phase1_q, &t);
                send_gpu_task(cpu_free[--n_cpu_free], t.a, t.b, t.c, workers, n_registered);
            } else {
                break;
            }
        }

        /* Terminate when all work done */
        if (it.done && queue_empty(&phase1_q) && queue_empty(&phase2_q)) {
            while (n_gpu_free > 0) {
                send_terminate(gpu_free[--n_gpu_free]);
                terminated++;
            }
            while (n_cpu_free > 0) {
                send_terminate(cpu_free[--n_cpu_free]);
                terminated++;
            }
        }

        /* Write checkpoint every 30 seconds */
        time_t now = time(NULL);
        if (last_ckpt_a >= 0 && now - last_checkpoint >= CHECKPOINT_INTERVAL_SEC) {
            checkpoint_save(a_min, a_max, c_min, last_ckpt_a, last_ckpt_b, last_ckpt_c);
            checkpoint_save_phase2(&phase2_q);
            last_checkpoint = now;
        }

        /* Write monitor every second */
        if (now - last_monitor >= MONITOR_INTERVAL_SEC) {
            write_monitor("monitor.txt",
                          workers, n_registered,
                          queue_size(&phase1_q), queue_size(&phase2_q),
                          a_min, a_max, c_min, it.done, &hist);
            last_monitor = now;
        }

        /* Brief sleep to avoid busy-waiting when no messages */
        if (!flag) {
            struct timespec ts = { 0, 10000000 };  /* 10ms */
            nanosleep(&ts, NULL);
        }
    }

    /* Final monitor update */
    write_monitor("monitor.txt", workers, n_registered, 0, 0,
                  a_min, a_max, c_min, 1, &hist);

    sieve_table_free(&st);
    free(phase1_q.data);
    free(phase2_q.data);
    fclose(f1);
    fclose(f2);
}
