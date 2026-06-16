// perf/perf_timer.cuh — Cronômetro GPU baseado em ring de eventos CUDA.
//
// start()/stop(node) gravam marcos no stream SEM sincronizar; flush() sincroniza
// UMA vez e acumula cada intervalo no PerfNode correspondente. Mesmo esquema do
// antigo TSTART/TSTOP, mas o alvo é um PerfNode* (grafo dinâmico) em vez de um
// campo fixo de struct.
#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "perf/perf_node.cuh"

class PerfTimer
{
public:
    static constexpr int RING = 64; // marcos por chamada pública (folga p/ pipelines longos)

    void init()
    {
        for (int i = 0; i <= RING; i++)
            check(cudaEventCreate(&ev_[i]));
    }
    void destroy()
    {
        for (int i = 0; i <= RING; i++)
            if (ev_[i])
                cudaEventDestroy(ev_[i]);
    }

    // Marco de início da próxima seção. O gate de perf vive na macro TSTART
    // (checa BatchModCtx::perf_enabled); aqui apenas gravamos o evento.
    void start(cudaStream_t s)
    {
        check(cudaEventRecord(ev_[cur_], s));
    }
    // Marco de fim; registra o nó alvo. Sem sync.
    void stop(PerfNode *node, cudaStream_t s)
    {
        check(cudaEventRecord(ev_[cur_ + 1], s));
        acc_[cur_] = node;
        node->calls++;
        cur_++;
    }
    // Sincroniza o último marco e acumula todos os intervalos pendentes.
    // Seguro chamar sempre: se nada foi gravado (cur_ == 0), é no-op.
    void flush(cudaStream_t)
    {
        if (cur_ == 0)
            return;
        check(cudaEventSynchronize(ev_[cur_]));
        for (int i = 0; i < cur_; i++)
        {
            float ms = 0;
            check(cudaEventElapsedTime(&ms, ev_[i], ev_[i + 1]));
            acc_[i]->ms += ms;
        }
        cur_ = 0;
    }

private:
    static void check(cudaError_t e)
    {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("[perf] CUDA: ") + cudaGetErrorString(e));
    }
    cudaEvent_t ev_[RING + 1] = {};
    PerfNode *acc_[RING] = {};
    int cur_ = 0;
};
