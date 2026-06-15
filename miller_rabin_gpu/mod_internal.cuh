#pragma once
// mod_internal.cuh — macros internas compartilhadas pelos .cu do contexto modular
// (batch_mod_ctx.cu, reduce_montgomery.cu, reduce_barrett.cu). NÃO incluir no header
// público: estas macros referenciam membros de BatchModCtx (perf_enabled, ev_ring,
// ring_cur, acc_ring) e a variável de stream `s` no escopo da função-membro.

#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

// Checagem de erro CUDA → exceção.
#define CU(expr)                                                                                  \
    do                                                                                            \
    {                                                                                             \
        cudaError_t _e = (expr);                                                                  \
        if (_e != cudaSuccess)                                                                    \
            throw std::runtime_error(std::string("[CUDA] " #expr ": ") + cudaGetErrorString(_e)); \
    } while (0)

// Grava o marco de início da seção no ring (sem sync). No-op se perf_enabled == false.
#define TSTART()                                       \
    do                                                 \
    {                                                  \
        if (perf_enabled)                              \
            CU(cudaEventRecord(ev_ring[ring_cur], s)); \
    } while (0)

// Grava o marco de fim, registra o acumulador — sem sync. perf_flush() sincroniza
// uma única vez no final da função pública. No-op se perf_enabled == false.
#define TSTOP(stage)                                       \
    do                                                     \
    {                                                      \
        if (perf_enabled)                                  \
        {                                                  \
            CU(cudaEventRecord(ev_ring[ring_cur + 1], s)); \
            acc_ring[ring_cur] = &((stage).ms);            \
            (stage).calls++;                               \
            ring_cur++;                                    \
        }                                                  \
    } while (0)
