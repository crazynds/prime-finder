#pragma once
// helpers/timers.cuh — macros internas compartilhadas pelos .cu do contexto modular
// (batch_mod_ctx.cu, reductions/*.cu). NÃO incluir no header público: estas macros
// referenciam membros de BatchModCtx (timer, perf_enabled) e a variável de stream `s`.

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

// Marco de início da próxima seção (sem sync). No-op se perf_enabled == false.
#define TSTART()          \
    do                    \
    {                     \
        if (perf_enabled) \
            timer.start(s); \
    } while (0)

// Marco de fim acumulando no PerfNode* `node` (sem sync). timer.flush() sincroniza
// uma única vez no fim da função pública. No-op se perf_enabled == false.
#define TSTOP(node)            \
    do                         \
    {                          \
        if (perf_enabled)      \
            timer.stop((node), s); \
    } while (0)
