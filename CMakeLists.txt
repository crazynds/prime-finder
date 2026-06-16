cmake_minimum_required(VERSION 3.18)

# ── Parâmetros do usuário ─────────────────────────────────────────────────────
include(${CMAKE_CURRENT_SOURCE_DIR}/params.cmake)

# CUDA 11.x supports GCC <= 11. If the system default is newer, try gcc-11.
find_program(GCC11 gcc-11)
if(GCC11)
    set(CMAKE_CUDA_HOST_COMPILER "${GCC11}")
    set(CMAKE_C_COMPILER         "${GCC11}")
    find_program(GPP11 g++-11)
    if(GPP11)
        set(CMAKE_CXX_COMPILER "${GPP11}")
    endif()
    message(STATUS "CUDA host compiler forced to ${GCC11} (system GCC may be > 11)")
else()
    # gcc-11 not found — allow nvcc to use the unsupported version
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
    message(STATUS "gcc-11 not found; using -allow-unsupported-compiler")
endif()

project(PrimeHunter LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

# ── Flags de otimização ───────────────────────────────────────────────────────
# Host (GCC): O3 + arquitetura nativa da CPU
set(CMAKE_C_FLAGS_RELEASE   "-O3 -march=native -funroll-loops")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops")
# Garante O3 mesmo sem -DCMAKE_BUILD_TYPE=Release explícito
add_compile_options(
    $<$<COMPILE_LANGUAGE:C>:-O3>
    $<$<COMPILE_LANGUAGE:C>:-march=native>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-march=native>
)
# NVCC: O3 + fast math + detecção automática de arquitetura via CMAKE_CUDA_ARCHITECTURES
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# Limite de registradores por thread (params.cmake: MR_MAXRREGCOUNT, 0 = sem limite)
if(DEFINED MR_MAXRREGCOUNT AND NOT MR_MAXRREGCOUNT EQUAL 0)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -maxrregcount=${MR_MAXRREGCOUNT}")
    message(STATUS "CUDA -maxrregcount=${MR_MAXRREGCOUNT}")
endif()

# ----- Dependencies -----
find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

# GMP
find_library(GMP_LIB gmp REQUIRED)
find_path(GMP_INCLUDE gmp.h REQUIRED)

# ----- gen_primes standalone tool -----
add_executable(gen_primes gen_primes.c)
target_link_libraries(gen_primes m)

# ----- prime_hunter main binary -----
set(COMMON_SOURCES
    prime_list.c
    sieve_table.c
    sieve.c
    miller_rabin.c
    trial_div_cpu.c
    worker.c
    master.c
    main.c
)

add_executable(prime_hunter
    ${COMMON_SOURCES}
    trial_div.cu
)

target_include_directories(prime_hunter PRIVATE
    ${MPI_C_INCLUDE_DIRS}
    ${GMP_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(prime_hunter
    MPI::MPI_C
    CUDA::cudart
    ${GMP_LIB}
    m
)

# CUDA device linking
set_target_properties(prime_hunter PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ----- unified test + bench runner -----
enable_testing()

add_executable(prime_tests
    tests/runner.cu
    tests/test_main.c
    tests/test_math.c
    tests/test_sieve.c
    tests/test_phase2.c
    tests/test_gpu.c
    tests/bench_miller_rabin.c
    tests/bench_gpu.cu
    prime_list.c
    sieve_table.c
    sieve.c
    miller_rabin.c
    trial_div.cu
)

target_include_directories(prime_tests PRIVATE
    ${GMP_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(prime_tests
    CUDA::cudart
    ${GMP_LIB}
    m
)

set_target_properties(prime_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_test(NAME unit_tests COMMAND prime_tests --test)
add_test(NAME gpu_tests  COMMAND prime_tests --test)

# ── Miller-Rabin GPU (bench_mr_gpu) ───────────────────────────────────────────

# Algoritmo de multiplicação (params.cmake → MUL_ALG): compila apenas o .cu da
# classe selecionada. SCHOOLBOOK e NTT_MERGE usam a classe merge; só NTT_4STEP
# troca para ntt_4step.cu. carry_norm.cu é agnóstico (define Multiplier::).
if(MUL_ALG STREQUAL "MUL_NTT_4STEP")
    set(MUL_BACKEND_SRC miller_rabin_gpu/ops/mul/ntt_4step.cu)
else()
    set(MUL_BACKEND_SRC miller_rabin_gpu/ops/mul/ntt_merge.cu)
endif()
message(STATUS "MUL_ALG = ${MUL_ALG} → ${MUL_BACKEND_SRC}")

# Gera config.h a partir do template + params.cmake
if(MR_ADVANCED_MONITOR)
    set(MR_ADVANCED_MONITOR 1)
else()
    unset(MR_ADVANCED_MONITOR)
endif()
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/miller_rabin_gpu/config.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/miller_rabin_gpu/config.h
)

include(FetchContent)
FetchContent_Declare(
    GPUNTT
    GIT_REPOSITORY https://github.com/Alisah-Ozcan/GPU-NTT.git
    GIT_TAG        main
    PATCH_COMMAND  git apply --ignore-whitespace
                   ${CMAKE_CURRENT_SOURCE_DIR}/patches/gpuntt_remove_cc89.patch
                   || true
    UPDATE_DISCONNECTED TRUE
)
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/gpuntt_install" CACHE PATH "" FORCE)
FetchContent_MakeAvailable(GPUNTT)

if(NOT TARGET GPUNTT::ntt)
    find_package(GPUNTT REQUIRED
        PATHS "${CMAKE_BINARY_DIR}/gpuntt_install/lib/cmake/GPUNTT-1.0"
        NO_DEFAULT_PATH)
endif()

add_executable(bench_mr_gpu
    miller_rabin_gpu/bench_mr_gpu.cu
    miller_rabin_gpu/helpers/bench_ops.cu
    miller_rabin_gpu/miller_rabin_runner.cu
    ${MUL_BACKEND_SRC}
    miller_rabin_gpu/ops/carry/carry_norm.cu
    miller_rabin_gpu/ops/shift/shift.cu
    miller_rabin_gpu/ops/sub/sub.cu
    miller_rabin_gpu/batch_mod_ctx.cu
    miller_rabin_gpu/helpers/mod_perf.cu
    miller_rabin_gpu/reductions/montgomery.cu
    miller_rabin_gpu/reductions/barrett.cu
)

target_include_directories(bench_mr_gpu PRIVATE
    ${GMP_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}/miller_rabin_gpu
)

target_link_libraries(bench_mr_gpu PRIVATE
    GPUNTT::ntt
    CUDA::cudart
    ${GMP_LIB}
)

# Repassa flags GPU-NTT de params.cmake para o compilador
if(GPUNTT_CC89)
    target_compile_definitions(bench_mr_gpu PRIVATE CC_89)
endif()
target_compile_definitions(bench_mr_gpu PRIVATE
    GPUNTT_NTT_LAYOUT=${GPUNTT_NTT_LAYOUT}
)


set_target_properties(bench_mr_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

