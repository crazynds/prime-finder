cmake_minimum_required(VERSION 3.18)

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

include(FetchContent)
FetchContent_Declare(
    GPUNTT
    GIT_REPOSITORY https://github.com/Alisah-Ozcan/GPU-NTT.git
    GIT_TAG        main
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
    miller_rabin_gpu/bench_ops.cu
    miller_rabin_gpu/miller_rabin_runner.cu
    miller_rabin_gpu/bigint_ntt.cu
    miller_rabin_gpu/montgomery.cu
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

target_compile_options(bench_mr_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_89,code=sm_89>
)

set_target_properties(bench_mr_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

