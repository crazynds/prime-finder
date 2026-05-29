cmake_minimum_required(VERSION 3.18)
project(PrimeHunter LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
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

# ----- tests -----
enable_testing()

add_executable(test_suite
    tests/test_main.c
    tests/test_math.c
    tests/test_sieve.c
    tests/test_phase2.c
    prime_list.c
    sieve_table.c
    sieve.c
    miller_rabin.c
)

target_include_directories(test_suite PRIVATE
    ${GMP_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(test_suite
    ${GMP_LIB}
    m
)

add_test(NAME unit_tests COMMAND test_suite)

# ----- GPU test (needs CUDA) -----
add_executable(test_gpu
    tests/test_gpu.c
    prime_list.c
    sieve_table.c
    sieve.c
    trial_div.cu
)

target_include_directories(test_gpu PRIVATE
    ${GMP_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(test_gpu
    CUDA::cudart
    ${GMP_LIB}
    m
)

set_target_properties(test_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_test(NAME gpu_tests COMMAND test_gpu)

# ----- GPU benchmark -----
add_library(prime_list_obj OBJECT prime_list.c)

add_executable(bench_gpu
    bench_gpu.cu
    trial_div.cu
    $<TARGET_OBJECTS:prime_list_obj>
)

target_include_directories(bench_gpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(bench_gpu
    CUDA::cudart
    m
)

set_target_properties(bench_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
