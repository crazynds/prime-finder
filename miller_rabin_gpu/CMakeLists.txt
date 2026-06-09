cmake_minimum_required(VERSION 3.18)
project(bench_mr_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

include(FetchContent)

FetchContent_Declare(
    GPUNTT
    GIT_REPOSITORY https://github.com/Alisah-Ozcan/GPU-NTT.git
    GIT_TAG        main
)

# GPU-NTT usa cmake com install, entao adicionamos como subdirectory
# e apontamos o install para dentro do build.
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/gpuntt_install" CACHE PATH "" FORCE)
FetchContent_MakeAvailable(GPUNTT)

# Apos FetchContent o target GPUNTT::ntt deve estar disponivel.
# Se o projeto nao exportar o target direto, incluimos o diretorio de headers.
if(NOT TARGET GPUNTT::ntt)
    find_package(GPUNTT REQUIRED
        PATHS "${CMAKE_BINARY_DIR}/gpuntt_install/lib/cmake/GPUNTT-1.0"
        NO_DEFAULT_PATH)
endif()

find_package(CUDAToolkit REQUIRED)

find_library(GMP_LIB gmp REQUIRED)
find_path(GMP_INCLUDE gmp.h REQUIRED)

add_executable(bench_mr_gpu
    bench_mr_gpu.cu
    bigint_ntt.cu
    montgomery.cu
)

target_include_directories(bench_mr_gpu PRIVATE
    ${GMP_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(bench_mr_gpu PRIVATE
    GPUNTT::ntt
    CUDA::cudart
    ${GMP_LIB}
)

target_compile_options(bench_mr_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_89,code=sm_89>
)

# Modo de teste: cmake -DTEST_MODE=ON ..  (ou  cmake -DTEST_MODE=OFF ..)
option(TEST_MODE "Habilita testes de corretude contra GMP" OFF)
if(TEST_MODE)
    message(STATUS "TEST_MODE habilitado")
    target_compile_definitions(bench_mr_gpu PRIVATE TEST_MODE)
endif()

set_target_properties(bench_mr_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
