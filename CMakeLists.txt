cmake_minimum_required(VERSION 3.16)
project(stochastic_sde_montecarlo VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -fno-finite-math-only -funroll-loops -DNDEBUG -Wall -Wextra -pedantic")

include_directories(${PROJECT_SOURCE_DIR}/cpp_core/include)

add_library(stoch_core STATIC
    cpp_core/src/gmm_em.cpp
    cpp_core/src/barrier_mc.cpp
)

add_executable(main_benchmark
    cpp_core/src/main_benchmark.cpp
)

target_link_libraries(main_benchmark PRIVATE stoch_core m)
