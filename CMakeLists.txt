cmake_minimum_required(VERSION 3.10.2)
project(cuda_examples LANGUAGES CUDA C)
set(CMAKE_CUDA_FLAGS "-w")
set(CMAKE_CUDA_STANDARD 14)
add_executable(cuda_examples ${PROJECT_SOURCE_DIR}/chapter_3/simpleDivergence.cu)
set_target_properties(
        cuda_examples
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)