cmake_minimum_required(VERSION 4.0)
project(lab9 CUDA)


set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
set(CMAKE_CUDA_STANDARD 20)

add_executable(lab9 main.cu)

set_target_properties(lab9 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
