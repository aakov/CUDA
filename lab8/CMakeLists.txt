cmake_minimum_required(VERSION 4.0)
project(lab8 CUDA)

set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe")
set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")

set(CMAKE_CUDA_STANDARD 20)

add_executable(lab8 main.cu)

set_target_properties(lab8 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
