cmake_minimum_required(VERSION 4.0)
project(lab7 CUDA)

set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")

set(CMAKE_CUDA_STANDARD 20)

add_executable(lab7 main.cu)

set_target_properties(lab7 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
