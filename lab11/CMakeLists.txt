cmake_minimum_required(VERSION 4.0)
project(lab11 CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")


add_executable(lab11 main.cu)

set_target_properties(lab11 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
