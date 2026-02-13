cmake_minimum_required(VERSION 4.0)
project(CUDA CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")


add_executable(CUDA main.cu)

set_target_properties(CUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
