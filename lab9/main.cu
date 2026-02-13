#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
using namespace std;

__global__ void reduceKernel(int* data, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && idx + stride < n) {
            data[idx] += data[idx + stride];
        }
        __syncthreads();
    }
}

__global__ void sequentialReduce(int* data, int n, int* result) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    *result = sum;
}

int main() {
    int N = 100000;
    int BLOCK_SIZE = 1024;

    int* h_data = new int[N];
    for (int i = 0; i < N; i++) {
        h_data[i] = 1;
    }

    int *d_data, *d_result;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));

    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    auto start_parallel = std::chrono::high_resolution_clock::now();

    int activeElements = N;
    while (activeElements > 1) {
        int blocks = (activeElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        reduceKernel<<<blocks, BLOCK_SIZE>>>(d_data, activeElements);
        cudaDeviceSynchronize();
        activeElements = blocks * (BLOCK_SIZE / 2);
        if (activeElements < 1) activeElements = 1;
    }

    auto end_parallel = std::chrono::high_resolution_clock::now();

    int result_parallel;
    cudaMemcpy(&result_parallel, d_data, sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    auto start_seq = std::chrono::high_resolution_clock::now();
    sequentialReduce<<<1, 1>>>(d_data, N, d_result);
    cudaDeviceSynchronize();
    auto end_seq = std::chrono::high_resolution_clock::now();

    int result_seq;
    cudaMemcpy(&result_seq, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    auto time_parallel = std::chrono::duration_cast<std::chrono::microseconds>(end_parallel - start_parallel).count();
    auto time_seq = std::chrono::duration_cast<std::chrono::microseconds>(end_seq - start_seq).count();

    cout << "Parallel result: " << result_parallel << " (time: " << time_parallel << " us)" << endl;
    cout << "Sequential result: " << result_seq << " (time: " << time_seq << " us)" << endl;
    cout << "Speedup: " << (double)time_seq / time_parallel << "x" << endl;

    delete[] h_data;
    cudaFree(d_data);
    cudaFree(d_result);

    return 0;
}
