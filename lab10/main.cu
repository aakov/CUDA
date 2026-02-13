#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void squareKernel(int* input, int* output, int n) {
    __shared__ int sharedMem[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //kopiowanie do shared memory
    if (idx < n) {
        sharedMem[threadIdx.x] = input[idx];
    }

    __syncthreads();

    if (idx < n) {
        sharedMem[threadIdx.x] = sharedMem[threadIdx.x] * sharedMem[threadIdx.x];
    }

    __syncthreads();

    if (idx < n) {
        output[idx] = sharedMem[threadIdx.x];
    }
}

int main() {
    int N = 102400;
    int size = N * sizeof(int);

    int* tablica_input = new int[N];
    int* tablica_output = new int[N];

    for (int i = 0; i < N; i++) {
        tablica_input[i] = i;
    }

    int *tablica_input_gpu, *tablica_output_gpu;
    cudaMalloc(&tablica_input_gpu, size);
    cudaMalloc(&tablica_output_gpu, size);

    cudaMemcpy(tablica_input_gpu, tablica_input, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    squareKernel<<<blocksPerGrid, threadsPerBlock>>>(tablica_input_gpu, tablica_output_gpu, N);

    cudaMemcpy(tablica_output, tablica_output_gpu, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 20; i++) {
        cout << tablica_output[i] << " ";
    }

    cudaFree(tablica_input_gpu);
    cudaFree(tablica_output_gpu);
    delete[] tablica_input;
    delete[] tablica_output;

    return 0;
}