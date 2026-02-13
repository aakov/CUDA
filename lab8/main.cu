#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void squareKernel(int* data, int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * blockDim.x * gridDim.x + x;

    if (idx < size) {
        data[idx] = data[idx] * data[idx];
    }
}

int main() {

    int tablica[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    int *tablica_gpu;

    cudaMalloc(&tablica_gpu, 10 * sizeof(int));
    cudaMemcpy(tablica_gpu, tablica, 10 * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(2, 2);
    dim3 gridDim(2, 2);

    squareKernel<<<gridDim, blockDim>>>(tablica_gpu, 10);
    // cudaDeviceSynchronize();

    cudaMemcpy(tablica, tablica_gpu, 10 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++) {
        cout << tablica[i] << " ";
    }
    cudaFree(tablica_gpu);
    return 0;
}
