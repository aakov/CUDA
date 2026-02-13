#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void kernel(int *t1, int *t2, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        t2[idx] = t1[idx] * t1[idx];
    }
}

int main() {

    int tablica[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int wynik[10];

    int *t1gpu, *t2gpu;
    cudaMalloc(&t1gpu, 10 * sizeof(int));
    cudaMalloc(&t2gpu, 10 * sizeof(int));

    cudaMemcpy(t1gpu, tablica, 10 * sizeof(int), cudaMemcpyHostToDevice);

    kernel<<<1, 10>>>(t1gpu, t2gpu, 10);
    cudaDeviceSynchronize();
    cudaMemcpy(wynik, t2gpu, 10 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        cout << wynik[i] << " ";
    }

    cudaFree(t1gpu);
    cudaFree(t2gpu);

    return 0;
}