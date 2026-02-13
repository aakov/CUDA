#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void squareKernel(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * data[idx];
    }
}

int main() {
    int N = 1024;
    const int numStreams = 2;
    int streamSize = N / numStreams;
    int streamBytes = streamSize * sizeof(int);

    int* tablica;
    cudaMallocHost(&tablica, N * sizeof(int));

    for (int i = 0; i < N; i++) {
        tablica[i] = i;
    }

    int* tablica_gpu;
    cudaMalloc(&tablica_gpu, N * sizeof(int));

    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int threadsPerBlock = 256;
    int blocksPerStream = (streamSize + threadsPerBlock - 1) / threadsPerBlock;

    for (int i = 0; i < numStreams; i++) {
        int offset = i * streamSize;

        cudaMemcpyAsync(&tablica_gpu[offset], &tablica[offset], streamBytes,
                        cudaMemcpyHostToDevice, streams[i]);

        squareKernel<<<blocksPerStream, threadsPerBlock, 0, streams[i]>>>(
            &tablica_gpu[offset], streamSize);

        cudaMemcpyAsync(&tablica[offset], &tablica_gpu[offset], streamBytes,
                        cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < numStreams; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    for (int i = 0; i < 10; i++) {
        cout << tablica[i] << "\n";
    }

    for (int i = 0; i < numStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(tablica_gpu);
    cudaFreeHost(tablica);

    return 0;
}
