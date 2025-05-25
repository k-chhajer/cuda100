// Adding all the libraries
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>


// Cuda Kernel for vector addition
__global__ void vectorAddition(int* a, int* b, int* c, int n) {
    // Calculate thread id
    int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Vector boundary guard
    if (thread_id < n) {
        // Each thread adds a single element
        c[thread_id] = a[thread_id] + b[thread_id];
    }
}

// Initialize vector of size n
void matrixInit(int* a, int n) {
    for(int i=0; i<n; i++) {
        a[i] = rand() % 100;
    }
}

// Check add result
void errorCheck(int* a, int* b, int* c, int n) {
    for(int i=0; i<n; i++) {
        assert(c[i] == (a[i] + b[i]));
    }
}

int main() {
    int n = 1 << 16; // size of array/vector
    int id = cudaGetDevice(&id); // get device id
    int *a, *b, *c; // vector pointers
    size_t bytes = sizeof(int) * n; // size of array/vector in bytes

    // Allocate device memory
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // Initialize vectors
    matrixInit(a, n);
    matrixInit(b, n);

    // Threadblock size
    int NUM_THREADS = 256;

    // Grid size
    int NUM_BLOCKS = (int)ceil(n / NUM_THREADS);

    // Launch Kernel on default stream w/o shmem
    // code below to prefetch
    // cudaMemPrefetchAsync(a, bytes, id);
    // cudaMemPrefetchAsync(b, bytes, id);
    vectorAddition<<<NUM_BLOCKS, NUM_THREADS>>>(a, b, c, n);

    // Synchronize (wait for operations to finish before using values)
    cudaDeviceSynchronize();

    // code below to prefetch c vector to host
    // cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

    // Check for errors
    errorCheck(a, b, c, n);

    printf("SUCCESS- Completed\n");

    return 0;
}