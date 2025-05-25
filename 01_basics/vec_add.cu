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

    int *h_a, *h_b, *h_c; // host vector pointers
    int *d_a, *d_b, *d_c; // device vector pointers

    size_t bytes = sizeof(int) * n; // size of array/vector in bytes

    // Allocate memory for host
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    // Allocate device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Initialize vectors
    matrixInit(h_a, n);
    matrixInit(h_b, n);

    // Copy data to GPU
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Threadblock size
    int NUM_THREADS = 256;

    // Grid size
    int NUM_BLOCKS = (int)ceil(n / NUM_THREADS);

    // Launch Kernel on default stream w/o shmem
    vectorAddition<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, n);

    // Copy sum vector from device to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Check for errors
    errorCheck(h_a, h_b, h_c, n);

    printf("SUCCESS- Completed\n");

    return 0;
}