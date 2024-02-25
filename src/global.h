#ifndef GLOBAL_H
#define GLOBAL_H

// #define PRINT_CHECK
#define ASSERT_CHECK

#define CUDA_KERNEL_LOOP(i, n)                          \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n);                                       \
         i += blockDim.x * gridDim.x)

// Use 512 threads per block
const int kCudaThreadsNum = 512;

inline int CudaGetBlocks(const int N) {
    return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}

#endif