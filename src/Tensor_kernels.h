#ifndef TENSOR_KERNELS_H
#define TENSOR_KERNELS_H

#include <curand_kernel.h>

template <typename TContent>
__device__ void dprintf(const TContent& x);

template <typename TContent>
__global__ void cudaDPrintf(TContent* data, int size);

template <typename TContent>
__global__ void cudaAdd(const TContent* data, const TContent* other_data, TContent* result_data, int size);

template <typename TContent>
__global__ void cudaMinus(const TContent* data, const TContent* other_data, TContent* result_data, int size);

template <typename TContent>
__global__ void cudaOnes(TContent* data, int size);

template <typename TContent>
__global__ void cudaNegative(TContent* data, int size);

template <typename TContent>
__global__ void cudaMults(TContent* data, int size, const TContent scalar);

template <typename TContent>
__global__ void cudaFloatKs(TContent* data, int size, float k);

template <typename TContent>
__global__ void cudaRandom(TContent* data, int size, TContent a, TContent b);

template <typename TContent>
__global__ void ReluForward(TContent* data, int size);

template <typename TContent>
__global__ void ReluBackward(TContent* data, TContent* gradata, int size);

// Sigmoid activation forward pass CUDA kernel
template <typename TContent>
__global__ void SigmoidForward(TContent* data, int size);

// Sigmoid activation backward pass (computing gradients) CUDA kernel
template <typename TContent>
__global__ void SigmoidBackward(TContent* data, TContent* gradata, int size);

template <typename TContent>
__global__ void SqrtForward(TContent* data, int size);

#include "Tensor_kernels.inl"

#endif