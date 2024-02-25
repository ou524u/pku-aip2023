#ifndef TENSOR_KERNELS_INL
#define TENSOR_KERNELS_INL

#include "Tensor_kernels.h"
#include "global.h"

#include <cfloat>

template <typename TContent>
__device__ void dprintf(const TContent& x) {
    if constexpr (std::is_same<TContent, float>::value) {
        printf("%f ", x);
        return;
    }
    if constexpr (std::is_same<TContent, int>::value) {
        printf("%d ", x);
        return;
    }

    return;
}

template <typename TContent>
__global__ void cudaDPrintf(TContent* data, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        dprintf(data[i]);
    }
}

template <typename TContent>
__global__ void cudaAdd(const TContent* data, const TContent* other_data, TContent* result_data, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        result_data[i] = data[i] + other_data[i];
    }
}

template <typename TContent>
__global__ void cudaMinus(const TContent* data, const TContent* other_data, TContent* result_data, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        result_data[i] = data[i] - other_data[i];
    }
}

template <typename TContent>
__global__ void cudaOnes(TContent* data, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        data[i] = 1;
    }
}

template <typename TContent>
__global__ void cudaNegative(TContent* data, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        data[i] = -data[i];
    }
}

template <typename TContent>
__global__ void cudaMults(TContent* data, int size, const TContent scalar) {
    CUDA_KERNEL_LOOP(i, size) {
        data[i] *= scalar;
    }
}

template <typename TContent>
__global__ void cudaFloatKs(TContent* data, int size, float k) {
    CUDA_KERNEL_LOOP(i, size) {
        data[i] = k;
    }
}
#include <curand_kernel.h>
template <typename TContent>
__global__ void cudaRandom(TContent* data, int size, TContent a, TContent b) {
    CUDA_KERNEL_LOOP(i, size) {
        curandState state;
        curand_init(clock64(), threadIdx.x, 0, &state);
        float rand_val = curand_uniform(&state);     // Generate a random value between 0 and 1
        data[i] = TContent(a + (b - a) * rand_val);  // Scale to the desired range [a, b]
    }
}

template <typename TContent>
__global__ void ReluForward(TContent* data, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        data[i] = (data[i] > 0) ? data[i] : 0;
    }
}

template <typename TContent>
__global__ void ReluBackward(TContent* data, TContent* gradata, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        data[i] = (data[i] > 0) ? gradata[i] : 0;
    }
}
// Sigmoid activation forward pass CUDA kernel
template <typename TContent>
__global__ void SigmoidForward(TContent* data, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        data[i] = 1 / (1 + expf(-data[i]));
    }
}

// Sigmoid activation backward pass (computing gradients) CUDA kernel
template <typename TContent>
__global__ void SigmoidBackward(TContent* data, TContent* gradata, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        TContent sigmoid_output = 1 / (1 + expf(-data[i]));
        data[i] = sigmoid_output * (1 - sigmoid_output) * gradata[i];
    }
}

// Sigmoid activation forward pass CUDA kernel
template <typename TContent>
__global__ void SqrtForward(TContent* data, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        data[i] = (data[i] > 0) ? sqrt(data[i]) : -sqrt(-data[i]);
    }
}

#include "Tensor_kernels.inl"

#endif