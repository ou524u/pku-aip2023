#ifndef LAYER_KERNELS_H
#define LAYER_KERNELS_H

#include <curand.h>

// // Function to perform matrix multiplication using cuBLAS
// C(m,n) = A(m,k) * B(k,n)
void cudaGemm(
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C);

template <typename TContent>
__global__ void cudaIm2Col(
    const TContent* data_im,
    TContent* data_col,
    const int num_kernels,
    const int channels,
    const int height_col,
    const int width_col,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w);

template <typename TContent>
__global__ void cudaCol2Im(
    const TContent* data_col,
    TContent* data_im,
    const int num_kernels,
    const int channels,
    const int height_col,
    const int width_col,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w);

template <typename TContent>
__global__ void cudaMaxPoolingForward(
    const TContent* data_in,
    TContent* data_out,
    TContent* mask_out,
    const int num_kernels,
    const int channels,
    const int height_col,
    const int width_col,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w);

template <typename TContent>
// this kernel do only the setting, doesn't care about the zeros.
__global__ void cudaMaxPoolingBackward(
    const TContent* grad_out,
    const TContent* mask_out,
    TContent* grad_in,
    const int num_kernels,
    const int channels,
    const int height_col,
    const int width_col,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w);

// without using thrust::, we have to write a bunch of kernels for softmax
template <typename TContent>
// data_in shape is [N,C], while data_out shape is [N]
__global__ void cudaChannelMax(
    const TContent* data_in,
    TContent* data_out,
    const int num_kernels,
    const int channels);

template <typename TContent>
// data_sub shape is [N], while data_out shape is [N,C]
__global__ void cudaChannelSubtract(
    const TContent* data_sub,
    TContent* data_out,
    const int num_kernels,
    const int channels);

template <typename TContent>
__global__ void cudaExp(
    TContent* data_out,
    const int num_kernels);

template <typename TContent>
__global__ void cudaChannelSum(
    const TContent* data_in,
    TContent* data_out,
    const int num_kernels,
    const int channels);

template <typename TContent>
__global__ void cudaChannelDiv(
    const TContent* data_div,
    TContent* data_out,
    const int num_kernels,
    const int channels);

template <typename TContent>
__global__ void cudaChannelLog(
    const TContent* data_in,
    TContent* log_out,
    const int* real_labels,
    const int num_kernels,
    const int channels);

template <typename TContent>
// this kernel cares only about the ones.
__global__ void cudaGetOnehot(
    const int* real_labels,
    TContent* real_values,
    const int num_kernels,
    const int channels);

#include "Layer_kernels.inl"
#endif