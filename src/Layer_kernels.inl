#ifndef LAYER_KERNELS_INL
#define LAYER_KERNELS_INL

#include <cublas_v2.h>

#include "Layer_kernels.h"
#include "global.h"

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
    float* C) {
    // cublasSgemm function is col-first
    int lda = k, ldb = n, ldc = n;
    if (transa != CUBLAS_OP_N) {
        lda = m;
    }
    if (transb != CUBLAS_OP_N) {
        ldb = k;
    }

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, transb, transa, n, m, k, &alpha, B, ldb, A, lda, &beta, C, ldc);
    cublasDestroy(handle);
    cudaDeviceSynchronize();
}

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
    const int dilation_w) {
    CUDA_KERNEL_LOOP(i, num_kernels) {
        TContent* col_ptr = data_col;
        col_ptr += (i * kernel_h * kernel_w);

        int channel_index = i % channels;
        int off_channel = i / channels;
        int colw_index = off_channel % width_col;
        int colh_index = off_channel / width_col;

        int h_offset = colh_index * stride_h - pad_h;
        int w_offset = colw_index * stride_w - pad_w;

        for (int ix = 0; ix < kernel_h; ++ix) {
            for (int iy = 0; iy < kernel_w; ++iy) {
                int h_im = h_offset + ix * dilation_h;
                int w_im = w_offset + iy * dilation_w;

                *col_ptr =
                    (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ? data_im[((channel_index)*height + h_im) * width + w_im] : 0;

                col_ptr += 1;
            }
        }
    }
}

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
    const int dilation_w) {
    CUDA_KERNEL_LOOP(i, num_kernels) {
        const int w_index = i % width + pad_w;
        const int h_index = (i / width) % height + pad_h;
        const int c_index = i / (width * height);

        int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
        int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;

        const int w_col_start =
            (w_index < kernel_extent_w) ? 0 : (w_index - kernel_extent_w) / stride_w + 1;
        const int w_col_end = min(w_index / stride_w + 1, width_col);
        const int h_col_start =
            (h_index < kernel_extent_h) ? 0 : (h_index - kernel_extent_h) / stride_h + 1;
        const int h_col_end = min(h_index / stride_h + 1, height_col);

        TContent val = 0;
        for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
            for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
                int h_k = (h_index - h_col * stride_h);
                int w_k = (w_index - w_col * stride_w);
                if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
                    h_k /= dilation_h;
                    w_k /= dilation_w;
                    // int data_col_index = (((c_index * kernel_h + h_k) * kernel_w + w_k) *
                    //     height_col + h_col) * width_col + w_col;
                    // channel=c_index, kerH=h_k, kerW=w_k, colH=h_col, colW=w_col

                    int data_col_index = (((h_col * width_col + w_col) * channels + c_index) *
                                              kernel_h +
                                          h_k) *
                                             kernel_w +
                                         w_k;

                    val += data_col[data_col_index];
                }
            }
        }
        data_im[i] = val;
    }
}

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
    const int stride_w) {
    CUDA_KERNEL_LOOP(i, num_kernels) {
        int batch_index = i / (width_col * height_col * channels);
        int channel_index = (i / (width_col * height_col)) % channels;
        int ph = (i / width_col) % height_col;
        int pw = i % width_col;

        // Calculate the starting point of the window
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        // Calculate the ending point of the window

        // Adjust starting point within the valid range
        int hend = min(hstart + kernel_h, height);
        int wend = min(wstart + kernel_w, width);

        // Adjust starting point within the valid range
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);

        TContent max_val = -FLT_MAX;
        int max_idx = -1;
        const TContent* ptr_offset = data_in + (batch_index * channels + channel_index) * height * width;

        // Iterate over the window and find the max value and its index
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                if (ptr_offset[h * width + w] > max_val) {
                    max_val = ptr_offset[h * width + w];
                    max_idx = h * width + w;
                }
            }
        }

        // Store the max value and its index in the output tensors
        data_out[i] = max_val;
        mask_out[i] = max_idx;
    }
}

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
    const int stride_w) {
    CUDA_KERNEL_LOOP(i, num_kernels) {
        int batch_index = i / (width_col * height_col * channels);
        int channel_index = (i / (width_col * height_col)) % channels;
        int ph = (i / width_col) % height_col;
        int pw = i % width_col;

        // Calculate the starting point of the window
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;

        // Adjust starting point within the valid range
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);

        int grad_in_index = (batch_index * channels + channel_index) * height * width + mask_out[i];
        grad_in[grad_in_index] = grad_out[i];
    }
}

// without using thrust::, we have to write a bunch of kernels for softmax
template <typename TContent>
// data_in shape is [N,C], while data_out shape is [N]
__global__ void cudaChannelMax(
    const TContent* data_in,
    TContent* data_out,
    const int num_kernels,
    const int channels) {
    CUDA_KERNEL_LOOP(i, num_kernels) {
        // choose max from data_in[i*channels] to data_in[(i+1)*channels]
        TContent maxval = -FLT_MAX;
        for (int c = 0; c < channels; ++c) {
            // maxval = (data_in[i * channels + c] < maxval) ? maxval : data_in[i * channels + c];
            maxval = max(data_in[i * channels + c], maxval);
        }
        data_out[i] = maxval;
    }
}
template <typename TContent>
// data_sub shape is [N], while data_out shape is [N,C]
__global__ void cudaChannelSubtract(
    const TContent* data_sub,
    TContent* data_out,
    const int num_kernels,
    const int channels) {
    CUDA_KERNEL_LOOP(i, num_kernels) {
        // choose max from data_in[i*channels] to data_in[(i+1)*channels]
        for (int c = 0; c < channels; ++c) {
            data_out[i * channels + c] -= data_sub[i];
        }
    }
}

template <typename TContent>
__global__ void cudaExp(
    TContent* data_out,
    const int num_kernels) {
    CUDA_KERNEL_LOOP(i, num_kernels) {
        // choose max from data_in[i*channels] to data_in[(i+1)*channels]
        data_out[i] = exp(data_out[i]);
    }
}
template <typename TContent>
__global__ void cudaChannelSum(
    const TContent* data_in,
    TContent* data_out,
    const int num_kernels,
    const int channels) {
    CUDA_KERNEL_LOOP(i, num_kernels) {
        // choose max from data_in[i*channels] to data_in[(i+1)*channels]
        TContent sumval = 0;
        for (int c = 0; c < channels; ++c) {
            sumval += data_in[i * channels + c];
        }
        data_out[i] = sumval;
    }
}
template <typename TContent>
__global__ void cudaChannelDiv(
    const TContent* data_div,
    TContent* data_out,
    const int num_kernels,
    const int channels) {
    CUDA_KERNEL_LOOP(i, num_kernels) {
        // choose max from data_in[i*channels] to data_in[(i+1)*channels]
        for (int c = 0; c < channels; ++c) {
            data_out[i * channels + c] /= data_div[i];
        }
    }
}
template <typename TContent>
__global__ void cudaChannelLog(
    const TContent* data_in,
    TContent* log_out,
    const int* real_labels,
    const int num_kernels,
    const int channels) {
    CUDA_KERNEL_LOOP(i, num_kernels) {
        log_out[i] = -log(data_in[i * channels + real_labels[i]]);
    }
}

template <typename TContent>
// this kernel cares only about the ones.
__global__ void cudaGetOnehot(
    const int* real_labels,
    TContent* real_values,
    const int num_kernels,
    const int channels) {
    CUDA_KERNEL_LOOP(i, num_kernels) {
        real_values[i * channels + real_labels[i]] = 1;
    }
}

#endif