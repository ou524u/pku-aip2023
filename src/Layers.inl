#ifndef LAYERS_INL
#define LAYERS_INL

#include <assert.h>
#include <cublas_v2.h>

#include "Layer_kernels.h"
#include "Layers.h"
#include "global.h"

// Fully Connected Layer Forward Pass
template <typename TContent>
void fc_forward(
    const Tensor<TContent>& input,
    Tensor<TContent>& output,
    const Tensor<TContent>& weights,
    const Tensor<TContent>& bias) {
    // matrix product with gemm
#ifdef ASSERT_CHECK
    assert(input.device == "GPU" && output.device == "GPU" && weights.device == "GPU" && bias.device == "GPU");
#endif
    int batch_size = 1;
    std::vector<int> output_shape;
    for (size_t i = 0; i < input.shape.size() - 1; ++i) {
        batch_size *= input.shape[i];
        output_shape.push_back(input.shape[i]);
    }
    int in_features = input.shape.back();
#ifdef ASSERT_CHECK
    assert(in_features == weights.shape[0] && weights.shape.size() == 2);
#endif
    int out_features = weights.shape[1];

    output_shape.push_back(out_features);
    output.resize(output_shape);

    cudaGemm(CUBLAS_OP_N, CUBLAS_OP_N, batch_size, out_features, in_features,
             1.0f, input.data, weights.data, 0.0f, output.data);

    Tensor<TContent> out_ones(std::vector<int>{batch_size, 1}, "GPU");
    out_ones.ones();
    // here we need ones_

    if (bias.shape.back() == 1 && bias.shape.size() == 1) {
        // bias shape is (1)

        // here is the problem! when bias is on GPU you cannot access bias.data like this.
        // float bias_val = bias.data[0];
        Tensor<TContent> temp_bias(bias);
        float bias_val = temp_bias.cpu().data[0];

        Tensor<TContent> out_bias(std::vector<int>{1, out_features}, "GPU");
        out_bias.floatKs(bias_val);
        cudaGemm(CUBLAS_OP_N, CUBLAS_OP_N, batch_size, out_features, 1,
                 1.0f, out_ones.data, out_bias.data, 1.0f, output.data);
    } else {
        // bias shape should be 1*out_features
        cudaGemm(CUBLAS_OP_N, CUBLAS_OP_N, batch_size, out_features, 1,
                 1.0f, out_ones.data, bias.data, 1.0f, output.data);
    }
}

// Fully Connected Layer Backward Pass
template <typename TContent>
void fc_backward(
    const Tensor<TContent>& input,
    const Tensor<TContent>& output,
    const Tensor<TContent>& weights,
    const Tensor<TContent>& bias,
    Tensor<TContent>& grad_input,
    const Tensor<TContent>& grad_output,
    Tensor<TContent>& grad_weights,
    Tensor<TContent>& grad_bias) {
    // matrix product with gemm
#ifdef ASSERT_CHECK
    assert(input.device == "GPU" && weights.device == "GPU" && bias.device == "GPU");
    assert(grad_input.device == "GPU" && grad_output.device == "GPU" && grad_weights.device == "GPU" && grad_bias.device == "GPU");
#endif
    grad_input.resize(input.shape);
    grad_weights.resize(weights.shape);
    grad_bias.resize(bias.shape);

    grad_input.zeros();
    grad_weights.zeros();
    grad_bias.zeros();

    int batch_size = 1;
    for (size_t i = 0; i < input.shape.size() - 1; ++i) {
        batch_size *= input.shape[i];
    }
    int in_features = input.shape.back();
#ifdef ASSERT_CHECK
    assert(in_features == weights.shape[0] && weights.shape.size() == 2);
#endif
    int out_features = weights.shape[1];

    // Calculate gradients with respect to input
    // ronL/ronx(bs*inf) = ronL/rony(bs*outf) * Wt(outf*inf)
    cudaGemm(CUBLAS_OP_N, CUBLAS_OP_T, batch_size, in_features, out_features,
             1.0f, grad_output.data, weights.data, 0.0f, grad_input.data);

    // Calculate gradients with respect to weights
    // ronL/ronW(inf*outf) = Xt(inf*bs) * ronL/rony(bs*outf)
    cudaGemm(CUBLAS_OP_T, CUBLAS_OP_N, in_features, out_features, batch_size,
             1.0f, input.data, grad_output.data, 0.0f, grad_weights.data);

    Tensor<TContent> batch_ones(std::vector<int>{1, batch_size}, "GPU");
    batch_ones.ones();
    // here we need ones_

    if (bias.shape.back() == 1 && bias.shape.size() == 1) {  // bias is only a number

        Tensor<TContent> temp_bias(std::vector<int>{1, out_features}, "GPU");
        // Calculate gradients with respect to bias
        cudaGemm(CUBLAS_OP_N, CUBLAS_OP_N, 1, out_features, batch_size,
                 1.0f, batch_ones.data, grad_output.data, 0.0f, temp_bias.data);

        Tensor<TContent> outf_ones(std::vector<int>{out_features, 1}, "GPU");
        outf_ones.ones();
        cudaGemm(CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, out_features,
                 1.0f, temp_bias.data, outf_ones.data, 0.0f, grad_bias.data);
    } else {
        // Calculate gradients with respect to bias
        cudaGemm(CUBLAS_OP_N, CUBLAS_OP_N, 1, out_features, batch_size,
                 1.0f, batch_ones.data, grad_output.data, 0.0f, grad_bias.data);
    }
}

template <typename TContent>
void im2col(
    const Tensor<TContent>& im_tensor,
    Tensor<TContent>& col_tensor,
    const std::vector<int>& kernel_shape,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w) {
#ifdef ASSERT_CHECK
    assert(im_tensor.device == "GPU" && col_tensor.device == "GPU");
    assert(im_tensor.shape.size() >= 3 && kernel_shape.size() >= 2);
#endif
    int channels = im_tensor.shape[im_tensor.shape.size() - 3];
    int height = im_tensor.shape[im_tensor.shape.size() - 2];
    int width = im_tensor.shape.back();
    int batch_size = 1;
    for (size_t i = 0; i < im_tensor.shape.size() - 3; ++i) {
        batch_size *= im_tensor.shape[i];
    }

    int kernel_h = kernel_shape[kernel_shape.size() - 2];
    int kernel_w = kernel_shape.back();

    int height_col = (height + 2 * pad_h -
                      (dilation_h * (kernel_h - 1) + 1)) /
                         stride_h +
                     1;
    int width_col = (width + 2 * pad_w -
                     (dilation_w * (kernel_w - 1) + 1)) /
                        stride_w +
                    1;

    if (im_tensor.shape.size() == 3) {
        col_tensor.resize(std::vector<int>{height_col * width_col, channels * kernel_h * kernel_w});
    } else {
        col_tensor.resize(std::vector<int>{batch_size, height_col * width_col, channels * kernel_h * kernel_w});
    }

    int num_kernels = height_col * width_col * channels;
    int im_batchlen = channels * height * width;
    int col_batchlen = num_kernels * kernel_h * kernel_w;

    for (int i = 0; i < batch_size; i++) {
        cudaIm2Col<TContent><<<CudaGetBlocks(num_kernels), kCudaThreadsNum>>>(
            im_tensor.data + i * im_batchlen, col_tensor.data + i * col_batchlen,
            num_kernels, channels, height_col, width_col,
            height, width, kernel_h, kernel_w,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
    }
    cudaDeviceSynchronize();
    return;
}

template <typename TContent>
// need an extra (N,C,H,W) imshape for function
// should be able to take no batch_size
void col2im(
    const Tensor<TContent>& col_tensor,
    Tensor<TContent>& im_tensor,
    const std::vector<int>& kernel_shape,
    const std::vector<int>& im_shape,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w) {
#ifdef ASSERT_CHECK
    assert(im_tensor.device == "GPU" && col_tensor.device == "GPU");
    assert(col_tensor.shape.size() >= 2 && kernel_shape.size() >= 2);
#endif

    int channels = im_shape[im_shape.size() - 3];
    int height = im_shape[im_shape.size() - 2];
    int width = im_shape.back();
    int batch_size = 1;
    for (size_t i = 0; i < im_shape.size() - 3; ++i) {
        batch_size *= im_tensor.shape[i];
    }

    int kernel_h = kernel_shape[kernel_shape.size() - 2];
    int kernel_w = kernel_shape.back();

    int height_col = (height + 2 * pad_h -
                      (dilation_h * (kernel_h - 1) + 1)) /
                         stride_h +
                     1;
    int width_col = (width + 2 * pad_w -
                     (dilation_w * (kernel_w - 1) + 1)) /
                        stride_w +
                    1;

    if (im_shape.size() == 3) {
        im_tensor.resize(std::vector<int>{channels, height, width});
    } else {
        im_tensor.resize(std::vector<int>{batch_size, channels, height, width});
    }

    int num_kernels = channels * height * width;
    int im_batchlen = num_kernels;
    int col_batchlen = height_col * width_col * channels * kernel_h * kernel_w;

    for (int i = 0; i < batch_size; i++) {
        cudaCol2Im<TContent><<<CudaGetBlocks(num_kernels), kCudaThreadsNum>>>(
            col_tensor.data + i * col_batchlen, im_tensor.data + i * im_batchlen,
            num_kernels, channels, height_col, width_col,
            height, width, kernel_h, kernel_w,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
    }
    cudaDeviceSynchronize();
    return;
}

template <typename TContent>
void conv_forward(
    const Tensor<TContent>& input,
    Tensor<TContent>& output,
    const Tensor<TContent>& weights,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w) {
#ifdef ASSERT_CHECK
    assert(input.device == "GPU" && output.device == "GPU" && weights.device == "GPU");
    assert(input.shape.size() >= 3 && weights.shape.size() == 4);
#endif
    // input shape must be: [bs,fin,h,w]
    int channels_in = input.shape[input.shape.size() - 3];
    int height = input.shape[input.shape.size() - 2];
    int width = input.shape.back();

    int batch_size = 1;
    std::vector<int> output_shape;
    for (size_t i = 0; i < input.shape.size() - 3; ++i) {
        batch_size *= input.shape[i];
        output_shape.push_back(input.shape[i]);
    }

    // weights shape: [fout,fin,kerh,kerw]
    int channels_out = weights.shape[weights.shape.size() - 4];
#ifdef ASSERT_CHECK
    assert(weights.shape[weights.shape.size() - 3] == channels_in);
#endif
    int kernel_h = weights.shape[weights.shape.size() - 2];
    int kernel_w = weights.shape.back();

    int height_col =
        (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;

    int width_col =
        (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    output_shape.push_back(channels_out);
    output_shape.push_back(height_col);
    output_shape.push_back(width_col);
    output.resize(output_shape);

    // here proves that input is unable to handle multiple batch-size like [1,2]
    // input.resize(std::vector<int>{batch_size, channels_in, height, width});
    // output.resize({batch_size, channels_in, height_col, width_col})

    int output_batchlen = channels_out * height_col * width_col;

    for (int bs = 0; bs < batch_size; ++bs) {
        Tensor<TContent> input2col(
            std::vector<int>{height_col * width_col, channels_in * kernel_h * kernel_w}, "GPU");

        input2col.zeros();
        // it's so important here!!!
        Tensor<TContent> input_sliced(input[bs]);
        // never pass directly input[bs] into im2col, that would cause problem releasing the input.data
        // don't know why yet.
        im2col(input_sliced, input2col, {kernel_h, kernel_w},
               pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);

        // weights shape: [fout,fin,kerh,kerw]
        // input2col shape: [colh*colw,fin*kerh*kerw]
        // weights*input2col.T
        cudaGemm(CUBLAS_OP_N, CUBLAS_OP_T, channels_out, height_col * width_col, channels_in * kernel_h * kernel_w,
                 1.0f, weights.data, input2col.data, 0.0f, output.data + bs * output_batchlen);
    }

    cudaDeviceSynchronize();

    return;
}

template <typename TContent>
void conv_backward(
    const Tensor<TContent>& input,
    const Tensor<TContent>& output,
    const Tensor<TContent>& weights,
    Tensor<TContent>& grad_input,
    const Tensor<TContent>& grad_output,
    Tensor<TContent>& grad_weights,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w) {
#ifdef ASSERT_CHECK
    // actually we don't need output Tensor
    assert(input.device == "GPU" && weights.device == "GPU");
    assert(grad_input.device == "GPU" && grad_output.device == "GPU" && grad_weights.device == "GPU");
#endif
    grad_input.resize(input.shape);
    grad_input.zeros();

    grad_weights.resize(grad_weights.shape);
    grad_weights.zeros();

#ifdef ASSERT_CHECK
    // assert(input.shape == grad_input.shape && weights.shape == grad_weights.shape);
    assert(grad_output.shape.size() >= 3 && input.shape.size() >= 3 && weights.shape.size() == 4);
#endif
    // input shape: [bs,fin,h,w]
    int channels_in = input.shape[input.shape.size() - 3];
    int height = input.shape[input.shape.size() - 2];
    int width = input.shape.back();
    int batch_size = 1;
    for (size_t i = 0; i < input.shape.size() - 3; ++i) {
        batch_size *= input.shape[i];
    }

    // weights shape: [fout,fin,kerh,kerw]
    int channels_out = weights.shape[weights.shape.size() - 4];
#ifdef ASSERT_CHECK
    assert(weights.shape[weights.shape.size() - 3] == channels_in);
#endif
    int kernel_h = weights.shape[weights.shape.size() - 2];
    int kernel_w = weights.shape.back();

    int height_col =
        (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;

    int width_col =
        (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
#ifdef ASSERT_CHECK
    // grad_output shape: [bs,fout,colh,colw]
    assert(grad_output.shape[grad_output.shape.size() - 3] == channels_out);
    assert(grad_output.shape[grad_output.shape.size() - 2] == height_col);
    assert(grad_output.shape.back() == width_col);
#endif

    int output_batch_size = 1;
    for (size_t i = 0; i < output.shape.size() - 3; ++i) {
        output_batch_size *= grad_output.shape[i];
    }

#ifdef ASSERT_CHECK
    assert(output_batch_size == batch_size);
#endif

    // ronL/ronW(fout,fin*kerh*kerw) = ronL/rony(fout,colh*colw) * input2col(colh*colw,fin*kerh*kerw)
    // this should be sumed accorss batch_size

    // ronL/ronx(colh*colw,fin*kerh*kerw) = ronL/rony.T(fout,colh*colw) * W(fout,fin*kerh*kerw)
    // this lies along batchs of x
    // input shape: [bs,fin,h,w]
    Tensor<TContent> grad_input2col(
        std::vector<int>{batch_size, height_col * width_col, channels_in * kernel_h * kernel_w}, "GPU");

    grad_input2col.zeros();

    int input2col_batchlen = height_col * width_col * channels_in * kernel_h * kernel_w;
    grad_weights.zeros();
    for (int bs = 0; bs < batch_size; ++bs) {
        Tensor<TContent> input2col(
            std::vector<int>{height_col * width_col, channels_in * kernel_h * kernel_w}, "GPU");

        input2col.zeros();
        Tensor<TContent> input_sliced(input[bs]);
        Tensor<TContent> grad_output_sliced(grad_output[bs]);

        im2col(input_sliced, input2col, {kernel_h, kernel_w},
               pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);

        // calulate ronL/ronW, need adding so that use 1.0f
        cudaGemm(CUBLAS_OP_N, CUBLAS_OP_N, channels_out, channels_in * kernel_h * kernel_w, height_col * width_col,
                 1.0f, grad_output_sliced.data, input2col.data, 1.0f, grad_weights.data);

        // sliced grad_output is (fout,colh,colw)
        cudaGemm(CUBLAS_OP_T, CUBLAS_OP_N, height_col * width_col, channels_in * kernel_h * kernel_w, channels_out,
                 1.0f, grad_output_sliced.data, weights.data, 0.0f, grad_input2col.data + bs * input2col_batchlen);

        // std::cout << "@Layers.inl conv_backward check grad_input2col " << grad_input2col << std::endl;
    }
    cudaDeviceSynchronize();

    col2im(grad_input2col, grad_input, {kernel_h, kernel_w}, input.shape,
           pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);

    return;
}

template <typename TContent>
void maxpool_forward(
    const Tensor<TContent>& input,
    Tensor<TContent>& output,
    Tensor<TContent>& masks,
    const std::vector<int>& kernel_shape,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w) {
#ifdef ASSERT_CHECK
    assert(input.device == "GPU" && output.device == "GPU" && masks.device == "GPU");
    assert(input.shape.size() >= 3);

#endif
    // input shape: [bs,fin,h,w]
    int channels = input.shape[input.shape.size() - 3];
    int height = input.shape[input.shape.size() - 2];
    int width = input.shape.back();

    // int batch_size = 1;
    std::vector<int> output_shape;
    for (size_t i = 0; i < input.shape.size() - 3; ++i) {
        // batch_size *= input.shape[i];
        output_shape.push_back(input.shape[i]);
    }

    int kernel_h = kernel_shape[kernel_shape.size() - 2];
    int kernel_w = kernel_shape.back();

    int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    output_shape.push_back(channels);
    output_shape.push_back(height_col);
    output_shape.push_back(width_col);

    output.resize(output_shape);
    masks.resize(output_shape);

    int num_kernels = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());

    cudaMaxPoolingForward<TContent><<<CudaGetBlocks(num_kernels), kCudaThreadsNum>>>(
        input.data, output.data, masks.data,
        num_kernels, channels, height_col, width_col,
        height, width, kernel_h, kernel_w,
        pad_h, pad_w, stride_h, stride_w);
    cudaDeviceSynchronize();

    return;
}

template <typename TContent>
void maxpool_backward(
    const Tensor<TContent>& grad_output,
    const Tensor<TContent>& masks,
    Tensor<TContent>& grad_input,
    const std::vector<int>& kernel_shape,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w) {
#ifdef ASSERT_CHECK
    assert(grad_output.device == "GPU" && masks.device == "GPU" && grad_input.device == "GPU");
    assert(grad_output.shape.size() >= 3 && masks.shape == grad_output.shape);
#endif
    // input shape: [bs,fin,h,w]
    int channels = grad_output.shape[grad_output.shape.size() - 3];
    int height_col = grad_output.shape[grad_output.shape.size() - 2];
    int width_col = grad_output.shape.back();

    int batch_size = 1;
    std::vector<int> grad_input_shape;
    for (size_t i = 0; i < grad_output.shape.size() - 3; ++i) {
        batch_size *= grad_output.shape[i];
        grad_input_shape.push_back(grad_output.shape[i]);
    }

    int kernel_h = kernel_shape[kernel_shape.size() - 2];
    int kernel_w = kernel_shape.back();

    // int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    // int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    int height = (height_col - 1) * stride_h + kernel_h - 2 * pad_h;
    int width = (width_col - 1) * stride_w + kernel_w - 2 * pad_w;

    grad_input_shape.push_back(channels);
    grad_input_shape.push_back(height);
    grad_input_shape.push_back(width);

    grad_input.resize(grad_input_shape);
    grad_input.zeros();

    int num_kernels = batch_size * channels * height_col * width_col;

    cudaMaxPoolingBackward<TContent><<<CudaGetBlocks(num_kernels), kCudaThreadsNum>>>(
        grad_output.data, masks.data, grad_input.data,
        num_kernels, channels, height_col, width_col,
        height, width, kernel_h, kernel_w,
        pad_h, pad_w, stride_h, stride_w);
    cudaDeviceSynchronize();
    return;
}

template <typename TContent>
void softmax_forward(
    const Tensor<TContent>& input,
    Tensor<TContent>& output) {
#ifdef ASSERT_CHECK
    assert(input.device == "GPU" && output.device == "GPU");
    assert(input.shape.size() >= 2);
#endif
    // copy input to output
    output = input;
    int channels = input.shape.back();

    int batch_size = 1;
    for (size_t i = 0; i < input.shape.size() - 1; ++i) {
        batch_size *= input.shape[i];
    }

    Tensor<TContent> temp_batch(std::vector<int>{batch_size}, "GPU");
    temp_batch.zeros();
    cudaChannelMax<TContent><<<CudaGetBlocks(batch_size), kCudaThreadsNum>>>(
        input.data, temp_batch.data, batch_size, channels);
    cudaDeviceSynchronize();
    cudaChannelSubtract<TContent><<<CudaGetBlocks(batch_size), kCudaThreadsNum>>>(
        temp_batch.data, output.data, batch_size, channels);
    cudaDeviceSynchronize();
    cudaExp<TContent><<<CudaGetBlocks(batch_size * channels), kCudaThreadsNum>>>(
        output.data, batch_size * channels);
    cudaDeviceSynchronize();
    temp_batch.zeros();
    cudaChannelSum<TContent><<<CudaGetBlocks(batch_size), kCudaThreadsNum>>>(
        output.data, temp_batch.data, batch_size, channels);
    cudaDeviceSynchronize();
    cudaChannelDiv<TContent><<<CudaGetBlocks(batch_size), kCudaThreadsNum>>>(
        temp_batch.data, output.data, batch_size, channels);
    cudaDeviceSynchronize();
    return;
}

template <typename TContent>
float softmax_loss(
    const Tensor<TContent>& softmax_output,
    const Tensor<int>& real_labels) {
    int channels = softmax_output.shape.back();
    int batch_size = 1;
    for (size_t i = 0; i < softmax_output.shape.size() - 1; ++i) {
        batch_size *= softmax_output.shape[i];
    }
    // omit the shape checks

    Tensor<TContent> sft_out(softmax_output);
    sft_out.cpu();
    Tensor<int> real(real_labels);
    real.cpu();
    float error_hit = 0;
    for (int bs = 0; bs < batch_size; bs++) {
        int argmax = 0;
        TContent cmax = sft_out.data[bs * channels];
        for (int c = 1; c < channels; c++) {
            if (sft_out.data[bs * channels + c] > cmax) {
                cmax = sft_out.data[bs * channels + c];
                argmax = c;
            }
        }
        if (argmax != real.data[bs]) {
            error_hit += 1;
        }
    }
    return error_hit / batch_size;
}

template <typename TContent>
// change loss_tensor but not return loss as a value.
// loss value was sumed but not averaged
void crossEntropy_forward(
    const Tensor<TContent>& input,
    Tensor<TContent>& loss,
    const Tensor<int>& real_labels) {
#ifdef ASSERT_CHECK
    assert(input.device == "GPU" && loss.device == "GPU" && real_labels.device == "GPU");
    assert(input.shape.size() >= 2 && real_labels.shape.size() >= 1);
#endif
    int channels = input.shape.back();
    int batch_size = 1;
    for (size_t i = 0; i < input.shape.size() - 1; ++i) {
        batch_size *= input.shape[i];
    }
#ifdef ASSERT_CHECK
    assert(real_labels.shape.back() == batch_size);
#endif
    Tensor<TContent> logs_batch(std::vector<int>{batch_size}, "GPU");
    logs_batch.zeros();

    loss.resize(std::vector<int>{1});
    loss.zeros();

    cudaChannelLog<TContent><<<CudaGetBlocks(batch_size), kCudaThreadsNum>>>(
        input.data, logs_batch.data, real_labels.data, batch_size, channels);
    cudaDeviceSynchronize();

    cudaChannelSum<TContent><<<CudaGetBlocks(1), kCudaThreadsNum>>>(
        logs_batch.data, loss.data, 1, batch_size);

    cudaDeviceSynchronize();
    return;
}

template <typename TContent>
void softmax_corssEntropy_backward(
    const Tensor<TContent>& softmax_output,
    Tensor<TContent>& grad_input,
    const Tensor<int>& real_labels) {
#ifdef ASSERT_CHECK
    assert(softmax_output.device == "GPU" && grad_input.device == "GPU" && real_labels.device == "GPU");
    assert(softmax_output.shape.size() >= 2 && real_labels.shape.size() >= 1);
#endif
    int channels = softmax_output.shape.back();
    int batch_size = 1;
    for (size_t i = 0; i < softmax_output.shape.size() - 1; ++i) {
        batch_size *= softmax_output.shape[i];
    }
#ifdef ASSERT_CHECK
    assert(real_labels.shape.back() == batch_size);
#endif

    Tensor<TContent> real_values(softmax_output.shape, "GPU");
    real_values.zeros();

    cudaGetOnehot<TContent><<<CudaGetBlocks(batch_size), kCudaThreadsNum>>>(
        real_labels.data, real_values.data, batch_size, channels);
    cudaDeviceSynchronize();

    grad_input.resize(softmax_output.shape);
    grad_input = softmax_output - real_values;
    return;
}

#endif