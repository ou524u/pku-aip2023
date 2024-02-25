#ifndef LAYERS_H
#define LAYERS_H

#include <cublas_v2.h>
#include "Tensor.h"

// Fully Connected Layer Forward Pass
template <typename TContent>
void fc_forward(
    const Tensor<TContent>& input,
    Tensor<TContent>& output,
    const Tensor<TContent>& weights,
    const Tensor<TContent>& bias);

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
    Tensor<TContent>& grad_bias);

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
    const int dilation_w);

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
    const int dilation_w);

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
    const int dilation_w);

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
    const int dilation_w);

template <typename TContent>
void maxpool_forward(
    const Tensor<TContent>& input,
    Tensor<TContent>& output,
    Tensor<TContent>& masks,
    const std::vector<int>& kernel_shape,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w);

template <typename TContent>
void maxpool_backward(
    const Tensor<TContent>& grad_output,
    const Tensor<TContent>& masks,
    Tensor<TContent>& grad_input,
    const std::vector<int>& kernel_shape,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w);

template <typename TContent>
void softmax_forward(
    const Tensor<TContent>& input,
    Tensor<TContent>& output);

template <typename TContent>
float softmax_loss(
    const Tensor<TContent>& softmax_output,
    const Tensor<int>& real_labels);

template <typename TContent>
// change loss_tensor but not return loss as a value.
void crossEntropy_forward(
    const Tensor<TContent>& input,
    Tensor<TContent>& loss,
    const Tensor<int>& real_labels);

template <typename TContent>
void softmax_corssEntropy_backward(
    const Tensor<TContent>& softmax_output,
    Tensor<TContent>& grad_input,
    const Tensor<int>& real_labels);

#include "Layers.inl"
#endif