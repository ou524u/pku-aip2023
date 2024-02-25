#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>

template <typename TContent = float>
class Tensor {
   public:
    TContent* data;  // Pointer to store tensor data

    std::vector<int> shape;
    std::string device;  // Device information, can be "CPU" or "GPU"
    size_t size;

   public:
    // Constructor for CPU
    Tensor(const std::vector<int>& shape, std::string device);

    Tensor(const std::vector<int>& shape, std::string device, const std::vector<TContent>& content);

    // Destructor
    ~Tensor();

    // Deep copy constructor
    Tensor(const Tensor<TContent>& other);

    // Deep copy assignment operator
    Tensor<TContent>& operator=(const Tensor<TContent>& other);

    // Calculate total size of tensor data based on shape
    inline size_t calculateSize();

    // Switch to CPU
    Tensor<TContent>& cpu();
    // Switch to GPU
    Tensor<TContent>& gpu();

    // element-wise adding
    Tensor<TContent> operator+(const Tensor<TContent>& other_tensor) const;
    // element-wise minusing
    Tensor<TContent> operator-(const Tensor<TContent>& other_tensor) const;

    // // random
    Tensor<TContent>& random(TContent a, TContent b);

    // // ones. datatype uncheck
    Tensor<TContent>& ones();

    // // floatKs. datatype uncheck
    Tensor<TContent>& floatKs(float k);

    Tensor<TContent>& zeros();

    Tensor<TContent>& negative();

    Tensor<TContent>& mults(const TContent scalar);

    Tensor<TContent>& resize(const std::vector<int>& new_shape);

    Tensor<TContent> sqrtForward() const;

    // Tensor<TContent> sqrtBackward() const;

    // Forward pass for ReLU activation
    Tensor<TContent> reluForward() const;

    // Backward pass for ReLU activation (computing gradients)
    Tensor<TContent> reluBackward(const Tensor<TContent>& gradients) const;

    // Forward pass for Sigmoid activation
    Tensor<TContent> sigmoidForward() const;

    // Backward pass for Sigmoid activation (computing gradients)
    Tensor<TContent> sigmoidBackward(const Tensor<TContent>& gradients) const;

    template <typename TContent_>
    friend std::ostream& operator<<(std::ostream& os, const Tensor<TContent_>& tensor);

    // slicing
    Tensor<TContent> operator[](const int k) const;

    // std::string toString();

    std::vector<int> getshape();
    std::vector<TContent> getdata();
    std::string getdevice();

    Tensor<TContent> getflat() const;
};
// Overload << operator to print Tensor information

#include "Tensor.inl"

#endif