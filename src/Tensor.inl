#ifndef TENSOR_INL
#define TENSOR_INL

#include <iostream>
#include <random>  // For random number generation
#include <sstream>
#include <vector>

#include "Tensor.h"
#include "Tensor_kernels.h"
#include "global.h"

template <typename TContent>
Tensor<TContent>::Tensor(const std::vector<int>& shape, std::string device)
    : shape(shape), device(device) {
    size = calculateSize();
    if (device == "CPU") {
        data = new TContent[size];
        return;
    }

    else if (device == "GPU") {
        cudaMalloc(&data, size * sizeof(TContent));
        // cudaMemcpy(d_data, data, size * sizeof(TContent), cudaMemcpyHostToDevice);
        return;
    }
    throw "error unknown device";
}

template <typename TContent>
Tensor<TContent>::Tensor(const std::vector<int>& shape, std::string device, const std::vector<TContent>& content)
    : shape(shape), device(device) {
    size = calculateSize();
    if (device == "CPU") {
        data = new TContent[size];
        std::copy_n(content.begin(), std::min(size, content.size()), data);
        return;
    }

    else if (device == "GPU") {
        TContent* c_data = new TContent[size];
        std::copy_n(content.begin(), std::min(size, content.size()), c_data);
        cudaMalloc(&data, size * sizeof(TContent));
        cudaMemcpy(data, c_data, size * sizeof(TContent), cudaMemcpyHostToDevice);
        delete[] c_data;
        return;
    }
    throw "error unknown device";
}

// Destructor
template <typename TContent>
Tensor<TContent>::~Tensor() {
    if (device == "GPU") {
        cudaFree(data);
    } else {
        delete[] data;
    }
#ifdef PRINT_CHECK
    std::cout << "Tensor destroyed on " << device << std::endl;
#endif
}

// Deep copy constructor
template <typename TContent>
Tensor<TContent>::Tensor(const Tensor& other)
    : shape(other.shape), device(other.device) {
    size = calculateSize();
    if (device == "CPU") {
        data = new TContent[size];
        std::copy(other.data, other.data + size, data);
        return;
    }

    else if (device == "GPU") {
        cudaMalloc(&data, size * sizeof(TContent));
        cudaMemcpy(data, other.data, size * sizeof(TContent), cudaMemcpyDeviceToDevice);
        return;
    }
#ifdef PRINT_CHECK
    std::cout << "Tensor copied to " << device << std::endl;
#endif
    throw "error unknown device";
}

// Deep copy assignment operator
template <typename TContent>
Tensor<TContent>& Tensor<TContent>::operator=(const Tensor<TContent>& other) {
    if (this != &other) {  // Check for self-assignment
        // Deallocate existing data
        if (device == "GPU") {
            cudaFree(data);
        } else {
            delete[] data;
        }

        // Copy data from the other tensor
        shape = other.shape;
        device = other.device;
        size = calculateSize();

        if (device == "CPU") {
            data = new TContent[size];
            std::copy(other.data, other.data + size, data);
        } else if (device == "GPU") {
            cudaMalloc(&data, size * sizeof(TContent));
            cudaMemcpy(data, other.data, size * sizeof(TContent), cudaMemcpyDeviceToDevice);
        }
    }
#ifdef PRINT_CHECK
    std::cout << "Tensor copied to " << device << std::endl;
#endif
    return *this;
}

// Calculate total size of tensor data based on shape
template <typename TContent>
inline size_t Tensor<TContent>::calculateSize() {
    size_t totalSize = 1;
    for (int dim : shape) {
        totalSize *= dim;
    }
    return totalSize;
}

// Switch to CPU
template <typename TContent>
Tensor<TContent>& Tensor<TContent>::cpu() {
    if (device == "CPU") {
#ifdef PRINT_CHECK
        std::cout << "Tensor is already on CPU" << std::endl;
#endif
        return *this;
    }
    if (device == "GPU") {
#ifdef PRINT_CHECK
        std::cout << "Moving tensor to CPU" << std::endl;
#endif

        TContent* c_data = new TContent[size];
        cudaMemcpy(c_data, data, size * sizeof(TContent), cudaMemcpyDeviceToHost);
        cudaFree(data);
        data = c_data;
        device = "CPU";
        return *this;
    }
    throw "error unknown device";
}

// Switch to GPU
template <typename TContent>
Tensor<TContent>& Tensor<TContent>::gpu() {
    if (device == "GPU") {
#ifdef PRINT_CHECK
        std::cout << "Tensor is already on GPU" << std::endl;
#endif
        return *this;
    }
    if (device == "CPU") {
#ifdef PRINT_CHECK
        std::cout << "Moving tensor to GPU" << std::endl;
#endif

        TContent* d_data = nullptr;
        cudaMalloc(&d_data, size * sizeof(TContent));
        cudaMemcpy(d_data, data, size * sizeof(TContent), cudaMemcpyHostToDevice);

        delete[] data;
        data = d_data;

        device = "GPU";
        return *this;
    }
    throw "error unknown device";
}

// element-wise adding
template <typename TContent>
Tensor<TContent> Tensor<TContent>::operator+(const Tensor<TContent>& other_tensor) const {
    if (shape != other_tensor.shape) {
        throw "error adding tensor with different shape";
    }
    if (device != other_tensor.device) {
        throw "error adding tensor with different device";
    }
    Tensor<TContent> res_tensor(shape, device);
    res_tensor.zeros();
    if (device == "CPU") {
        for (int i = 0; i < size; ++i) {
            res_tensor.data[i] = data[i] + other_tensor.data[i];
        }
        return res_tensor;
    }
    if (device == "GPU") {
        cudaAdd<<<CudaGetBlocks(size), kCudaThreadsNum>>>(
            data, other_tensor.data, res_tensor.data, size);
        cudaDeviceSynchronize();
        return res_tensor;
    }

    throw "error unknown device";
}

// element-wise minusing
template <typename TContent>
Tensor<TContent> Tensor<TContent>::operator-(const Tensor<TContent>& other_tensor) const {
    if (shape != other_tensor.shape) {
        throw "error minusing tensor with different shape";
    }
    if (device != other_tensor.device) {
        throw "error minusing tensor with different device";
    }
    Tensor<TContent> res_tensor(shape, device);
    res_tensor.zeros();
    if (device == "CPU") {
        for (int i = 0; i < size; ++i) {
            res_tensor.data[i] = data[i] - other_tensor.data[i];
        }
        return res_tensor;
    }
    if (device == "GPU") {
        cudaMinus<<<CudaGetBlocks(size), kCudaThreadsNum>>>(
            data, other_tensor.data, res_tensor.data, size);
        cudaDeviceSynchronize();
        return res_tensor;
    }

    throw "error unknown device";
}

// random
template <typename TContent>
Tensor<TContent>& Tensor<TContent>::random(TContent a, TContent b) {
    if (device == "CPU") {
        std::random_device rd;
        std::default_random_engine generator(rd());
        if constexpr (std::is_floating_point_v<TContent>) {
            std::uniform_real_distribution<TContent> distribution(a, b);
            for (size_t i = 0; i < size; ++i) {
                data[i] = distribution(generator);
            }
        } else if constexpr (std::is_integral_v<TContent>) {
            std::uniform_int_distribution<TContent> distribution(a, b);
            for (size_t i = 0; i < size; ++i) {
                data[i] = distribution(generator);
            }
        } else {
            throw "Tensor.random datatype not suppoted";
        }
        return *this;
    }
    if (device == "GPU") {
        // kernels from gpu
        if constexpr (std::is_floating_point_v<TContent> || std::is_integral_v<TContent>) {
            cudaRandom<<<CudaGetBlocks(size), kCudaThreadsNum>>>(data, size, a, b);
            cudaDeviceSynchronize();
        } else {
            throw "Tensor.random datatype not suppoted";
        }
        cudaDeviceSynchronize();
        return *this;
    }
    throw "error unknown device";
}

// // ones. datatype uncheck
template <typename TContent>
Tensor<TContent>& Tensor<TContent>::ones() {
    if (device == "CPU") {
        for (size_t i = 0; i < size; ++i) {
            data[i] = 1;
        }
        return *this;
    }
    if (device == "GPU") {
        // kernels from gpu
        cudaOnes<<<CudaGetBlocks(size), kCudaThreadsNum>>>(data, size);
        cudaDeviceSynchronize();
        return *this;
    }
    throw "error unknown device";
}

template <typename TContent>
Tensor<TContent>& Tensor<TContent>::negative() {
    if (device == "CPU") {
        for (size_t i = 0; i < size; ++i) {
            data[i] = -data[i];
        }
        return *this;
    }
    if (device == "GPU") {
        // kernels from gpu
        cudaNegative<<<CudaGetBlocks(size), kCudaThreadsNum>>>(data, size);
        cudaDeviceSynchronize();
        return *this;
    }
    throw "error unknown device";
}

template <typename TContent>
Tensor<TContent>& Tensor<TContent>::mults(const TContent scalar) {
    if (device == "CPU") {
        for (size_t i = 0; i < size; ++i) {
            data[i] *= scalar;
        }
        return *this;
    }
    if (device == "GPU") {
        // kernels from gpu
        cudaMults<<<CudaGetBlocks(size), kCudaThreadsNum>>>(data, size, scalar);
        cudaDeviceSynchronize();
        return *this;
    }
    throw "error unknown device";
}

// // floatKs. datatype uncheck
template <typename TContent>
Tensor<TContent>& Tensor<TContent>::floatKs(float k) {
    if (device == "CPU") {
        for (size_t i = 0; i < size; ++i) {
            data[i] = k;
        }
        return *this;
    }
    if (device == "GPU") {
        // kernels from gpu
        cudaFloatKs<<<CudaGetBlocks(size), kCudaThreadsNum>>>(data, size, k);
        cudaDeviceSynchronize();
        return *this;
    }
    throw "error unknown device";
}

// zeros
template <typename TContent>
Tensor<TContent>& Tensor<TContent>::zeros() {
    if (device == "CPU") {
        memset(data, 0, size * sizeof(TContent));
        return *this;
    }
    if (device == "GPU") {
        cudaMemset(data, 0, size * sizeof(TContent));
        return *this;
    }
    throw "error unknown device";
}

// my resize: only care about total size
template <typename TContent>
Tensor<TContent>& Tensor<TContent>::resize(const std::vector<int>& new_shape) {
    size_t new_size = 1;
    for (int i = 0; i < new_shape.size(); ++i) {
        new_size *= new_shape[i];
    }
    size_t old_size = calculateSize();
    shape = new_shape;
    size = calculateSize();
    if (new_size == old_size) {
        return *this;
    }
    if (device == "CPU") {
        TContent* new_data = new TContent[size];
        std::copy(data, data + std::min(new_size, old_size), new_data);
        delete[] data;  // Free the original memory
        data = new_data;
        return *this;
    }
    if (device == "GPU") {
        TContent* new_data = nullptr;
        cudaMalloc(&new_data, new_size * sizeof(TContent));
        cudaMemcpy(new_data, data, std::min(new_size, old_size) * sizeof(TContent), cudaMemcpyDeviceToDevice);
        cudaFree(data);  // Free the original GPU memory
        data = new_data;
        return *this;
    }
    throw "error unknown device";
}

// // Forward pass for ReLU activation
// template <typename TContent>
// Tensor<TContent>& Tensor<TContent>::reluForward() {
//     if (device == "CPU") {
//         for (size_t i = 0; i < size; ++i) {
//             data[i] = (data[i] > 0) ? data[i] : 0;
//         }
//         return *this;
//     }
//     if (device == "GPU") {
//         // Create a kernel for ReLU forward pass
//         ReluForward<<<CudaGetBlocks(size), kCudaThreadsNum>>>(data, size);
//         cudaDeviceSynchronize();
//         return *this;
//     }
//     throw "error unknown device";
// }

// Forward pass for ReLU activation, without changing the ori Tensor
template <typename TContent>
Tensor<TContent> Tensor<TContent>::reluForward() const {
    Tensor<TContent> result(*this);
    if (device == "CPU") {
        for (size_t i = 0; i < result.size; ++i) {
            result.data[i] = (result.data[i] > 0) ? result.data[i] : 0;
        }
        return result;
    }
    if (device == "GPU") {
        // Create a kernel for ReLU forward pass
        ReluForward<<<CudaGetBlocks(result.size), kCudaThreadsNum>>>(result.data, result.size);
        cudaDeviceSynchronize();
        return result;
    }
    throw "error unknown device";
}

// // Backward pass for ReLU activation (computing gradients)
// template <typename TContent>
// Tensor<TContent>& Tensor<TContent>::reluBackward(const Tensor& gradients) {
//     if (device == "CPU") {
//         if (gradients.device == "GPU") {
//             TContent* d_gradients = new TContent[size];
//             cudaMemcpy(d_gradients, gradients.data, size * sizeof(TContent), cudaMemcpyDeviceToHost);
//             for (size_t i = 0; i < size; ++i) {
//                 data[i] = (data[i] > 0) ? d_gradients[i] : 0;
//             }
//             delete[] d_gradients;
//             return *this;
//         }
//         if (gradients.device == "CPU") {
//             for (size_t i = 0; i < size; ++i) {
//                 data[i] = (data[i] > 0) ? gradients.data[i] : 0;
//             }
//             return *this;
//         }
//         throw "error unknown device";
//     }
//     if (device == "GPU") {
//         ReluBackward<<<CudaGetBlocks(size), kCudaThreadsNum>>>(data, gradients.data, size);
//         cudaDeviceSynchronize();
//         return *this;
//     }
//     throw "error unknown device";
// }

template <typename TContent>
Tensor<TContent> Tensor<TContent>::reluBackward(const Tensor<TContent>& gradients) const {
    Tensor<TContent> result(*this);  // Create a new Tensor as a copy of the original
    if (device == "CPU") {
        if (gradients.device == "GPU") {
            TContent* d_gradients = new TContent[size];
            cudaMemcpy(d_gradients, gradients.data, size * sizeof(TContent), cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < size; ++i) {
                result.data[i] = (result.data[i] > 0) ? d_gradients[i] : 0;
            }
            delete[] d_gradients;
            return result;
        }
        if (gradients.device == "CPU") {
            for (size_t i = 0; i < result.size; ++i) {
                result.data[i] = (result.data[i] > 0) ? gradients.data[i] : 0;
            }
            return result;
        }
        throw "error unknown device";
    }
    if (device == "GPU") {
        ReluBackward<<<CudaGetBlocks(result.size), kCudaThreadsNum>>>(result.data, gradients.data, result.size);
        cudaDeviceSynchronize();
        return result;
    }
    throw "error unknown device";
}

// // Forward pass for Sigmoid activation
// template <typename TContent>
// Tensor<TContent>& Tensor<TContent>::sigmoidForward() {
//     if (device == "CPU") {
//         for (size_t i = 0; i < size; ++i) {
//             data[i] = 1 / (1 + expf(-data[i]));
//         }
//         return *this;
//     }
//     if (device == "GPU") {
//         // Create a kernel for Sigmoid forward pass
//         SigmoidForward<<<CudaGetBlocks(size), kCudaThreadsNum>>>(data, size);
//         cudaDeviceSynchronize();
//         return *this;
//     }
//     throw "error unknown device";
// }

// // Backward pass for Sigmoid activation (computing gradients)
// template <typename TContent>
// Tensor<TContent>& Tensor<TContent>::sigmoidBackward(const Tensor& gradients) {
//     if (device == "CPU") {
//         if (gradients.device == "GPU") {
//             TContent* d_gradients = new TContent[size];
//             cudaMemcpy(d_gradients, gradients.data, size * sizeof(TContent), cudaMemcpyDeviceToHost);
//             for (size_t i = 0; i < size; ++i) {
//                 TContent sigmoid_output = 1 / (1 + expf(-data[i]));
//                 data[i] = sigmoid_output * (1 - sigmoid_output) * d_gradients[i];
//             }
//             delete[] d_gradients;
//             return *this;
//         }
//         if (device == "GPU") {
//             for (size_t i = 0; i < size; ++i) {
//                 TContent sigmoid_output = 1 / (1 + expf(-data[i]));
//                 data[i] = sigmoid_output * (1 - sigmoid_output) * gradients.data[i];
//             }
//             return *this;
//         }
//         throw "error unknown device";
//     }
//     if (device == "GPU") {
//         // Create a kernel for Sigmoid backward pass (computing gradients)
//         SigmoidBackward<<<CudaGetBlocks(size), kCudaThreadsNum>>>(data, gradients.data, size);
//         cudaDeviceSynchronize();
//         return *this;
//     }
//     throw "error unknown device";
// }

// Forward pass for Sigmoid activation
template <typename TContent>
Tensor<TContent> Tensor<TContent>::sigmoidForward() const {
    Tensor<TContent> result(*this);  // Create a new Tensor as a copy of the original

    if (device == "CPU") {
        for (size_t i = 0; i < result.size; ++i) {
            result.data[i] = 1 / (1 + expf(-result.data[i]));
        }
        return result;
    }
    if (device == "GPU") {
        // Create a kernel for Sigmoid forward pass
        SigmoidForward<<<CudaGetBlocks(result.size), kCudaThreadsNum>>>(result.data, result.size);
        cudaDeviceSynchronize();
        return result;
    }
    throw "error unknown device";
}

// Backward pass for Sigmoid activation (computing gradients)
template <typename TContent>
Tensor<TContent> Tensor<TContent>::sigmoidBackward(const Tensor<TContent>& gradients) const {
    Tensor<TContent> result(*this);  // Create a new Tensor as a copy of the original

    if (device == "CPU") {
        if (gradients.device == "GPU") {
            TContent* d_gradients = new TContent[size];
            cudaMemcpy(d_gradients, gradients.data, size * sizeof(TContent), cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < result.size; ++i) {
                TContent sigmoid_output = 1 / (1 + expf(-result.data[i]));
                result.data[i] = sigmoid_output * (1 - sigmoid_output) * d_gradients[i];
            }
            delete[] d_gradients;
            return result;
        }

        if (device == "GPU") {
            for (size_t i = 0; i < result.size; ++i) {
                TContent sigmoid_output = 1 / (1 + expf(-result.data[i]));
                result.data[i] = sigmoid_output * (1 - sigmoid_output) * gradients.data[i];
            }
            return result;
        }
        throw "error unknown device";
    }
    if (device == "GPU") {
        // Create a kernel for Sigmoid backward pass (computing gradients)
        SigmoidBackward<<<CudaGetBlocks(result.size), kCudaThreadsNum>>>(result.data, gradients.data, result.size);
        cudaDeviceSynchronize();
        return result;
    }
    throw "error unknown device";
}

// Forward pass for Sigmoid activation
template <typename TContent>
Tensor<TContent> Tensor<TContent>::sqrtForward() const {
    Tensor<TContent> result(*this);  // Create a new Tensor as a copy of the original
    if (device == "CPU") {
        for (size_t i = 0; i < result.size; ++i) {
            if (result.data[i] > 0) {
                result.data[i] = sqrt(result.data[i]);
            } else {
                result.data[i] = -sqrt(-result.data[i]);
            }
        }
        return result;
    }
    if (device == "GPU") {
        // Create a kernel for Sigmoid forward pass
        SqrtForward<<<CudaGetBlocks(result.size), kCudaThreadsNum>>>(result.data, result.size);
        cudaDeviceSynchronize();
        return result;
    }
    throw "error unknown device";
}

// Overload << operator to print Tensor information
template <typename TContent_>
std::ostream& operator<<(std::ostream& os, const Tensor<TContent_>& tensor) {
    os << "Device: " << tensor.device << "\nShape: [";
    for (size_t i = 0; i < tensor.shape.size(); ++i) {
        os << tensor.shape[i];
        if (i < tensor.shape.size() - 1) {
            os << ", ";
        }
    }
    os << "]\n";
    size_t size = tensor.size;
    if (tensor.device == "GPU") {
        Tensor<TContent_> temp_tensor(tensor);
        temp_tensor.cpu();
        for (size_t i = 0; i < temp_tensor.size; ++i) {
            os << temp_tensor.data[i] << " ";
            if ((i + 1) % temp_tensor.shape.back() == 0) {
                os << "\n";  // Newline after each row
            }
        }
        return os;
    }

    if (tensor.device == "CPU") {
        // Print data based on the first dimension of the shape
        for (size_t i = 0; i < size; ++i) {
            os << tensor.data[i] << " ";
            if ((i + 1) % tensor.shape.back() == 0) {
                os << "\n";  // Newline after each row
            }
        }

        return os;
    }
    // if (tensor.device == "GPU") {
    //     // Print data based on the first dimension of the shape
    //     cudaDPrintf<<<CudaGetBlocks(size), kCudaThreadsNum>>>(tensor.data, size);
    //     cudaDeviceSynchronize();
    //     os << "\n";
    //     return os;
    // }
    throw "error unknown device";
}

// slicing, returning as a new Tensor to avoid problem that:
// when you release a sliced one, the original one not deleted
// with more capability
template <typename TContent>
Tensor<TContent> Tensor<TContent>::operator[](const int k) const {
    if (k >= size) {
        throw "error slicing overbound";
        return *this;
    }
    size_t i = 0;
    size_t mul = 1;
    for (; i < shape.size();) {
        mul *= shape[i];
        ++i;
        if (k <= mul) {
            break;
        }
    }
    size_t new_size = 1;
    std::vector<int> new_shape;
    for (; i < shape.size(); i++) {
        new_shape.push_back(shape[i]);
        new_size *= shape[i];
    }

    if (new_shape.empty()) {
        new_shape.push_back(1);
    }

    Tensor<TContent> result(new_shape, device);
    if (device == "CPU") {
        std::copy(data + k * new_size, data + (k + 1) * new_size, result.data);
        return result;
    }
    if (device == "GPU") {
        cudaMemcpy(result.data, data + k * new_size, new_size * sizeof(TContent), cudaMemcpyDeviceToDevice);
        return result;
    }
    throw "error unknown device";
}

template <typename TContent>
std::vector<int> Tensor<TContent>::getshape() {
    return shape;
}

template <typename TContent>
std::vector<TContent> Tensor<TContent>::getdata() {
    TContent* c_data = new TContent[size];
    cudaMemcpy(c_data, data, size * sizeof(TContent), cudaMemcpyDeviceToHost);
    cudaFree(data);
    // initialize the return vector<TContent> with the copied pointer c_data
    std::vector<TContent> result(c_data, c_data + size);
    delete[] c_data;

    return result;
}

template <typename TContent>
std::string Tensor<TContent>::getdevice() {
    return device;
}

template <typename TContent>
Tensor<TContent> Tensor<TContent>::getflat() const {
    size_t batch_size = 1;
    for (int i = 0; i < shape.size() - 3; i++) {
        batch_size *= shape[i];
    }

    Tensor<TContent> result(*this);
    std::vector<int> flat_shape;

    flat_shape.push_back(batch_size);

    flat_shape.push_back(result.size / batch_size);
    result.resize(flat_shape);
    return result;
}
#endif