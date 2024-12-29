#include "Activation.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

// CUDA kernel for ReLU forward pass
__global__ void relu_forward(const float* input, float* output, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// CUDA kernel for ReLU backward pass
__global__ void relu_backward(const float* input, const float* d_output, float* d_input, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        d_input[idx] = (input[idx] > 0.0f) ? d_output[idx] : 0.0f;
    }
}

// CUDA kernel for Sigmoid forward pass
__global__ void sigmoid_forward(const float* input, float* output, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

// CUDA kernel for Sigmoid backward pass
__global__ void sigmoid_backward(const float* input, const float* d_output, float* d_input, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        float sigmoid_val = 1.0f / (1.0f + expf(-input[idx]));
        d_input[idx] = d_output[idx] * sigmoid_val * (1.0f - sigmoid_val);
    }
}

// CUDA kernel for Tanh forward pass
__global__ void tanh_forward(const float* input, float* output, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

// CUDA kernel for Tanh backward pass
__global__ void tanh_backward(const float* input, const float* d_output, float* d_input, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        float tanh_val = tanhf(input[idx]);
        d_input[idx] = d_output[idx] * (1.0f - tanh_val * tanh_val);
    }
}

// Constructor
Activation::Activation(const std::string& activation_type, int size, int batch_size)
    : activation_type_(activation_type), size_(size), batch_size_(batch_size),
      d_input_(nullptr), d_output_(nullptr), d_d_input_(nullptr) {
    allocate_buffers();
}

// Destructor
Activation::~Activation() {
    cudaFree(d_output_);
    cudaFree(d_d_input_);
}

// Allocate GPU buffers
void Activation::allocate_buffers() {
    cudaFree(d_output_);
    cudaFree(d_d_input_);
    cudaMalloc(&d_output_, batch_size_ * size_ * sizeof(float));
    cudaMalloc(&d_d_input_, batch_size_ * size_ * sizeof(float));
}

// Set batch size and reallocate buffers
void Activation::set_batch_size(int batch_size) {
    batch_size_ = batch_size;
    allocate_buffers();
}

// Forward pass
void Activation::forward() {
    int total_size = batch_size_ * size_;
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;

    if (activation_type_ == "relu") {
        relu_forward<<<blocks, threads>>>(d_input_, d_output_, total_size);
    } else if (activation_type_ == "sigmoid") {
        sigmoid_forward<<<blocks, threads>>>(d_input_, d_output_, total_size);
    } else if (activation_type_ == "tanh") {
        tanh_forward<<<blocks, threads>>>(d_input_, d_output_, total_size);
    } else {
        throw std::invalid_argument("Unsupported activation type: " + activation_type_);
    }
    cudaDeviceSynchronize();
}

// Backward pass
void Activation::backward(const float* d_output) {
    int total_size = batch_size_ * size_;
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;

    if (activation_type_ == "relu") {
        relu_backward<<<blocks, threads>>>(d_input_, d_output, d_d_input_, total_size);
    } else if (activation_type_ == "sigmoid") {
        sigmoid_backward<<<blocks, threads>>>(d_input_, d_output, d_d_input_, total_size);
    } else if (activation_type_ == "tanh") {
        tanh_backward<<<blocks, threads>>>(d_input_, d_output, d_d_input_, total_size);
    } else {
        throw std::invalid_argument("Unsupported activation type: " + activation_type_);
    }
    cudaDeviceSynchronize();
}
