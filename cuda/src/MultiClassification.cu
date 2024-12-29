#include "MultiClassification.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

// CUDA kernel for softmax activation
__global__ void softmax_activation(const float* input, float* scores, int num_classes, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float max_val = -INFINITY;
        float sum = 0.0f;

        // Find max for numerical stability
        for (int i = 0; i < num_classes; i++) {
            max_val = fmaxf(max_val, input[idx * num_classes + i]);
        }

        // Compute softmax scores
        for (int i = 0; i < num_classes; i++) {
            float exp_val = expf(input[idx * num_classes + i] - max_val);
            scores[idx * num_classes + i] = exp_val;
            sum += exp_val;
        }

        // Normalize
        for (int i = 0; i < num_classes; i++) {
            scores[idx * num_classes + i] /= sum;
        }
    }
}

// CUDA kernel for sigmoid activation
__global__ void sigmoid_activation(const float* input, float* scores, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        scores[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

// CUDA kernel for cross-entropy loss
__global__ void cross_entropy_loss(const float* scores, const float* labels, float* loss, int num_classes, int batch_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < batch_size) {
        float sample_loss = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            float log_score = logf(fmaxf(scores[idx * num_classes + i], 1e-7)); // Avoid log(0)
            sample_loss -= labels[idx * num_classes + i] * log_score;
        }
        loss[idx] = sample_loss;
    }
}

// CUDA kernel for accuracy computation
__global__ void accuracy_kernel(const float* scores, const float* labels, int* correct_count, int num_classes, int batch_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < batch_size) {
        int max_idx = 0;
        int label_idx = 0;
        float max_val = -INFINITY;

        for (int i = 0; i < num_classes; i++) {
            if (scores[idx * num_classes + i] > max_val) {
                max_val = scores[idx * num_classes + i];
                max_idx = i;
            }
            if (labels[idx * num_classes + i] == 1.0f) {
                label_idx = i;
            }
        }

        if (max_idx == label_idx) {
            atomicAdd(correct_count, 1);
        }
    }
}

// CUDA kernel for softmax gradient (backward pass)
__global__ void softmax_backward(const float* scores, const float* labels, float* d_input, int num_classes, int batch_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < batch_size * num_classes) {
        int example_idx = idx / num_classes;
        d_input[idx] = scores[idx] - labels[example_idx * num_classes + idx % num_classes];
    }
}

// Constructor
MultiClassification::MultiClassification(int num_classes, int batch_size, const std::string& activation)
    : num_classes_(num_classes), batch_size_(batch_size), activation_(activation), h_loss_(0.0f), h_accuracy_(0.0f),
      d_input_(nullptr), d_scores_(nullptr), d_loss_(nullptr), d_d_input_(nullptr) {
    allocate_buffers();
}

// Destructor
MultiClassification::~MultiClassification() {
    cudaFree(d_scores_);
    cudaFree(d_loss_);
    cudaFree(d_d_input_);
}

// Allocate GPU buffers
void MultiClassification::allocate_buffers() {
    size_t scores_size = batch_size_ * num_classes_ * sizeof(float);
    size_t loss_size = batch_size_ * sizeof(float);

    cudaMalloc(&d_scores_, scores_size);
    cudaMalloc(&d_loss_, loss_size);
    cudaMalloc(&d_d_input_, scores_size);
}

// Set batch size and reallocate buffers
void MultiClassification::set_batch_size(int batch_size) {
    batch_size_ = batch_size;
    allocate_buffers();
}

// Forward pass
void MultiClassification::forward() {
    int threads = 256;
    int blocks = (batch_size_ + threads - 1) / threads;

    if (activation_ == "softmax") {
        softmax_activation<<<blocks, threads>>>(d_input_, d_scores_, num_classes_, batch_size_);
    } else if (activation_ == "sigmoid") {
        int total_size = batch_size_ * num_classes_;
        int total_blocks = (total_size + threads - 1) / threads;
        sigmoid_activation<<<total_blocks, threads>>>(d_input_, d_scores_, total_size);
    } else {
        throw std::invalid_argument("Unsupported activation type: " + activation_);
    }
    cudaDeviceSynchronize();
}

// Compute loss
void MultiClassification::compute_loss(const float* labels) {
    int threads = 256;
    int blocks = (batch_size_ + threads - 1) / threads;

    cross_entropy_loss<<<blocks, threads>>>(d_scores_, labels, d_loss_, num_classes_, batch_size_);
    cudaDeviceSynchronize();

    // Reduce loss to the host
    float* h_loss_array = new float[batch_size_];
    cudaMemcpy(h_loss_array, d_loss_, batch_size_ * sizeof(float), cudaMemcpyDeviceToHost);

    h_loss_ = 0.0f;
    for (int i = 0; i < batch_size_; i++) {
        h_loss_ += h_loss_array[i];
    }
    h_loss_ /= batch_size_;

    delete[] h_loss_array;
}

// Compute accuracy
void MultiClassification::compute_accuracy(const float* labels) {
    int threads = 256;
    int blocks = (batch_size_ + threads - 1) / threads;

    int* d_correct_count;
    int h_correct_count = 0;

    cudaMalloc(&d_correct_count, sizeof(int));
    cudaMemset(d_correct_count, 0, sizeof(int));

    accuracy_kernel<<<blocks, threads>>>(d_scores_, labels, d_correct_count, num_classes_, batch_size_);
    cudaMemcpy(&h_correct_count, d_correct_count, sizeof(int), cudaMemcpyDeviceToHost);

    h_accuracy_ = static_cast<float>(h_correct_count) / batch_size_;

    cudaFree(d_correct_count);
}

// Backward pass
void MultiClassification::backward(const float* labels) {
    int threads = 256;
    int blocks = (batch_size_ * num_classes_ + threads - 1) / threads;

    if (activation_ == "softmax") {
        softmax_backward<<<blocks, threads>>>(d_scores_, labels, d_d_input_, num_classes_, batch_size_);
    } else {
        throw std::invalid_argument("Backward pass not implemented for this activation function.");
    }
    cudaDeviceSynchronize();
}
