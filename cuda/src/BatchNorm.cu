#include "BatchNorm.hpp"
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <cmath>

// CUDA kernel for computing mean
__global__ void compute_mean(const float* input, float* mean, int batch_size, int feature_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < feature_size) {
        float sum = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            sum += input[i * feature_size + idx];
        }
        mean[idx] = sum / batch_size;
    }
}

// CUDA kernel for computing variance
__global__ void compute_variance(const float* input, const float* mean, float* variance, int batch_size, int feature_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < feature_size) {
        float sum = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            float diff = input[i * feature_size + idx] - mean[idx];
            sum += diff * diff;
        }
        variance[idx] = sum / batch_size;
    }
}

// CUDA kernel for forward pass
__global__ void batchnorm_forward(const float* input, const float* mean, const float* variance,
                                  float* output, float* norm, const float* gamma, const float* beta,
                                  int batch_size, int feature_size, float epsilon) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < batch_size * feature_size) {
        int feature_idx = idx % feature_size;
        float x_hat = (input[idx] - mean[feature_idx]) / sqrtf(variance[feature_idx] + epsilon);
        norm[idx] = x_hat;
        output[idx] = gamma[feature_idx] * x_hat + beta[feature_idx];
    }
}

// CUDA kernel for backward pass (compute gradients for gamma and beta)
__global__ void compute_gradients(const float* norm, const float* d_output, float* d_gamma, float* d_beta,
                                  int batch_size, int feature_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < feature_size) {
        float grad_gamma = 0.0f;
        float grad_beta = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            grad_gamma += d_output[i * feature_size + idx] * norm[i * feature_size + idx];
            grad_beta += d_output[i * feature_size + idx];
        }
        d_gamma[idx] = grad_gamma;
        d_beta[idx] = grad_beta;
    }
}

// CUDA kernel for Adam optimizer
__global__ void adam_update(float* param, float* m, float* v, const float* grad, float learning_rate, float beta1, float beta2, float epsilon, int timestep, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad[idx];
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad[idx] * grad[idx];

        float m_hat = m[idx] / (1.0f - powf(beta1, timestep));
        float v_hat = v[idx] / (1.0f - powf(beta2, timestep));

        param[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

// Constructor
BatchNorm::BatchNorm(int feature_size, int batch_size, float epsilon)
    : feature_size_(feature_size), batch_size_(batch_size), epsilon_(epsilon) {
    initialize_parameters();
    allocate_buffers();
    initialize_adam();
}

// Destructor
BatchNorm::~BatchNorm() {
    cudaFree(d_gamma_);
    cudaFree(d_beta_);
    cudaFree(d_running_mean_);
    cudaFree(d_running_var_);
    cudaFree(d_mean_);
    cudaFree(d_var_);
    cudaFree(d_output_);
    cudaFree(d_norm_);
    cudaFree(d_d_gamma_);
    cudaFree(d_d_beta_);
    cudaFree(d_m_gamma_);
    cudaFree(d_v_gamma_);
    cudaFree(d_m_beta_);
    cudaFree(d_v_beta_);
}

void BatchNorm::initialize_parameters() {
    size_t param_size = feature_size_ * sizeof(float);

    cudaMalloc(&d_gamma_, param_size);
    cudaMalloc(&d_beta_, param_size);
    cudaMalloc(&d_running_mean_, param_size);
    cudaMalloc(&d_running_var_, param_size);

    std::vector<float> h_gamma(feature_size_, 1.0f);
    std::vector<float> h_beta(feature_size_, 0.0f);
    cudaMemcpy(d_gamma_, h_gamma.data(), param_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta_, h_beta.data(), param_size, cudaMemcpyHostToDevice);

    cudaMemset(d_running_mean_, 0, param_size);
    cudaMemset(d_running_var_, 0, param_size);
}

void BatchNorm::allocate_buffers() {
    cudaFree(d_output_);
    cudaFree(d_norm_);
    cudaMalloc(&d_output_, batch_size_ * feature_size_ * sizeof(float));
    cudaMalloc(&d_norm_, batch_size_ * feature_size_ * sizeof(float));
}

void BatchNorm::initialize_adam() {
    size_t param_size = feature_size_ * sizeof(float);

    cudaMalloc(&d_m_gamma_, param_size);
    cudaMalloc(&d_v_gamma_, param_size);
    cudaMalloc(&d_m_beta_, param_size);
    cudaMalloc(&d_v_beta_, param_size);

    cudaMemset(d_m_gamma_, 0, param_size);
    cudaMemset(d_v_gamma_, 0, param_size);
    cudaMemset(d_m_beta_, 0, param_size);
    cudaMemset(d_v_beta_, 0, param_size);
}

void BatchNorm::set_batch_size(int batch_size) {
    batch_size_ = batch_size;
    allocate_buffers();
}

void BatchNorm::forward() {
    int threads = 256;
    int blocks = (feature_size_ + threads - 1) / threads;

    compute_mean<<<blocks, threads>>>(d_input_, d_mean_, batch_size_, feature_size_);
    compute_variance<<<blocks, threads>>>(d_input_, d_mean_, d_var_, batch_size_, feature_size_);
    cudaDeviceSynchronize();

    int total_elements = batch_size_ * feature_size_;
    int forward_blocks = (total_elements + threads - 1) / threads;

    batchnorm_forward<<<forward_blocks, threads>>>(d_input_, d_mean_, d_var_, d_output_, d_norm_,
                                                   d_gamma_, d_beta_, batch_size_, feature_size_, epsilon_);
    cudaDeviceSynchronize();
}

void BatchNorm::backward(const float* d_output) {
    int threads = 256;
    int blocks = (feature_size_ + threads - 1) / threads;

    compute_gradients<<<blocks, threads>>>(d_norm_, d_output, d_d_gamma_, d_d_beta_, batch_size_, feature_size_);
    cudaDeviceSynchronize();
}

void BatchNorm::update_weights(float learning_rate, float beta1, float beta2, float epsilon, int timestep) {
    int threads = 256;
    int blocks = (feature_size_ + threads - 1) / threads;

    adam_update<<<blocks, threads>>>(d_gamma_, d_m_gamma_, d_v_gamma_, d_d_gamma_,
                                     learning_rate, beta1, beta2, epsilon, timestep, feature_size_);

    adam_update<<<blocks, threads>>>(d_beta_, d_m_beta_, d_v_beta_, d_d_beta_,
                                     learning_rate, beta1, beta2, epsilon, timestep, feature_size_);
    cudaDeviceSynchronize();
}
