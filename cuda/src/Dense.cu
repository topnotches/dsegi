#include "Dense.hpp"
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>

__global__ void dense_forward(const float* input, const float* weights, const float* biases, float* output, int input_size, int output_size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * output_size) {
        int batch_idx = idx / output_size;
        int output_idx = idx % output_size;
        float sum = biases[output_idx];
        for (int i = 0; i < input_size; i++) {
            sum += input[batch_idx * input_size + i] * weights[output_idx * input_size + i];
        }
        output[idx] = sum;
    }
}

__global__ void dense_backward_weights(const float* input, const float* d_output, float* d_weights, int input_size, int output_size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_size * output_size) {
        int row = idx / input_size;
        int col = idx % input_size;
        float grad = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            grad += input[i * input_size + col] * d_output[i * output_size + row];
        }
        d_weights[idx] += grad;
    }
}

__global__ void adam_update(
    float* param, float* m, float* v, const float* grad,
    float learning_rate, float beta1, float beta2, float epsilon,
    int timestep, int size
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        // Update first and second moments
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad[idx];
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad[idx] * grad[idx];

        // Correct bias in moments
        float m_hat = m[idx] / (1.0f - powf(beta1, timestep));
        float v_hat = v[idx] / (1.0f - powf(beta2, timestep));

        // Update parameter
        param[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

Dense::Dense(int input_size, int output_size, int batch_size)
    : input_size_(input_size), output_size_(output_size), batch_size_(batch_size) {
    initialize_parameters();
    allocate_buffers();
}

Dense::~Dense() {
    cudaFree(d_weights_);
    cudaFree(d_biases_);
    cudaFree(d_output_);
    cudaFree(d_d_weights_);
    cudaFree(d_d_biases_);
    cudaFree(d_d_input_);
}

void Dense::initialize_parameters() {
    size_t weights_size = input_size_ * output_size_ * sizeof(float);
    size_t biases_size = output_size_ * sizeof(float);

    cudaMalloc(&d_weights_, weights_size);
    cudaMalloc(&d_biases_, biases_size);

    std::vector<float> h_weights(input_size_ * output_size_);
    std::vector<float> h_biases(output_size_);

    for (auto& w : h_weights) w = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    for (auto& b : h_biases) b = 0.0f;

    cudaMemcpy(d_weights_, h_weights.data(), weights_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases_, h_biases.data(), biases_size, cudaMemcpyHostToDevice);
}

void Dense::allocate_buffers() {
    cudaFree(d_output_);
    cudaFree(d_d_input_);
    cudaMalloc(&d_output_, batch_size_ * output_size_ * sizeof(float));
    cudaMalloc(&d_d_input_, batch_size_ * input_size_ * sizeof(float));
}

void Dense::set_batch_size(int batch_size) {
    batch_size_ = batch_size;
    allocate_buffers();
}

void Dense::forward() {
    int threads = 256;
    int blocks = (batch_size_ * output_size_ + threads - 1) / threads;
    dense_forward<<<blocks, threads>>>(d_input_, d_weights_, d_biases_, d_output_, input_size_, output_size_, batch_size_);
    cudaDeviceSynchronize();
}

void Dense::backward(const float* d_output) {
    cudaMemcpy(d_d_input_, d_output, batch_size_ * output_size_ * sizeof(float), cudaMemcpyDeviceToDevice);

    int threads = 256;
    int blocks = (input_size_ * output_size_ + threads - 1) / threads;
    dense_backward_weights<<<blocks, threads>>>(d_input_, d_output, d_d_weights_, input_size_, output_size_, batch_size_);
    cudaDeviceSynchronize();
}
void Dense::update_weights(float learning_rate, float beta1, float beta2, float epsilon, int timestep) {
    // Allocate memory for Adam variables (if not already allocated)
    static bool initialized = false;
    if (!initialized) {
        cudaMalloc(&d_m_weights_, input_size_ * output_size_ * sizeof(float));
        cudaMalloc(&d_v_weights_, input_size_ * output_size_ * sizeof(float));
        cudaMalloc(&d_m_biases_, output_size_ * sizeof(float));
        cudaMalloc(&d_v_biases_, output_size_ * sizeof(float));

        cudaMemset(d_m_weights_, 0, input_size_ * output_size_ * sizeof(float));
        cudaMemset(d_v_weights_, 0, input_size_ * output_size_ * sizeof(float));
        cudaMemset(d_m_biases_, 0, output_size_ * sizeof(float));
        cudaMemset(d_v_biases_, 0, output_size_ * sizeof(float));

        initialized = true;
    }

    // Update weights
    int threads = 256;
    int weight_blocks = (input_size_ * output_size_ + threads - 1) / threads;
    adam_update<<<weight_blocks, threads>>>(
        d_weights_, d_m_weights_, d_v_weights_, d_d_weights_,
        learning_rate, beta1, beta2, epsilon, timestep, input_size_ * output_size_
    );

    // Update biases
    int bias_blocks = (output_size_ + threads - 1) / threads;
    adam_update<<<bias_blocks, threads>>>(
        d_biases_, d_m_biases_, d_v_biases_, d_d_biases_,
        learning_rate, beta1, beta2, epsilon, timestep, output_size_
    );

    cudaDeviceSynchronize();
}
