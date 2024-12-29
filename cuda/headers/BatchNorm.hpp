#ifndef BATCHNORM_HPP
#define BATCHNORM_HPP

#include <cuda_runtime.h>

class BatchNorm {
public:
    BatchNorm(int feature_size, int batch_size, float epsilon = 1e-5);
    ~BatchNorm();

    void forward();
    void backward(const float* d_output);
    void update_weights(float learning_rate, float beta1, float beta2, float epsilon, int timestep);

    void set_batch_size(int batch_size);

    const float* get_output() const { return d_output_; }
    void set_input(const float* input) { d_input_ = input; }

private:
    int feature_size_;
    int batch_size_;
    float epsilon_;

    float* d_gamma_;    // Scale parameter
    float* d_beta_;     // Shift parameter
    float* d_running_mean_;
    float* d_running_var_;

    float* d_mean_;     // Batch mean
    float* d_var_;      // Batch variance
    float* d_input_;
    float* d_output_;
    float* d_norm_;     // Normalized input
    float* d_d_gamma_;  // Gradient of gamma
    float* d_d_beta_;   // Gradient of beta

    float* d_m_gamma_;  // First moment for gamma (Adam)
    float* d_v_gamma_;  // Second moment for gamma (Adam)
    float* d_m_beta_;   // First moment for beta (Adam)
    float* d_v_beta_;   // Second moment for beta (Adam)

    void allocate_buffers();
    void initialize_parameters();
    void initialize_adam();
};

#endif // BATCHNORM_HPP
