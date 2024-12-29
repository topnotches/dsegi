#ifndef DENSE_HPP
#define DENSE_HPP

#include <cuda_runtime.h>

class Dense {
public:
    Dense(int input_size, int output_size, int batch_size);
    ~Dense();

    void forward();
    void backward(const float* d_output);
    void update_weights(float learning_rate, float beta1, float beta2, float epsilon, int timestep);

    void set_batch_size(int batch_size);

    const float* get_output() const { return d_output_; }
    void set_input(const float* input) { d_input_ = input; }

private:
    int input_size_;
    int output_size_;
    int batch_size_;

    float* d_weights_;
    float* d_biases_;
    float* d_input_;
    float* d_output_;
    float* d_d_weights_;
    float* d_d_biases_;
    float* d_d_input_;

    void allocate_buffers();
    void initialize_parameters();
};

#endif // DENSE_HPP
