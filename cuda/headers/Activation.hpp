#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <cuda_runtime.h>
#include <string>

class Activation {
public:
    Activation(const std::string& activation_type, int size, int batch_size);
    ~Activation();

    void forward();
    void backward(const float* d_output);

    void set_batch_size(int batch_size);

    const float* get_output() const { return d_output_; }
    void set_input(const float* input) { d_input_ = input; }

private:
    std::string activation_type_;
    int size_;
    int batch_size_;

    float* d_input_;    // Input data pointer on the device
    float* d_output_;   // Output data pointer on the device
    float* d_d_input_;  // Gradients w.r.t. input on the device

    void allocate_buffers();
};

#endif // ACTIVATION_HPP
