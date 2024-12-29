#ifndef MULTICLASSIFICATION_HPP
#define MULTICLASSIFICATION_HPP

#include <cuda_runtime.h>
#include <string>

class MultiClassification {
public:
    MultiClassification(int num_classes, int batch_size, const std::string& activation = "softmax");
    ~MultiClassification();

    void forward();
    void compute_loss(const float* labels);
    void compute_accuracy(const float* labels);
    void backward(const float* labels);

    void set_batch_size(int batch_size);

    const float* get_scores() const { return d_scores_; }
    const float* get_loss() const { return &h_loss_; }
    float get_accuracy() const { return h_accuracy_; }
    void set_input(const float* input) { d_input_ = input; }

private:
    int num_classes_;
    int batch_size_;
    std::string activation_;
    float h_loss_;     // Host-side loss value
    float h_accuracy_; // Host-side accuracy

    // Device pointers
    float* d_input_;    // Input logits
    float* d_scores_;   // Predicted scores (softmax/sigmoid)
    float* d_loss_;     // Loss values for each sample
    float* d_d_input_;  // Gradients of loss w.r.t. logits

    void allocate_buffers();
    void compute_softmax(const float* input, float* output);
    void compute_sigmoid(const float* input, float* output);
};

#endif // MULTICLASSIFICATION_HPP
