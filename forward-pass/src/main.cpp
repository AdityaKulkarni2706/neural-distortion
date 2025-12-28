#include <iostream>
#include <vector>
#include "../include/NeuralDistortion.h"


int main() {
    NeuralDistortion pedal;

    // Test Inputs: Negative, Silence, Quiet, Loud
    std::vector<float> test_inputs = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};

    std::cout << "Neural Pedal Verification\n";
    for (float x : test_inputs) {
        float y = pedal.process(x);
        std::cout << "Input: " << x << " \t-> Output: " << y << std::endl;
    }
    
    return 0;
}
