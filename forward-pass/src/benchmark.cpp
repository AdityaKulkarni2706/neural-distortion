#include <iostream>
#include "benchmark.h"
#include "Weights.h"

#include <algorithm>

inline float taylor_tanh_input_clamped(float x) {
    // Force x to stay within the region where the polynomial behaves nicely
    if (x < -3.0f) return -1.0f;
    if (x > 3.0f) return 1.0f;
    
    // 2. Rational Pade Approximation (Very accurate for -3 to 3)
    // Formula: x * (27 + x^2) / (27 + 9x^2)
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

void FlatLayer::forward(const float* input, float* output){
    for(int i{0}; i < currentLayerSize; i++){
        float z = biases[i];
        const float* rowStart = &weights[i*previousLayerSize];
    
        for (int j{0}; j < previousLayerSize; j++){
            z += rowStart[j] * input[j];
        }
        // output[i] = std::tanh(z);
        output[i] = taylor_tanh_input_clamped(z);

    }
}

std::vector<float> NaiveLayer::forward(const std::vector<float>& input){
    std::vector<float> output(this->currentLayerSize, 0);

    //ORIGINAL CODE
    
    // for(int i{0}; i < output.size(); i++){
    //     float z = biases[i];
    //     for(int j{0}; j<previousLayerSize; j++){
    //         z += weights[i][j] * input[j];
    //     }
    //     output[i] = std::tanh(z);

    // }


    // MODIFICATION 1 : To test out of changing output[i] changes anything much
    //turns out it doesnt XD


    for(int i{0}; i < output.size(); i++){

        for(int j{0}; j<previousLayerSize; j++){
            output[i] += weights[i][j] * input[j];
        }
        // output[i] = std::tanh(output[i]);
        output[i] = taylor_tanh_input_clamped(output[i]);
        


    }


    return output;

}

int main(){
    const int NUM_SAMPLES = 48000 * 5;
    const int INPUT_SIZE = 16;
    const int HIDDEN_SIZE = 16;

    std::vector<float> input(INPUT_SIZE, 0.5f);
    std::vector<float> outputBuffer(HIDDEN_SIZE);

    std::cout << "Benchmarking " << NUM_SAMPLES << " forward passes...\n";

    //TEST NAIVE
    {
        NaiveLayer naive(INPUT_SIZE, HIDDEN_SIZE);
        auto start = std::chrono::high_resolution_clock::now();
        
        for(int i=0; i<NUM_SAMPLES; ++i) {
            // Volatile prevents compiler from optimizing away the loop
            volatile auto result = naive.forward(input);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        std::cout << "Naive (Vector<Vector>): " << ms << " ms\n";
    }

    // TEST FLAT 
    {
        FlatLayer flat(INPUT_SIZE, HIDDEN_SIZE);
        auto start = std::chrono::high_resolution_clock::now();
        
        for(int i=0; i<NUM_SAMPLES; ++i) {
            flat.forward(input.data(), outputBuffer.data());
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        std::cout << "Optimized (Flat):      " << ms << " ms\n";
    }

    return 0;
}