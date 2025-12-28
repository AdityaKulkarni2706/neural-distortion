#pragma once
#include <cmath>
#include "Weights.h"

class NeuralDistortion{

    public:
    NeuralDistortion(){}
    //fast tanh function : this function approximates tanh very well from 
    // [-3,3]

    inline float fast_tanh(float x) {
        if (x < -3.0f) return -1.0f;
        if (x > 3.0f) return 1.0f;
        float x2 = x * x;
        return x * (27.0f + x2) / (27.0f + 9.0f * x2);
    }

    inline float process(float input){
        float hidden_neurons[NeuralWeights::hidden_size];

        #pragma unroll

        //layer 1

        for(int i{0}; i < NeuralWeights::hidden_size; i++){
            float z = (NeuralWeights::w1[i] * input) * NeuralWeights::b1[i];
            hidden_neurons[i] = fast_tanh(z);
        }

        //layer 2

        float output = 0.0f;
        for (int i = 0; i < NeuralWeights::hidden_size; ++i) {
            output += hidden_neurons[i] * NeuralWeights::w2[i];
        }
        
        // Add final bias
        output += NeuralWeights::b2[0];
        return output;

    }

};