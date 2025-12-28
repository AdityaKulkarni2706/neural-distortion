#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>

#include <iomanip>


class FlatLayer{
    public:
    std::vector<float> weights; 
    std::vector<float> biases;
    int currentLayerSize, previousLayerSize;

    FlatLayer(int in, int out) : currentLayerSize(out), previousLayerSize(in){
        weights.resize(in*out, 0.01f);
        biases.resize(out, 0.01f);
    }

    void forward(const float* input, float* output);
    

};


class NaiveLayer{
    public:

    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
    int currentLayerSize, previousLayerSize;

    NaiveLayer(int in, int out) : currentLayerSize(out), previousLayerSize(in){
        weights.resize(currentLayerSize, std::vector<float> (previousLayerSize));
        biases.resize(currentLayerSize);

        for(int i{0}; i<out; i++){
            biases[i] = 0.01f;
            for (int j{0}; j < in; j++){
                weights[i][j] = 0.01f;
            }
        }

    }

    std::vector<float> forward(const std::vector<float>& input);


};


std::ostream& operator<<(std::ostream& os, const NaiveLayer& layer) {
    os << std::fixed << std::setprecision(4);

    os << "NaiveLayer\n";
    os << "  previousLayerSize = " << layer.previousLayerSize << "\n";
    os << "  currentLayerSize  = " << layer.currentLayerSize << "\n\n";

    os << "  Weights:\n";
    for (int i = 0; i < layer.currentLayerSize; ++i) {
        os << "    [ ";
        for (int j = 0; j < layer.previousLayerSize; ++j) {
            os << std::setw(8) << layer.weights[i][j] << " ";
        }
        os << "]\n";
    }

    os << "\n  Biases:\n";
    os << "    [ ";
    for (int i = 0; i < layer.currentLayerSize; ++i) {
        os << std::setw(8) << layer.biases[i] << " ";
    }
    os << "]\n";

    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i) {
        os << v[i];
        if (i != v.size() - 1)
            os << ", ";
    }
    os << "]\n";
    return os;
}