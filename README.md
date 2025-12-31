# Real-Time Neural Audio Distortion Plugin
**A zero-allocation, analog-style neural inference engine for JUCE audio plugins.**

## Demo

[![Neural Distortion Demo](https://img.youtube.com/vi/wBRHrmhN0mg/maxresdefault.jpg)](https://youtu.be/wBRHrmhN0mg)

*Click to watch: Real-time neural distortion processing with optimized CPU usage*

---

## Overview
This project implements a lightweight neural network inference engine in C++ designed to emulate analog distortion circuits (overdrive, fuzz) in real-time.

Unlike standard deep learning deployments (ONNX/TensorFlow), this engine is built from scratch for **audio-safety**:
* **Zero-Allocation:** No `malloc`/`new` in the audio callback (prevents dropouts/glitches).
* **Low Latency:** Optimized flat-array memory layout for maximum cache locality.
* **Dependency-Free:** Header-only inference (no heavy external libraries).

### Distortion(Clipping) Graph
*Input (Clean Sine) vs Output (Neural Hard Clipping)* ![Distortion Plot](https://github.com/AdityaKulkarni2706/neural-distortion/blob/main/images/distortion_demo.png)

---

##  Performance Optimization
The core engineering challenge was eliminating the overhead of `std::vector` dynamic allocations during the audio callback. Using **Valgrind** and **KCachegrind**, I profiled the instruction cost and optimized the forward pass.

### The Results
| Metric | Naive Implementation | Optimized (Flat) Implementation | Improvement |
| :--- | :--- | :--- | :--- |
| **CPU Load** | 77.63% | 21.38% | **3.6x Reduction** |
| **Instructions** | ~2 Billion | ~1 Billion | **50% Reduction** |
| **Allocation** | Heavy (`std::vector` resize) | **Zero** (Pre-allocated buffers) | **Real-Time Safe** |

### Visual Profiling Analysis
**1. Before: Naive Layer (Heavy Overhead)**
*Visible bottleneck in `std::vector::operator[]` and heap allocation.*
![Naive Graph](https://github.com/AdityaKulkarni2706/neural-distortion/blob/main/images/naive.png)

**2. After: Flat Layer (Pure Math)**
*Overhead eliminated. CPU time is spent almost entirely on DSP math.*
![Flat Graph](https://github.com/AdityaKulkarni2706/neural-distortion/blob/main/images/flat.png)

---

## Architecture & Code Comparison

The optimization involved moving from pointer-chasing (vector of vectors) to cache-friendly pointer arithmetic.

### 1. Naive Approach (Slow)
Standard bounds checking and dynamic resizing kill performance in the audio thread.
```cpp
// Bad for Audio: Heap allocation inside the loop!
std::vector<float> output(layerSize); 

for(int i=0; i < layerSize; ++i) {
    // Pointer chasing overhead (vector inside vector)
    output[i] += weights[i][j] * input[j]; 
}
```
### 2. Flat Approach (Fast)

Using raw pointers and flattened arrays ensures contiguous memory access (Spatial Locality).

```
// Good for Audio: Zero allocation, simple pointer arithmetic
// input/output buffers are pre-allocated at initialization
for(int i=0; i < layerSize; ++i) {
    // Sequential access = Happy CPU Cache
    const float* row = &weights[i * prevLayerSize]; 
    z += row[j] * input[j]; 
}
// Custom clamped tanh approximation for speed
output[i] = taylor_tanh_input_clamped(z);
```
### Workflow
1) Training (Python): PyTorch model trains on raw audio samples (sine sweeps/guitar DI)
2) Export: Python script extracts weights to a C++ header (Weights.h).
3) Inference (C++): The plugin compiles the weights directly into the binary for instant load times.

### Build Instructions
```
# Clone the repository
git clone https://github.com/AdityaKulkarni2706/neural-distortion.git

# Build the Benchmark tool
cd neural-distortion/src
g++ -O3 -o benchmark benchmark.cpp
./benchmark
```

