# Real-Time Neural Audio Distortion Plugin
**A zero-allocation, analog-style neural inference engine for JUCE audio plugins.**

![Platform](https://img.shields.io/badge/platform-JUCE%20%7C%20Windows%20%7C%20Linux-lightgrey)
![Language](https://img.shields.io/badge/language-C%2B%2B17%20%7C%20Python-blue)
![License](https://img.shields.io/badge/license-MIT-green)

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
