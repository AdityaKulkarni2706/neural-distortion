# Real-Time Neural Audio Distortion Plugin

**A high-performance, zero-allocation neural inference engine for real-time analog-style audio distortion in JUCE plugins**  
Nov 2025 – Dec 2025

<img src="https://github.com/AdityaKulkarni2706/neural-distortion/blob/main/images/flat.png" alt="Plugin UI Screenshot" width="800"/>
<img src="https://github.com/AdityaKulkarni2706/neural-distortion/blob/main/images/naive.png" alt="Plugin UI Screenshot" width="800"/>
> From 77% → 21% CPU usage at 240 000 inferences/sec — real-time safe on modest hardware

## Overview

This project implements a **real-time capable neural network inference engine** specifically designed to emulate analog distortion circuits (overdrive, distortion, fuzz, etc.) inside a JUCE-based audio plugin.

The core achievement is a **zero-allocation forward pass**, which eliminates the risk of audio dropouts caused by heap operations — a critical requirement for professional real-time audio processing.

Two main architectures were developed and profiled:

- **NaiveLayer** — straightforward but allocation-heavy implementation  
- **FlatLayer** — heavily optimized, allocation-free version with custom math and memory layout

## Key Achievements

- **3.5× latency / CPU reduction** — from **~77%** to **~21%** CPU load (measured with Valgrind + KCachegrind)
- **Completely eliminated dynamic allocations** during the audio callback / inference loop
- Inference remains real-time safe even at high sample rates and low buffer sizes
- Clean, scalable architecture ready for integration into JUCE audio plugins (VST3 / AU / AAX / Standalone)

### Before vs After Optimization (CPU Flame Graph Summary)

**Naive version** (77.63% CPU in main forward pass):

- Heavy usage of `std::vector` with default allocator → frequent heap allocations
- Multiple nested `operator[]` and container operations eating cycles

**Optimized Flat version** (21.38% CPU total):

- Flat memory layout + pointer arithmetic
- Custom clamped tanh approximation (`taylor_tanh_input_clamped`)
- Minimal branching and almost no STL container overhead

```text
NaiveLayer:    77.6% CPU ─┬─ std::vector access & allocations (~20–21%)
                           └─ Many small heap ops

FlatLayer:     21.4% CPU ─┬─ Linear memory walk (~1.2%)
                           └─ Fast math approx (~1.3%)
