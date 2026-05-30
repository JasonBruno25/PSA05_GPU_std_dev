# PSA05_GPU_std_dev

# GPU Standard Deviation – CMDA 3634 Parallel Programming

**Author:** Jason Bruno Terceros  
**Course:** CMDA 3634 (Spring 2024)  
**Assignment:** Parallel Programming Skills Activity – GPU Standard Deviation  

This repository contains CUDA implementations of the standard deviation of the first `N` natural numbers using a single thread block (Part 1) and multiple thread blocks (Part 2). The goal is to accelerate the computation using NVIDIA GPUs and compare performance.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Part 1: Single Thread Block](#part-1-single-thread-block)
  - [Kernel Implementation](#kernel-implementation)
  - [Compilation and Execution](#compilation-and-execution)
  - [Results](#results-part-1)
- [Part 2: Multiple Thread Blocks](#part-2-multiple-thread-blocks)
  - [Kernel Implementation](#kernel-implementation-part-2)
  - [Main Function and Block/Grid Calculation](#main-function-and-blockgrid-calculation)
  - [Results](#results-part-2)
- [Performance Comparison](#performance-comparison)
- [Discussion Questions & Answers](#discussion-questions--answers)
- [How to Run](#how-to-run)
- [Repository Structure](#repository-structure)
- [Academic Integrity & Honor Code](#academic-integrity--honor-code)

---

## Overview

The standard deviation of the numbers `1..N` is given by:

`σ = √((N² - 1) / 12)`


A sequential CPU implementation is provided for reference. This project implements two CUDA versions:

- **Version 1** – Uses a single thread block (`<<<1, num_threads>>>`).
- **Version 2** – Uses multiple thread blocks, each handling `T` terms per thread, to better utilise GPU streaming multiprocessors (SMs).

All code is written in CUDA C and tested on Google Colab with GPU runtime.

---

## Requirements

- NVIDIA GPU with Compute Capability ≥ 7.5 (for `-arch=sm_75`)
- CUDA Toolkit (nvcc)
- GCC or compatible C compiler
- Google Colab (optional, but used for development)

---

## Part 1: Single Thread Block

### Kernel Implementation

The kernel `stdevKernel` uses:
- Shared memory for `sum` and `sum_diff_sq`
- `__syncthreads()` for synchronisation (three barriers)
- `atomicAdd` for safe accumulation across threads

**Key features:**
- Threads loop with stride `num_threads` over the range `1..N`
- Thread 0 initialises shared variables
- Final standard deviation printed by thread 0

```cpp
__global__ void stdevKernel(uint64 N)
{
    int thread_num = threadIdx.x;
    int num_threads = blockDim.x;

    __shared__ uint64 sum;
    __shared__ double sum_diff_sq;

    if (thread_num == 0) {
        sum = 0;
        sum_diff_sq = 0.0;
    }
    __syncthreads();

    uint64 thread_sum = 0;
    for (uint64 i = 1 + thread_num; i <= N; i += num_threads)
        thread_sum += i;
    atomicAdd(&sum, thread_sum);
    __syncthreads();

    double mean = (1.0) * sum / N;

    double thread_sum_diff_sq = 0.0;
    for (uint64 i = 1 + thread_num; i <= N; i += num_threads)
        thread_sum_diff_sq += (i - mean) * (i - mean);
    atomicAdd(&sum_diff_sq, thread_sum_diff_sq);
    __syncthreads();

    if (thread_num == 0) {
        printf("computed std dev is %.1lf, sqrt((N^2-1)/12) is %.1lf\n",
               sqrt(sum_diff_sq / N), sqrt((N * N - 1) / 12.0));
    }
}
```
### Compilation and Execution
  ```bash
  nvcc -arch=sm_75 -o gpu_std_dev_v1 gpu_std_dev_v1.cu
  ./gpu_std_dev_v1 <N> <num_threads>
  ```

### Results (Parts 1)

`num_threads = 128`
> computed std dev is 288675134.6, `sqrt((N^2-1)/12)` is 288675134.6

| value | results |
|-----|-----------|
| real | 0m1.644s |
| user | 0m1.307s |
| sys | 0m0.236s |

N = 10,000,000,000 (10 billion) – overflow occurs due to double precision limits:
> computed std dev is 484971221.4, sqrt((N^2-1)/12) is 804481180.2

---

## Part 2: Multiple Thread Blocks

### Kernel Implementation (Part 2)
Two kernels are used:

sumKernel – computes partial sums

sumdiffsqKernel – computes sum of squared differences

Each thread processes T consecutive terms (except the last thread which handles the remainder).
















