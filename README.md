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

```text
num_threads = 128
computed std dev is 288675134.6, sqrt((N^2-1)/12) is 288675134.6

real    0m1.644s
user    0m1.307s
sys     0m0.236s
```

N = 10,000,000,000 (10 billion) – overflow occurs due to double precision limits:

```text
computed std dev is 484971221.4, sqrt((N^2-1)/12) is 804481180.2
```

---

## Part 2: Multiple Thread Blocks

### Kernel Implementation (Part 2)

Two kernels are used:
- `sumKernel` – computes partial sums
- `sumdiffsqKernel` – computes sum of squared differences

Each thread processes `T` consecutive terms (except the last thread which handles the remainder).

```cpp
__global__ void sumKernel(uint64 N, uint64 T, uint64* sum)
{
    uint64 thread_num = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
    uint64 thread_sum = 0;
    uint64 start = thread_num * T;
    uint64 end = start + T;
    if (end > N) end = N;
    for (uint64 i = 1 + start; i <= end; i++)
        thread_sum += i;
    atomicAdd(sum, thread_sum);
}

__global__ void sumdiffsqKernel(uint64 N, uint64 T, double mean, double* sum_diff_sq)
{
    uint64 thread_num = (uint64)blockIdx.x * blockDim.x + threadIdx.x;
    double thread_sum_diff_sq = 0.0;
    uint64 start = thread_num * T;
    uint64 end = start + T;
    if (end > N) end = N;
    for (uint64 i = 1 + start; i <= end; i++)
        thread_sum_diff_sq += (i - mean) * (i - mean);
    atomicAdd(sum_diff_sq, thread_sum_diff_sq);
}
```

### Main Function and Block/Grid Calculation

The number of thread blocks `G` is computed as:
  ```cpp
  int G = (N + (B * T) - 1) / (B * T);
  ```
  where:
  - `N` = total numbers
  - `T` = terms per thread
  - `B` = threads per block (multiple of 32, &le; 1024)

### Results (Part 2)

N = 10,000,000,000 (10 billion), T = 1000, B = 128:

```text
N = 10000000000
terms per thread T = 1000
threads per block B = 128
number of thread blocks G = 78125
computed std dev is 288675134.6, sqrt((N^2-1)/12) is 288675134.6

real    0m0.614s
user    0m0.433s
sys     0m0.143s
```

### Performance Comparison

|Version | N = 10 billion | T (terms/thread) | Real Time (s) | Speedup vs Part 1 |
|--------|----------------|------------------|---------------|------------------|
| Part 1 | 10,000,000,000 | N/A | 2.842 | 1.0x |
| Part 2 | 10,000,000,000 | 1000 | 0.614 | 4.63x |

> **Note:** Part 2 with `T = 1` is slower than Part 1 because the overhead of launching many threads/atomic operations outweighs the parallelism benefit. Larger `T` increases work per thread and reduces global synchronisation overhead

---

## Discussion Questions & Answers

### Part 1 - Single Thread Block

**Q:** Explain how you made your kernel code thread safe and parallel efficient.
**A:**
- Shared memory for `sum` and `sum_diff_sq` is initialised by thread 0 and synchronised with `__syncthreads()`
- Each thread accumulates into a local variable (`thread_sum`, `thread_sum_diff_sq`) to reduce atomic contention
- `atomicAdd` safely combines local results into shared variables
- Three barriers ensure proper ordering: after initialisation, after sum accumulation, and after sum-of-squares accumulation
- Loop indexing uses `i = thread_num + 1; i <= N; i += num_threads` to flatten the workload evenly across threads

**Q:** What is the primary limitation of using a single thread block? Explain in terms of GPU hardware (SMs)
**A:** A single thread block can only utilise one streaming multiprocessor (SM) at a time. Modern GPUs have many SMs (e.g., 20+), so the rest remain idle. This underutilises the GPU’s parallel capacity, limiting speedup regardless of thread count

### Part 2 – Multiple Thread Blocks

**Q:** Explain how you made your kernel code thread-safe and parallel efficient.
A:
- Each thread processes a contiguous chunk of `T` terms, using `start = thread_num * T` and `end = start + T`.
- Local accumulators (`thread_sum`, `thread_sum_diff_sq`) are used per thread.
- `atomicAdd` updates global counters safely.
- The grid is sized so that nearly all terms are covered (`G = ceil(N / (B*T))`).
- Separate kernels for sum and sum-of-squares allow reuse of the computed mean.

**Q:** About how many times faster is version 2 (with T=1000) than version 1 for N = 10 billion?
**A:** Part 1 real time = 2.842 s, Part 2 real time = 0.614 s → speedup ≈ 4.63×.

**Q:** Why is version 2 slower than version 1 when T = 1, despite using all SMs?
**A:** With `T = 1`, each thread does almost no work, but the overhead of launching tens of thousands of thread blocks and performing atomic operations for every single element dominates the runtime. The cost of scheduling and synchronising outweighs the benefits of parallelism. Larger `T` amortises this overhead.

---

## How to Run

### 1. Clone the repository:
  ```bash
  git clone https://github.com/JasonBruno25/gpu-std-dev.git
  cd gpu-std-dev
  ```
### 2. Compile Part 1:
  ```bash
  nvcc -arch=sm_75 -o gpu_std_dev_v1 gpu_std_dev_v1.cu
  ./gpu_std_dev_v1 1000000000 128
  ```

### 3. Compile Part 2:
  ```bash
  nvcc -arch=sm_75 -o gpu_std_dev_v2 gpu_std_dev_v2.cu
  ./gpu_std_dev_v2 10000000000 1000 128
  ```
> Note: Adjust `-arch` to match your GPU (e.g., `sm_61`, `sm_80`). For Google Colab, `sm_75` works for Tesla T4.

---

## Repository Structure
```text
gpu-std-dev/
├── README.md
├── gpu_std_dev_v1.cu      # Single thread block version
├── gpu_std_dev_v2.cu      # Multiple thread block version
└── sequential/            # (Optional) CPU reference
    └── std_dev.c
```

























