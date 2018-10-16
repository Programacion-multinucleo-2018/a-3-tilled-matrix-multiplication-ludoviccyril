#pragma once

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

// in util.cu

void safely_call(cudaError err, const char *msg, const char *file_name,
                 const int line_number);
void fill_matrix(float *m);
void check_result(float *a, float *b);
void multiply_cpu(float *a, float *b, float *c);

// in main.cu

__global__ void multiply_gpu_tiling(float *a, float *b, float *c);

#define SAFE_CALL(call, msg) safely_call(call, msg, __FILE__, __LINE__)

#define N 2000
#define TILE_DIM 8
#define BLOCK_DIM 32
