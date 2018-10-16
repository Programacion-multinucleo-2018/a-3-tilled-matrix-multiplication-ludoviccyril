#pragma once

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <string>

using namespace std;

// in util.cu

void safely_call(cudaError err, const char *msg, const char *file_name,
                 const int line_number);
void fill_matrix(double *m);
void check_result(double *a, double *b);
void multiply_cpu(double *a, double *b, double *c);

// in main.cu

__global__ void multiply_gpu_tiling(double *a, double *b, double *c);

#define SAFE_CALL(call, msg) safely_call(call, msg, __FILE__, __LINE__)

#define N 2000
#define TILE_DIM 32
#define BLOCK_DIM 32
