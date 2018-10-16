#include "main.h"

__global__ void multiply_gpu_tiling(double *a, double *b, double *c) {
  __shared__ double a_tile[TILE_DIM * TILE_DIM];
  __shared__ double b_tile[TILE_DIM * TILE_DIM];

  int n_tiles = (int)ceil((float)N / (float)TILE_DIM);

  int tile_row = blockIdx.x;
  int tile_col = blockIdx.y;

  int num_row = threadIdx.x;
  int num_col = threadIdx.y;

  double result = 0;

  for (int i = 0; i < n_tiles; i++) {
    if (i * TILE_DIM + num_col < N && tile_row * TILE_DIM + num_row < N) {
      a_tile[num_row * TILE_DIM + num_col] =
          a[tile_row * N * TILE_DIM + i * TILE_DIM + N * num_row + num_col];
    } else {
      a_tile[num_row * TILE_DIM + num_col] = 0;
    }

    if (i * TILE_DIM + num_col < N && tile_col * TILE_DIM + num_row < N) {
      b_tile[num_row * TILE_DIM + num_col] =
          b[tile_col * TILE_DIM + i * N * TILE_DIM + N * num_col + num_row];
    } else {
      b_tile[num_row * TILE_DIM + num_col] = 0;
    }

    __syncthreads();

    for (int j = 0; j < TILE_DIM; j++) {
      result += a_tile[num_row * TILE_DIM + j] * b_tile[num_col * TILE_DIM + j];
    }

    __syncthreads();
  }

  if (tile_col * TILE_DIM + num_col < N && tile_row * TILE_DIM + num_row < N) {
    c[tile_row * TILE_DIM * N + tile_col * TILE_DIM + num_col + num_row * N] =
        result;
  }
}

__global__ void multiply_gpu(double *a, double *b, double *c) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int idx = j * N + i;

  if (j < N && i < N) {
    double res = 0;
    for (int k = 0; k < N; k++) {
      res += a[j * N + k] * b[k * N + i];
    }
    c[idx] = res;
  }
}

int main(int argc, char *argv[]) {

  // set up device

  cudaDeviceProp deviceProp;
  SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0),
            "Error getting device properties");
  printf("Using device: %s\n", deviceProp.name);
  SAFE_CALL(cudaSetDevice(0), "Error setting device");

  // initialize matrices on host

  double *a = (double *)calloc(N * N, sizeof(double));
  double *b = (double *)calloc(N * N, sizeof(double));
  double *c = (double *)calloc(N * N, sizeof(double));
  double *d = (double *)calloc(N * N, sizeof(double));
  double *e = (double *)calloc(N * N, sizeof(double));

  fill_matrix(a);
  fill_matrix(b);

  // assign device global memory

  double *d_a, *d_b, *d_c, *d_d;
  SAFE_CALL(cudaMalloc((void **)&d_a, N * N * sizeof(double)),
            "Error allocating d_a");
  SAFE_CALL(cudaMalloc((void **)&d_b, N * N * sizeof(double)),
            "Error allocating d_b");
  SAFE_CALL(cudaMalloc((void **)&d_c, N * N * sizeof(double)),
            "Error allocating d_c");
  SAFE_CALL(cudaMalloc((void **)&d_d, N * N * sizeof(double)),
            "Error allocating d_d");

  // transfer data from host to device

  SAFE_CALL(cudaMemcpy(d_a, a, N * N * sizeof(double), cudaMemcpyHostToDevice),
            "Error copying a");
  SAFE_CALL(cudaMemcpy(d_b, b, N * N * sizeof(double), cudaMemcpyHostToDevice),
            "Error copying b");

  // configure grid & run multiply with tiling

  dim3 block_tile(TILE_DIM, TILE_DIM);
  dim3 grid_tile((N + block_tile.x - 1) / block_tile.x,
                 (N + block_tile.y - 1) / block_tile.y);

  auto start_gpu_tile = chrono::high_resolution_clock::now();

  multiply_gpu_tiling<<<grid_tile, block_tile>>>(d_a, d_b, d_c);
  SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");

  auto end_gpu_tile = chrono::high_resolution_clock::now();

  // check for kernel errors

  SAFE_CALL(cudaGetLastError(), "Error with last error");

  // configure grid & run multiply without tiling

  dim3 block(BLOCK_DIM, BLOCK_DIM);
  dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  auto start_gpu = chrono::high_resolution_clock::now();

  multiply_gpu<<<grid, block>>>(d_a, d_b, d_d);
  SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");

  auto end_gpu = chrono::high_resolution_clock::now();

  // check for kernel errors

  SAFE_CALL(cudaGetLastError(), "Error with last error");

  // copy results to host

  SAFE_CALL(cudaMemcpy(c, d_c, N * N * sizeof(double), cudaMemcpyDeviceToHost),
            "Error copying c");
  SAFE_CALL(cudaMemcpy(d, d_d, N * N * sizeof(double), cudaMemcpyDeviceToHost),
            "Error copying d");

  // free device global memory

  SAFE_CALL(cudaFree(d_a), "Error freeing memory");
  SAFE_CALL(cudaFree(d_b), "Error freeing memory");
  SAFE_CALL(cudaFree(d_c), "Error freeing memory");
  SAFE_CALL(cudaFree(d_d), "Error freeing memory");

  // reset device

  SAFE_CALL(cudaDeviceReset(), "Error resetting");

  // multiply on host

  multiply_cpu(a, b, e);

  // check results

  cout << "CHECKING RESULTS OBTAINED WITH TILING" << endl;
  check_result(c, e);

  cout << "CHECKING RESULTS OBTAINED WITHOUT TILING" << endl;
  check_result(d, e);

  // free host memory

  free(a);
  free(b);
  free(c);
  free(d);
  free(e);

  // print results

  chrono::duration<float, std::milli> duration_gpu_tile =
      end_gpu_tile - start_gpu_tile;
  chrono::duration<float, std::milli> duration_gpu = end_gpu - start_gpu;

  cout << "WITH TILING: " << duration_gpu_tile.count() << "ms" << endl;
  cout << "WITHOUT TILING: " << duration_gpu.count() << "ms" << endl;
  cout << "SPEEDUP: " << duration_gpu.count() / duration_gpu_tile.count()
       << endl;

  return 0;
}