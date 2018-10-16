#include "main.h"

void safely_call(cudaError err, const char *msg, const char *file_name,
                 const int line_number) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg,
            file_name, line_number, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void fill_matrix(double *m) {
  for (int i = 0; i < N * N; i++) {
    m[i] = (double)rand() * 9 / RAND_MAX + 1;
  }
}

void check_result(double *a, double *b) {
  int are_identical = 1;
  for (int i = 0; i < N * N; i++) {
    if (abs(a[i] - b[i]) > 1.0) {
      are_identical = 0;
      cout << i << ": " << a[i] << " / " << b[i] << endl;
      break;
    }
  }
  if (are_identical) {
    cout << "Valid result." << endl;
  } else {
    cout << "Invalid result." << endl;
  }
}

void multiply_cpu(double *a, double *b, double *c) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      c[j * N + i] = 0;
      for (int k = 0; k < N; k++) {
        c[j * N + i] += a[j * N + k] * b[i + k * N];
      }
    }
  }
}