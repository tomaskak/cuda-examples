
#include <stdio.h>
#include <stdlib.h>

__global__ void VecAdd(float *A, float *B, float *C, int X) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < X) {
    C[i] = A[i] + B[i];
  }
}

int main() {

  int N = 10000000000;

  size_t sz = N * sizeof(float);
  printf("sz=%lu\n", sz);

  float *A = (float *)malloc(sz);
  float *B = (float *)malloc(sz);
  float *C = (float *)malloc(sz);

  for (int i = 0; i < N; i++) {
    A[i] = 1.0;
    B[i] = 3.0;
    C[i] = 10.0;
  }

  float *d_A;
  cudaMalloc(&d_A, sz);
  float *d_B;
  cudaMalloc(&d_B, sz);
  float *d_C;
  cudaMalloc(&d_C, sz);

  cudaMemcpy(d_A, A, sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sz, cudaMemcpyHostToDevice);

  int threadsPerBlock = 1024;
  int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

  printf("threadsPerBlock=%d, blocks=%d", threadsPerBlock, blocks);
  VecAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

  cudaError_t rc = cudaMemcpy(C, d_C, sz, cudaMemcpyDeviceToHost);
  if (rc != cudaSuccess) {
    printf("\nfailed to cudaMemcpy, rc=%s\n", cudaGetErrorString(rc));
    return -1;
  }

  for (int i = 0; i < N; ++i) {
    if (C[i] != 4.0) {
      printf("\nFailure! value=%f, i=%d\n", C[i], i);
      printf("A[i]=%f, B[i]=%f\n", A[i], B[i]);
      return -1;
    }
  }

  printf("\nSuccess!\n");

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  free(A);
  free(B);
  free(C);

  return 0;
}
