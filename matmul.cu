#include <stdio.h>
#include <errno.h>
#include <stdlib.h>

__global__ void matmul(float *A, float *B, float *C, int N, int M) {
  int xx = blockIdx.x * blockDim.x + threadIdx.x;

  int i = xx / M;
  int j = xx % M;


  if (i < N && j < M) {
    for(int x = 0; x < N; ++x) {
      size_t idx = i*M + j;
      C[idx] += A[i*M + x] * B[x*M + j];
    }
  }
}

#define CUDA_FN(fn, msg, ...)				   \
  { \
    cudaError_t rc = fn(__VA_ARGS__);	\
    if (rc != cudaSuccess) { \
      printf("\n[failed run cuda fn %s], rc=%s\n", msg, cudaGetErrorString(rc));			\
      return -1; \
    } \
  } \

int main(int argc, char** argv) {
  /*
    Create a matrix of size MxN and multiply them on the GPU. Check result for correct value.
   */

  long N = 1000;
  long M = 1000;

  if (argc > 1) {
    if (argc != 3) {
      printf("Wrong number of arguments, expected 0 or 2 got %d\n", argc-1);
      return -1;
    }
    errno = 0;
    char * end;
    N = strtol(argv[1], &end, 10);
    if (errno != 0) {
      printf("errno received on parsing arg[1]: %d\n", errno);
      return -1;
    }

    M = strtol(argv[2], &end, 10);
    if (errno != 0) {
      printf("errno received on parsing arg[2]: %d\n", errno);
      return -1;
    }
  }

  size_t sz = N * M * sizeof(float);
  printf("sz=%lu\n", sz);
    
  float * A = (float *)malloc(sz);
  float * B = (float *)malloc(sz);
  float * C = (float *)malloc(sz);

  for (int i=0; i < N; i++){
    for (int j=0; j < M; ++j) {
      int idx = i*M + j;

      if (i == j) {
	A[idx] = 2.5;
        B[idx] = 500.0;
      } else {
	A[idx] = 0.0;
	B[idx] = 0.0;	
      }
      
      C[idx] = 33.3;
    }
  }

  float * d_A;
  CUDA_FN(cudaMalloc, "initiaizing d_A", &d_A, sz);
  float * d_B;
  CUDA_FN(cudaMalloc, "initializing d_B", &d_B, sz);
  float * d_C;
  CUDA_FN(cudaMalloc, "initializing d_C", &d_C, sz);

  CUDA_FN(cudaMemcpy, "copy A to device", d_A, A, sz, cudaMemcpyHostToDevice);
  CUDA_FN(cudaMemcpy, "copy B to device", d_B, B, sz, cudaMemcpyHostToDevice);
   

  int threadsPerBlock = 1024;
  int blocks = (N*M + threadsPerBlock - 1) / threadsPerBlock;

  printf("threadsPerBlock=%d, blocks=%d", threadsPerBlock, blocks);  
  matmul<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N, M);

  CUDA_FN(cudaMemcpy, "copy result to host", C, d_C, sz, cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; ++i){
    for (int j = 0; j < M; ++j){
      size_t idx = i*M + j;
      if (i == j && C[idx] != 1250.0 || i != j && C[idx] != 0.0) {
	printf("\nFailure! value=%f, idx=%lu\n", C[idx], idx);
	printf("A[idx]=%f, B[idx]=%f\n", A[idx], B[idx]);            
	return -1;
      }
    }
  }

  printf("\nSuccess!\n");

  CUDA_FN(cudaFree, "freeing device A", d_A);
  CUDA_FN(cudaFree, "freeing device B", d_B);
  CUDA_FN(cudaFree, "freeing device C", d_C);

  free(A);
  free(B);
  free(C);

  return 0;
}
