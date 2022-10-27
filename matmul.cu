#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 16
#define CC 3
#define CD 2

__global__ void matmul(float *A, float *B, float *C, const int N, const int M) {
  int i = blockIdx.x * blockDim.x * CC + threadIdx.x;
  int j = blockIdx.y * blockDim.y * CD + threadIdx.y;

  size_t x_b_offset = blockIdx.x * blockDim.x * CC;
  size_t y_b_offset = blockIdx.y * blockDim.y * CD;

  if (i < N && j < M) {
    __shared__ float sh_A[TILE_SIZE][TILE_SIZE]; // reuse A value space while
                                                 // storing B values
    __shared__ float sh_B[TILE_SIZE][TILE_SIZE * CC];

    float outValues[CC][CD];
    for (size_t row = 0; row < CC; row++) {
      for (size_t col = 0; col < CD; col++) {
        outValues[row][col] = 0.0f;
      }
    }
    for (size_t tile = 0; tile < ((N + TILE_SIZE - 1) / TILE_SIZE); tile++) {
      size_t t_offset = tile * TILE_SIZE;

      for (size_t col_coarse = 0; col_coarse < CD; col_coarse++) {
        size_t c_offset = col_coarse * TILE_SIZE;

        if (c_offset + x_b_offset + threadIdx.x < N &&
            t_offset + threadIdx.y < M) {
          // first TILE_SIZE values of A rows
          sh_A[threadIdx.x][threadIdx.y] =
              A[(c_offset + x_b_offset + threadIdx.x) * M + t_offset +
                threadIdx.y];
        } else {
          sh_A[threadIdx.x][threadIdx.y] = 0.0f;
        }

        if (col_coarse != 0) {
          __syncthreads();
        }

        for (size_t row_coarse = 0; row_coarse < CC; row_coarse++) {
          size_t r_offset = row_coarse * TILE_SIZE;

          // Load once, reuse on next rows
          if (col_coarse == 0) {
            if (((r_offset + y_b_offset + threadIdx.y) < N) &&
                (t_offset + threadIdx.x < M)) {
              // first TILE_SIZE values of all B cols
              sh_B[threadIdx.x][threadIdx.y + r_offset] =
                  B[(y_b_offset + r_offset + threadIdx.y) * M + t_offset +
                    threadIdx.x];
            } else {
              sh_B[threadIdx.x][threadIdx.y + r_offset] = 0.0f;
            }

            __syncthreads();
          }

          for (size_t t = 0; t < TILE_SIZE; t++) {
            outValues[col_coarse][row_coarse] +=
                sh_A[threadIdx.x][t] * sh_B[t][r_offset + threadIdx.y];
          }

          __syncthreads();
        }
      }
    }

    for (size_t row = 0; row < CC; ++row) {
      for (size_t col = 0; col < CD; col++) {
        size_t xx = x_b_offset + row * TILE_SIZE + threadIdx.x;
        size_t yy = y_b_offset + col * TILE_SIZE + threadIdx.y;
        if (xx < N && yy < M) {
          C[xx * M + yy] = outValues[row][col];
        }
      }
    }
  }
}

#define CUDA_FN(fn, msg, ...)                                                  \
  {                                                                            \
    cudaError_t rc = fn(__VA_ARGS__);                                          \
    if (rc != cudaSuccess) {                                                   \
      printf("\n[failed run cuda fn %s], rc=%s\n", msg,                        \
             cudaGetErrorString(rc));                                          \
      return -1;                                                               \
    }                                                                          \
  }

int main(int argc, char **argv) {
  /*
    Create a matrix of size MxN and multiply them on the GPU. Check result for
    correct value.
   */

  long N = 1000;
  long M = 1000;

  if (argc > 1) {
    if (argc != 3) {
      printf("Wrong number of arguments, expected 0 or 2 got %d\n", argc - 1);
      return -1;
    }
    errno = 0;
    char *end;
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

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);

  printf("device properties: \n"
         "multiProcessorCount: %d\n"
         "sharedMemPerBlock: %lu\n"
         "maxThreadsPerBlock: %d\n"
         "regsPerBlock: %d\n"
         "warpSize: %d\n"
         "memoryBusWidth: %d\n"
         "maxThreadsPerMultiProcessor: %d\n",
         props.multiProcessorCount, props.sharedMemPerBlock,
         props.maxThreadsPerBlock, props.regsPerBlock, props.warpSize,
         props.memoryBusWidth, props.maxThreadsPerMultiProcessor);

  size_t sz = N * M * sizeof(float);
  printf("sz=%lu\n", sz);

  float *A = (float *)malloc(sz);
  float *B = (float *)malloc(sz);
  float *C = (float *)malloc(sz);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; ++j) {
      int idx = i * M + j;

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

  float *d_A;
  CUDA_FN(cudaMalloc, "initiaizing d_A", &d_A, sz);
  float *d_B;
  CUDA_FN(cudaMalloc, "initializing d_B", &d_B, sz);
  float *d_C;
  CUDA_FN(cudaMalloc, "initializing d_C", &d_C, sz);

  CUDA_FN(cudaMemcpy, "copy A to device", d_A, A, sz, cudaMemcpyHostToDevice);
  CUDA_FN(cudaMemcpy, "copy B to device", d_B, B, sz, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE, 1);
  dim3 blocks((N + TILE_SIZE * CD - 1) / (TILE_SIZE * CD),
              (M + TILE_SIZE * CC - 1) / (TILE_SIZE * CC), 1);

  printf("threadsPerBlock=%d,%d, blocks=%d,%d", threadsPerBlock.x,
         threadsPerBlock.y, blocks.x, blocks.y);
  matmul<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N, M);

  CUDA_FN(cudaMemcpy, "copy result to host", C, d_C, sz,
          cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      size_t idx = i * M + j;
      if (i == j && C[idx] != 1250.0 || i != j && C[idx] != 0.0) {
        printf("\nFailure! value=%f, idx=%d,%d\n", C[idx], i, j);
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
