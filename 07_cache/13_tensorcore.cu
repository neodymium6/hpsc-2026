#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <mma.h>
#include <cmath>
#include <chrono>
using namespace std;
using namespace nvcuda;

// ---- tile shape (file scope so main() can size the dynamic shared memory) ----
constexpr int TILE_M  = 128;
constexpr int TILE_N  = 128;
constexpr int TILE_K  = 64;
constexpr int PAD     = 8;   // shared-memory padding to avoid bank conflicts (keeps 16B alignment)
constexpr int VEC     = 8;   // 8 half = 16 B = one cp.async transaction
constexpr int WMMA_M  = 16;
constexpr int WMMA_N  = 16;
constexpr int WARPS_N = 2;
constexpr int A_BUF   = TILE_K * (TILE_M + PAD);     // half per A sub-buffer
constexpr int B_BUF   = TILE_N * (TILE_K + PAD);     // half per B sub-buffer
constexpr int SMEM_HALFS = 2 * A_BUF + 2 * B_BUF;    // double-buffered total

__global__ void convert_to_half(int size, const float *src, half *dst) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    dst[idx] = __float2half(src[idx]);
}

// Issue async global->shared copies for the A and B tiles at global column kbase
// into shared buffer `buf`. Assumes tile-aligned dims (m,n,k multiples of the tiles),
// which holds for the fixed problem size below, so no bounds checks are needed.
__device__ __forceinline__ void load_tile(half (*block_a)[TILE_M + PAD],
                                          half (*block_b)[TILE_K + PAD],
                                          const half *d_a, const half *d_b,
                                          int dim_m, int dim_k,
                                          int offset_a_m, int offset_b_n,
                                          int buf, int kbase, int tid) {
  for (int idx = tid; idx < TILE_K * TILE_M / VEC; idx += blockDim.x) {
    int kk = idx / (TILE_M / VEC);
    int mm = (idx % (TILE_M / VEC)) * VEC;
    __pipeline_memcpy_async(&block_a[buf * TILE_K + kk][mm],
                            &d_a[(kbase + kk) * dim_m + offset_a_m + mm], 16);
  }
  for (int idx = tid; idx < TILE_N * TILE_K / VEC; idx += blockDim.x) {
    int nn = idx / (TILE_K / VEC);
    int kk = (idx % (TILE_K / VEC)) * VEC;
    __pipeline_memcpy_async(&block_b[buf * TILE_N + nn][kk],
                            &d_b[(offset_b_n + nn) * dim_k + kbase + kk], 16);
  }
}

__global__ void kernel(int dim_m, int dim_n, int dim_k,
		       const half *d_a, const half *d_b, float *d_c) {
  extern __shared__ __align__(16) half smem[];
  half (*block_a)[TILE_M + PAD] = reinterpret_cast<half(*)[TILE_M + PAD]>(smem);              // [2*TILE_K][TILE_M+PAD]
  half (*block_b)[TILE_K + PAD] = reinterpret_cast<half(*)[TILE_K + PAD]>(smem + 2 * A_BUF);  // [2*TILE_N][TILE_K+PAD]

  // Threadblock swizzle: remap blocks so that temporally-adjacent blocks share
  // A/B panels, improving L2 reuse. Bijection over all (bm,bn) -> correctness
  // unchanged. Assumes gridDim.x is a multiple of SWIZZLE (80 = 10*8 here).
  constexpr int SWIZZLE = 8;
  int bid = blockIdx.y * gridDim.x + blockIdx.x;
  int blocks_n = gridDim.y;
  int blocks_per_group = SWIZZLE * blocks_n;
  int group = bid / blocks_per_group;
  int gidx = bid % blocks_per_group;
  int bm = group * SWIZZLE + (gidx % SWIZZLE);
  int bn = gidx / SWIZZLE;
  int offset_a_m = TILE_M * bm;
  int offset_b_n = TILE_N * bn;
  int tid = threadIdx.x;
  int warp_id = threadIdx.x / 32;
  int warp_m = warp_id / WARPS_N;
  int warp_n = warp_id % WARPS_N;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
  for (int r = 0; r < 2; r++)
    for (int c = 0; c < 4; c++)
      wmma::fill_fragment(acc[r][c], 0.0f);

  int num_k = dim_k / TILE_K;

  // prologue: prefetch the first K-tile into buffer 0
  load_tile(block_a, block_b, d_a, d_b, dim_m, dim_k, offset_a_m, offset_b_n, 0, 0, tid);
  __pipeline_commit();

  for (int kt = 0; kt < num_k; kt++) {
    int cur = kt & 1;
    __pipeline_wait_prior(0);   // current buffer is loaded
    __syncthreads();            // make it visible to all warps

    if (kt + 1 < num_k) {       // prefetch next tile into the other buffer (overlaps the compute below)
      load_tile(block_a, block_b, d_a, d_b, dim_m, dim_k,
                offset_a_m, offset_b_n, (kt + 1) & 1, (kt + 1) * TILE_K, tid);
      __pipeline_commit();
    }

    for (int kk = 0; kk < TILE_K; kk += 16) {
#pragma unroll
      for (int r = 0; r < 2; r++) {
        int row_tile = warp_m * 2 + r;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
        wmma::load_matrix_sync(a_frag, &block_a[cur * TILE_K + kk][row_tile * WMMA_M], TILE_M + PAD);
#pragma unroll
        for (int c = 0; c < 4; c++) {
          int col_tile = warp_n * 4 + c;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
          wmma::load_matrix_sync(b_frag, &block_b[cur * TILE_N + col_tile * WMMA_N][kk], TILE_K + PAD);
          wmma::mma_sync(acc[r][c], a_frag, b_frag, acc[r][c]);
        }
      }
    }
  }
  for (int r = 0; r < 2; r++) {
    for (int c = 0; c < 4; c++) {
      int row_tile = warp_m * 2 + r;
      int col_tile = warp_n * 4 + c;
      int c_m = offset_a_m + row_tile * WMMA_M;
      int c_n = offset_b_n + col_tile * WMMA_N;
      if (c_n < dim_n && c_m < dim_m)
        wmma::store_matrix_sync(&d_c[c_n * dim_m + c_m], acc[r][c], dim_m, wmma::mem_col_major);
    }
  }
}

int main(int argc, const char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 8192;
  float alpha = 1.0;
  float beta = 0.0;
  int Nt = 10;
  float *A, *B, *C, *C2;
  half *Ahalf, *Bhalf;
  cudaMallocManaged(&A, m * k * sizeof(float));
  cudaMallocManaged(&B, k * n * sizeof(float));
  cudaMallocManaged(&C, m * n * sizeof(float));
  cudaMallocManaged(&C2, m * n * sizeof(float));
  cudaMallocManaged(&Ahalf, m * k * sizeof(half));
  cudaMallocManaged(&Bhalf, k * n * sizeof(half));
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A[k*i+j] = drand48();
  for (int i=0; i<k; i++)
    for (int j=0; j<n; j++)
      B[n*i+j] = drand48();
  for (int i=0; i<n; i++)
    for (int j=0; j<m; j++)
      C[m*i+j] = C2[m*i+j] = 0;

  int block_size = 256;
  convert_to_half<<<(m * k + block_size - 1) / block_size, block_size>>>(m * k, A, Ahalf);
  convert_to_half<<<(k * n + block_size - 1) / block_size, block_size>>>(k * n, B, Bhalf);
  cudaDeviceSynchronize();

  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  auto tic = chrono::steady_clock::now();
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    cublasGemmEx(cublas_handle,
		 CUBLAS_OP_N,
		 CUBLAS_OP_N,
		 m,
		 n,
		 k,
		 &alpha,
		 Ahalf, CUDA_R_16F, m,
		 Bhalf, CUDA_R_16F, k,
		 &beta,
		 C, CUDA_R_32F, m,
		 CUBLAS_COMPUTE_32F,
		 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
  }
  auto toc = chrono::steady_clock::now();
  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
  double cublas_flops = double(num_flops) / tcublas / 1.0e9;
  int tile = 128;
  dim3 block = dim3(256);
  dim3 grid = dim3((m+tile-1)/tile, (n+tile-1)/tile);
  int shmem_bytes = SMEM_HALFS * sizeof(half);
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes);
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    kernel<<< grid, block, shmem_bytes >>>(m,
				      n,
				      k,
				      Ahalf,
				      Bhalf,
				      C2);
    cudaDeviceSynchronize();
  }
  toc = chrono::steady_clock::now();
  double twmma = chrono::duration<double>(toc - tic).count() / Nt;
  double wmma_flops = double(num_flops) / twmma / 1.0e9;
  printf("CUBLAS: %.2f Gflops, WMMA: %.2f Gflops\n", cublas_flops, wmma_flops);
  double err = 0;
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      err += fabs(C[m*i+j] - C2[m*i+j]);
    }
  }
  printf("error: %lf\n", err/n/m);
  cudaFree(A);
  cudaFree(B);
  cudaFree(Ahalf);
  cudaFree(Bhalf);
  cudaFree(C);
  cudaFree(C2);
  cublasDestroy(cublas_handle);
}
