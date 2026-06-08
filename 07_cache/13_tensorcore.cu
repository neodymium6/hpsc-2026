#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include <chrono>
using namespace std;
using namespace nvcuda;

__global__ void convert_to_half(int size, const float *src, half *dst) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    dst[idx] = __float2half(src[idx]);
}

__global__ void kernel(int dim_m, int dim_n, int dim_k,
		       const half *d_a, const half *d_b, float *d_c) {
  constexpr int TILE_M = 128;
  constexpr int TILE_N = 128;
  constexpr int TILE_K = 64;
  constexpr int PAD = 8;
  constexpr int VEC = 8;
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WARPS_N = 2;

  int offset_a_m = TILE_M * blockIdx.x;
  int offset_b_n = TILE_N * blockIdx.y;
  int tid = threadIdx.x;
  int warp_id = threadIdx.x / 32;

  __shared__ half __align__(16) block_a[TILE_K][TILE_M + PAD];
  __shared__ half __align__(16) block_b[TILE_N][TILE_K + PAD];
  struct __align__(16) half8_t { half v[VEC]; };

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
  for (int r = 0; r < 2; r++)
    for (int c = 0; c < 4; c++)
      wmma::fill_fragment(acc[r][c], 0.0f);

  int warp_m = warp_id / WARPS_N;
  int warp_n = warp_id % WARPS_N;

  for (int k = 0; k < dim_k; k += TILE_K) {
    __syncthreads();

    for (int idx = tid; idx < TILE_K * TILE_M / VEC; idx += blockDim.x) {
      int kk = idx / (TILE_M / VEC);
      int mm = (idx % (TILE_M / VEC)) * VEC;
      int global_m = offset_a_m + mm;
      int global_k = k + kk;
      if (global_m + VEC <= dim_m && global_k < dim_k) {
        *reinterpret_cast<half8_t*>(&block_a[kk][mm]) =
          *reinterpret_cast<const half8_t*>(&d_a[global_k * dim_m + global_m]);
      } else {
        for (int v = 0; v < VEC; v++) {
          int m = global_m + v;
          block_a[kk][mm + v] = (m < dim_m && global_k < dim_k)
            ? d_a[global_k * dim_m + m]
            : __float2half(0.0f);
        }
      }
    }

    for (int idx = tid; idx < TILE_N * TILE_K / VEC; idx += blockDim.x) {
      int nn = idx / (TILE_K / VEC);
      int kk = (idx % (TILE_K / VEC)) * VEC;
      int global_n = offset_b_n + nn;
      int global_k = k + kk;
      if (global_n < dim_n && global_k + VEC <= dim_k) {
        *reinterpret_cast<half8_t*>(&block_b[nn][kk]) =
          *reinterpret_cast<const half8_t*>(&d_b[global_n * dim_k + global_k]);
      } else {
        for (int v = 0; v < VEC; v++) {
          int kk_global = global_k + v;
          block_b[nn][kk + v] = (global_n < dim_n && kk_global < dim_k)
            ? d_b[global_n * dim_k + kk_global]
            : __float2half(0.0f);
        }
      }
    }

    __syncthreads();

    for (int kk = 0; kk < TILE_K; kk += 16) {
#pragma unroll
      for (int r = 0; r < 2; r++) {
        int row_tile = warp_m * 2 + r;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
        wmma::load_matrix_sync(a_frag, &block_a[kk][row_tile * WMMA_M], TILE_M + PAD);
#pragma unroll
        for (int c = 0; c < 4; c++) {
          int col_tile = warp_n * 4 + c;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
          wmma::load_matrix_sync(b_frag, &block_b[col_tile * WMMA_N][kk], TILE_K + PAD);
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
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    kernel<<< grid, block >>>(m,
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
