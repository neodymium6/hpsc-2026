#include <cstdio>
#include <cstdlib>

__global__ void init_bucket(int *bucket, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < range) bucket[i] = 0;
}

__global__ void count_bucket(int *key, int *bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) atomicAdd(&bucket[key[i]], 1);
}

__global__ void copy_bucket(int *bucket, int *offset, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < range) offset[i] = bucket[i];
}

__global__ void copy_offset(int *offset, int *buffer, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < range) buffer[i] = offset[i];
}

__global__ void scan_step(int *offset, int *buffer, int range, int stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= stride && i < range) offset[i] += buffer[i-stride];
}

__global__ void write_sorted(int *key, int *bucket, int *offset, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < range) {
    int j = i == 0 ? 0 : offset[i-1];
    for(int k=0; k<bucket[i]; k++) {
      key[j+k] = i;
    }
  }
}

int main() {
  int n = 50;
  int range = 5;
  int *key, *bucket, *offset, *buffer;
  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));
  cudaMallocManaged(&offset, range*sizeof(int));
  cudaMallocManaged(&buffer, range*sizeof(int));

  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int threads = 256;
  int range_blocks = (range+threads-1)/threads;
  init_bucket<<<range_blocks,threads>>>(bucket, range);
  count_bucket<<<(n+threads-1)/threads,threads>>>(key, bucket, n);
  copy_bucket<<<range_blocks,threads>>>(bucket, offset, range);
  for(int stride=1; stride<range; stride<<=1) {
    copy_offset<<<range_blocks,threads>>>(offset, buffer, range);
    scan_step<<<range_blocks,threads>>>(offset, buffer, range, stride);
  }
  write_sorted<<<range_blocks,threads>>>(key, bucket, offset, range);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(key);
  cudaFree(bucket);
  cudaFree(offset);
  cudaFree(buffer);
}
