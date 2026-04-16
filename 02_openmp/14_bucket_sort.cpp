#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0); 
 #pragma omp parallel for
  for (int i=0; i<n; i++)
 #pragma omp atomic update
    bucket[key[i]]++;
  std::vector<int> offset = bucket;
  std::vector<int> buffer(range,0);
#pragma omp parallel
  for (int j=1; j<range; j<<=1) {
#pragma omp for
    for (int i=0; i<range; i++)
      buffer[i] = offset[i];
#pragma omp for
    for (int i=j; i<range; i++)
      offset[i] += buffer[i-j];
  }
#pragma omp parallel for
  for (int i=0; i<range; i++) {
    int j = i == 0 ? 0 : offset[i-1];
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
