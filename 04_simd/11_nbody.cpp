#include <cstdio>
#include <cstdlib>
#include <x86intrin.h>

int main() {
  const int N = 16;
  alignas(64) float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  __m512 xvec = _mm512_load_ps(x);
  __m512 yvec = _mm512_load_ps(y);
  __m512 mvec = _mm512_load_ps(m);

  for(int i=0; i<N; i++) {
    __m512 rx = _mm512_sub_ps(_mm512_set1_ps(x[i]), xvec);
    __m512 ry = _mm512_sub_ps(_mm512_set1_ps(y[i]), yvec);
    __m512 r2 = _mm512_add_ps(_mm512_mul_ps(rx, rx), _mm512_mul_ps(ry, ry));
    __mmask16 mask = 0xffff ^ (1 << i);
    __m512 inv_r = _mm512_rsqrt14_ps(r2);
    __m512 inv_r3 = _mm512_mul_ps(_mm512_mul_ps(inv_r, inv_r), inv_r);
    __m512 scale = _mm512_mul_ps(mvec, inv_r3);

    fx[i] -= _mm512_mask_reduce_add_ps(mask, _mm512_mul_ps(rx, scale));
    fy[i] -= _mm512_mask_reduce_add_ps(mask, _mm512_mul_ps(ry, scale));

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
