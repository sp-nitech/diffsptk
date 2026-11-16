// ------------------------------------------------------------------------ //
// Copyright 2022 SPTK Working Group                                        //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ------------------------------------------------------------------------ //

// Time-varying all-zero filter with linearly interpolated coefficients:
//
//   y[t] = sum_{m=0}^{L-1} h_t[m] x[t + z - m],
//   h_t  = b[n] + r[j] (b[n+1] - b[n]),  t = nP + j,  b[N] := b[N-1],
//
// where x is zero outside [0, T) and z is the index of the zeroth coefficient.
// SCALAR_T is given as a compile-time definition.

// Loads x[u] with zero padding outside [0, T).
__device__ __forceinline__ SCALAR_T load_or_zero(const SCALAR_T* __restrict__ x,
                                                 int u, int T) {
  return (0 <= u && u < T) ? x[u] : (SCALAR_T)0;
}

// Each thread computes BLOCKING consecutive outputs t0, ..., t0 + BLOCKING - 1
// of the same frame. The inputs x[t0 + r + z - m] are kept in a sliding window
// of registers, so that each pair of coefficients is used BLOCKING times.
//
// x: (B, T), b: (B, N, L), ramp: (P,), y: (B, T)
extern "C" __global__ void zerodf_forward(const SCALAR_T* __restrict__ x,
                                          const SCALAR_T* __restrict__ b,
                                          const SCALAR_T* __restrict__ ramp,
                                          SCALAR_T* __restrict__ y, int B, int T,
                                          int N, int L, int P, int z) {
  const int G = (P + BLOCKING - 1) / BLOCKING;
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long long)B * N * G) return;
  const long long bn = idx / G;
  const int g = (int)(idx - bn * G);
  const long long bi = bn / N;
  const int n = (int)(bn - bi * N);
  const int j0 = g * BLOCKING;
  const int t0 = n * P + j0;

  const SCALAR_T* xb = x + bi * T;
  const SCALAR_T* b1 = b + bn * L;
  const SCALAR_T* b2 = b + (bi * N + min(n + 1, N - 1)) * L;

  // Restrict m so that t0 + r + z - m < T for r = 0 and >= 0 for the last r.
  const int m_lo = max(0, t0 + z - (T - 1));
  const int m_hi = min(L - 1, t0 + BLOCKING - 1 + z);

  SCALAR_T w[BLOCKING], s1[BLOCKING], s2[BLOCKING];
#pragma unroll
  for (int r = 0; r < BLOCKING; r++) {
    w[r] = load_or_zero(xb, t0 + r + z - m_lo, T);
    s1[r] = 0;
    s2[r] = 0;
  }
  for (int m = m_lo; m <= m_hi; m++) {
    const SCALAR_T c1 = b1[m];
    const SCALAR_T c2 = b2[m];
#pragma unroll
    for (int r = 0; r < BLOCKING; r++) {
      s1[r] += c1 * w[r];
      s2[r] += c2 * w[r];
    }
#pragma unroll
    for (int r = BLOCKING - 1; 0 < r; r--) w[r] = w[r - 1];
    w[0] = load_or_zero(xb, t0 + z - m - 1, T);
  }

  SCALAR_T* yb = y + bi * T;
#pragma unroll
  for (int r = 0; r < BLOCKING; r++) {
    const int j = j0 + r;
    if (j < P) yb[t0 + r] = s1[r] + ramp[j] * (s2[r] - s1[r]);
  }
}

// Index of shared memory with a skew to avoid bank conflicts when threads access
// addresses separated by BLOCKING.
__device__ __forceinline__ int skew(int i) { return i + (i >> 5); }

// Gradient with respect to x:
//
//   gx[s] = sum_{m=0}^{L-1} gy[t] h_t[m],  t = s - z + m,  0 <= t < T.
//
// A block of BWD_X_THREADS threads computes S = BWD_X_THREADS * BLOCKING
// consecutive outputs, and each thread computes BLOCKING consecutive outputs
// s0, ..., s0 + BLOCKING - 1. The block visits t in chunks of at most BWD_X_CHUNK
// samples within a frame. For each chunk, the required coefficients of the two
// frames and the weighted gradients are staged in shared memory. Each thread keeps
// the coefficients b[t + z - s0 - r] in a sliding window of registers.
//
// gy: (B, T), b: (B, N, L), ramp: (P,), gx: (B, T)
#define BWD_X_SPAN (BWD_X_THREADS * BLOCKING)
#define BWD_X_WIDTH (BWD_X_CHUNK + BWD_X_SPAN - 1)

extern "C" __global__ void __launch_bounds__(BWD_X_THREADS)
    zerodf_backward_x(const SCALAR_T* __restrict__ gy,
                      const SCALAR_T* __restrict__ b,
                      const SCALAR_T* __restrict__ ramp, SCALAR_T* __restrict__ gx,
                      int B, int T, int N, int L, int P, int z) {
  __shared__ SCALAR_T b1s[BWD_X_WIDTH + (BWD_X_WIDTH >> 5) + 1];
  __shared__ SCALAR_T b2s[BWD_X_WIDTH + (BWD_X_WIDTH >> 5) + 1];
  __shared__ SCALAR_T v1s[BWD_X_CHUNK];
  __shared__ SCALAR_T v2s[BWD_X_CHUNK];

  const int n_tiles = (T + BWD_X_SPAN - 1) / BWD_X_SPAN;
  const long long bi = blockIdx.x / n_tiles;
  const int S0 = (int)(blockIdx.x - bi * n_tiles) * BWD_X_SPAN;
  const int s0 = S0 + threadIdx.x * BLOCKING;

  const SCALAR_T* gyb = gy + bi * T;
  const SCALAR_T* bb = b + bi * N * L;

  // Range of t that contributes to the block and to this thread.
  const int bt_lo = max(0, S0 - z);
  const int bt_hi = min(T - 1, S0 + BWD_X_SPAN - 1 - z + L - 1);
  const int tt_lo = max(0, s0 - z);
  const int tt_hi = min(T - 1, s0 + BLOCKING - 1 - z + L - 1);

  SCALAR_T acc[BLOCKING];
#pragma unroll
  for (int r = 0; r < BLOCKING; r++) acc[r] = 0;

  for (int n = bt_lo / P; n * P <= bt_hi; n++) {
    const SCALAR_T* b1 = bb + (long long)n * L;
    const SCALAR_T* b2 = bb + (long long)min(n + 1, N - 1) * L;
    const int f_hi = min(bt_hi, n * P + P - 1);
    for (int ta = max(bt_lo, n * P); ta <= f_hi; ta += BWD_X_CHUNK) {
      const int tb = min(f_hi, ta + BWD_X_CHUNK - 1);

      // Stage the coefficients b[base + i] for i in [0, width).
      const int base = ta + z - (S0 + BWD_X_SPAN - 1);
      const int width = tb - ta + BWD_X_SPAN;
      __syncthreads();
      for (int i = threadIdx.x; i < width; i += BWD_X_THREADS) {
        const int m = base + i;
        const bool valid = 0 <= m && m < L;
        b1s[skew(i)] = valid ? b1[m] : (SCALAR_T)0;
        b2s[skew(i)] = valid ? b2[m] : (SCALAR_T)0;
      }
      for (int i = threadIdx.x; i <= tb - ta; i += BWD_X_THREADS) {
        const int t = ta + i;
        const SCALAR_T v = gyb[t];
        const SCALAR_T v2 = ramp[t - n * P] * v;
        v1s[i] = v - v2;
        v2s[i] = v2;
      }
      __syncthreads();

      const int lo = max(ta, tt_lo);
      const int hi = min(tb, tt_hi);
      if (lo <= hi) {
        // Local index of b[t + z - s0] in the staged buffer.
        const int k = lo + z - s0 - base;
        SCALAR_T w1[BLOCKING], w2[BLOCKING];
#pragma unroll
        for (int r = 0; r < BLOCKING; r++) {
          w1[r] = b1s[skew(k - r)];
          w2[r] = b2s[skew(k - r)];
        }
        for (int t = lo; t <= hi; t++) {
          const SCALAR_T v1 = v1s[t - ta];
          const SCALAR_T v2 = v2s[t - ta];
#pragma unroll
          for (int r = 0; r < BLOCKING; r++) acc[r] += v1 * w1[r] + v2 * w2[r];
#pragma unroll
          for (int r = BLOCKING - 1; 0 < r; r--) {
            w1[r] = w1[r - 1];
            w2[r] = w2[r - 1];
          }
          const int kn = k + (t - lo) + 1;
          if (t < hi) {
            w1[0] = b1s[skew(kn)];
            w2[0] = b2s[skew(kn)];
          }
        }
      }
    }
  }

  SCALAR_T* gxb = gx + bi * T;
#pragma unroll
  for (int r = 0; r < BLOCKING; r++) {
    if (s0 + r < T) gxb[s0 + r] = acc[r];
  }
}

// Accumulates sum_j (a + c ramp[j]) gy[f0 + j] x[f0 + j + d - r] into acc[r] for
// r = 0, ..., BLOCKING - 1 over the frame starting at f0. The inputs are kept in
// a sliding window of registers.
__device__ __forceinline__ void accumulate_frame_b(
    const SCALAR_T* __restrict__ gy, const SCALAR_T* __restrict__ x,
    const SCALAR_T* __restrict__ ramp, int f0, int d, int P, int T, SCALAR_T a,
    SCALAR_T c, SCALAR_T* acc) {
  // Restrict j so that f0 + j + d - r is in [0, T) for some r.
  const int j_lo = max(0, -d - f0);
  const int j_hi = min(P - 1, T - 1 - d - f0 + BLOCKING - 1);
  if (j_hi < j_lo) return;

  SCALAR_T w[BLOCKING];
#pragma unroll
  for (int r = 0; r < BLOCKING; r++) w[r] = load_or_zero(x, f0 + j_lo + d - r, T);
  for (int j = j_lo; j <= j_hi; j++) {
    const SCALAR_T v = (a + c * ramp[j]) * gy[f0 + j];
#pragma unroll
    for (int r = 0; r < BLOCKING; r++) acc[r] += v * w[r];
#pragma unroll
    for (int r = BLOCKING - 1; 0 < r; r--) w[r] = w[r - 1];
    w[0] = load_or_zero(x, f0 + j + 1 + d, T);
  }
}

// Gradient with respect to b. Frame n receives the weight (1 - r) from its own
// samples and r from the samples of frame n - 1. The last frame also receives r
// from its own samples because b[N] := b[N-1]. Each thread computes BLOCKING
// consecutive coefficients m0, ..., m0 + BLOCKING - 1 of the same frame.
//
// gy: (B, T), x: (B, T), ramp: (P,), gb: (B, N, L)
extern "C" __global__ void zerodf_backward_b(const SCALAR_T* __restrict__ gy,
                                             const SCALAR_T* __restrict__ x,
                                             const SCALAR_T* __restrict__ ramp,
                                             SCALAR_T* __restrict__ gb, int B,
                                             int T, int N, int L, int P, int z) {
  const int G = (L + BLOCKING - 1) / BLOCKING;
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long long)B * N * G) return;
  const long long bn = idx / G;
  const int g = (int)(idx - bn * G);
  const long long bi = bn / N;
  const int n = (int)(bn - bi * N);
  const int m0 = g * BLOCKING;
  const int d = z - m0;

  const SCALAR_T* gyb = gy + bi * T;
  const SCALAR_T* xb = x + bi * T;

  SCALAR_T acc[BLOCKING];
#pragma unroll
  for (int r = 0; r < BLOCKING; r++) acc[r] = 0;
  if (n == N - 1) {
    accumulate_frame_b(gyb, xb, ramp, n * P, d, P, T, 1, 0, acc);
  } else {
    accumulate_frame_b(gyb, xb, ramp, n * P, d, P, T, 1, -1, acc);
  }
  if (0 < n) accumulate_frame_b(gyb, xb, ramp, (n - 1) * P, d, P, T, 0, 1, acc);

  SCALAR_T* gbb = gb + bn * L;
#pragma unroll
  for (int r = 0; r < BLOCKING; r++) {
    if (m0 + r < L) gbb[m0 + r] = acc[r];
  }
}
