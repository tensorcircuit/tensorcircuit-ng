// Minimal region/tile-fusion prototype KERNELS (nvrtc-compiled via cupy.RawKernel;
// see results/_phase0/region_proto.py). No host/runtime-API code here -- nvrtc only.
//
// Proves the 512 MiB C1 anchor producer output C = A@B (c64[4096,16384], from
// A=c64[4096,1024] x B=c64[1024,16384]) need NOT materialize: the fused kernel computes
// c = A@B per element in registers and reduces |c|^2 on-chip (no full C); the
// materialized kernel writes the full C then reduces it. Same compute, different
// materialization -- isolates the 512 MiB global buffer.
//
// Minimal viable subset (per checkpoint): naive per-element complex GEMM. Deferred to
// full Task 3: tiled/shared-memory realization, occupancy, pack/recompute/conversion
// bytes, latency vs c64 baseline.

struct c64 {  // complex64 = (real, imag), matches numpy/torch complex64 memory layout
    float x, y;
};

// acc = sum_k A[i,k] * B[k,j]  (complex). Row-major A[M,K], B[K,N].
__device__ inline void gemm_elem(const c64* A, const c64* B, int M, int N, int K,
                                 int i, int j, float* ox, float* oy) {
    float accx = 0.f, accy = 0.f;
    const c64* arow = A + (long)i * K;
    for (int k = 0; k < K; ++k) {
        const c64& a = arow[k];
        const c64& b = B[(long)k * N + j];
        accx += a.x * b.x - a.y * b.y;
        accy += a.x * b.y + a.y * b.x;
    }
    *ox = accx;
    *oy = accy;
}

// FUSED: per element compute c=A@B in registers, block-reduce |c|^2, one atomicAdd/block.
extern "C" __global__ void gemm_reduce_kernel(const c64* A, const c64* B, float* scalar,
                                              int M, int N, int K, int MN) {
    extern __shared__ float sh[];
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.f;
    if (t < MN) {
        int i = t / N, j = t % N;
        float cx, cy;
        gemm_elem(A, B, M, N, K, i, j, &cx, &cy);
        v = cx * cx + cy * cy;
    }
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(scalar, sh[0]);
}

// MATERIALIZED step 1: write the full C to global.
extern "C" __global__ void gemm_write_kernel(const c64* A, const c64* B, c64* C,
                                             int M, int N, int K, int MN) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= MN) return;
    int i = t / N, j = t % N;
    float cx, cy;
    gemm_elem(A, B, M, N, K, i, j, &cx, &cy);
    C[t].x = cx;
    C[t].y = cy;
}

// MATERIALIZED step 2: reduce |C|^2 over the full buffer.
extern "C" __global__ void reduce_sqsum_kernel(const c64* C, float* scalar, int MN) {
    extern __shared__ float sh[];
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (t < MN) ? (C[t].x * C[t].x + C[t].y * C[t].y) : 0.f;
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(scalar, sh[0]);
}
