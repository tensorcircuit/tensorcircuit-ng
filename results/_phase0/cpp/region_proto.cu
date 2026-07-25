// Real P->T->E two-stage region KERNELS (Task 4; nvrtc-compiled via cupy.RawKernel;
// see results/_phase0/region_proto.py). Replaces the rejected GEMM->norm reduce kernels.
//
// Computes E = D @ transform(A @ B) on the C1 anchor region WITHOUT materializing the full
// P (A@B, c64[4096,16384]) or T (transform(P), c64[64,1048576]). For each output E[i,j] the
// kernel gathers the producer column T[:,j] = transform(P)[:,j] (k=0..TM-1), computing each
// needed P[m,n] = A[m,:] @ B[:,n] on the fly (producer recompute -> FEASIBLE_WITH_RECOMPUTE).
//
// The transform is the fixed 8-D reshape->transpose->reshape from Task 2's edge map. Its
// inverse index math (T-linear -> P-linear) is computed inline from the reshape dims rd[8],
// transpose perm tp[8], and their C-order strides -- no large permutation buffer. Layouts in
// the real HLO are all row-major (== numpy/cupy C-order), so C-order flatten/unflatten matches
// the HLO bitcast exactly (validated in region_proto_test against Task 2's permutation).

struct c64 {  // complex64 = (real, imag), matches numpy/torch/cupy complex64 memory layout
    float x, y;
};

// Inverse transform: T-linear index -> P-linear index.
//   forward: P --reshape(rd)--> i8[8] --transpose(tp)--> o8[8] --reshape(outdim)--> T   (C-order)
//   inverse: unflatten t to o8 via outdim; i8[tp[b]] = o8[b]; flatten i8 via rd.
// outdim[b] = rd[tp[b]]; rd_stride[a] = prod(rd[a+1:]); out_stride[b] = prod(outdim[b+1:]).
__device__ __forceinline__ long long inv_transform(int t_lin, const int* outdim,
                                                   const int* out_stride, const int* rd_stride,
                                                   const int* tp) {
    int o8[8];
    int tt = t_lin;
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        o8[b] = tt / out_stride[b];
        tt -= o8[b] * out_stride[b];
    }
    int i8[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (int b = 0; b < 8; ++b) i8[tp[b]] = o8[b];
    long long p = 0;
    #pragma unroll
    for (int a = 0; a < 8; ++a) p += (long long)i8[a] * rd_stride[a];
    return p;
}

// E[i,j] = sum_{k=0}^{TM-1} D[i,k] * T[k,j],  T[k,j] = P[m,n] = sum_l A[m,l]*B[l,n],
// where (m,n) = divmod(inv_transform(k*TN + j), PN). P and T are never written to global.
extern "C" __global__ void __launch_bounds__(256) fused_pte_kernel(
    const c64* A, const c64* B, const c64* D, c64* E,
    int PM, int PN, int K1, int TM, int TN,
    const int* outdim, const int* out_stride, const int* rd_stride, const int* tp) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // output column in [0, TN)
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // output row    in [0, TM)
    if (i >= TM || j >= TN) return;
    const c64* drow = D + (long long)i * TM;
    float accx = 0.f, accy = 0.f;
    for (int k = 0; k < TM; ++k) {
        int t_lin = k * TN + j;
        long long p = inv_transform(t_lin, outdim, out_stride, rd_stride, tp);
        int m = (int)(p / PN);
        int n = (int)(p % PN);
        const c64* arow = A + (long long)m * K1;
        float px = 0.f, py = 0.f;
        for (int l = 0; l < K1; ++l) {
            const c64& a = arow[l];
            const c64& b = B[(long long)l * PN + n];
            px += a.x * b.x - a.y * b.y;
            py += a.x * b.y + a.y * b.x;
        }
        const c64& d = drow[k];
        accx += d.x * px - d.y * py;
        accy += d.x * py + d.y * px;
    }
    long long eidx = (long long)i * TN + j;
    E[eidx].x = accx;
    E[eidx].y = accy;
}

// Producer-tiled streaming kernel (Task G3).
//
// Computes E = D @ transform(A @ B) with producer tiling: each CTA owns an output tile
// E[i0:i0+BM_c, j0:j0+BN_c] and computes the needed T[k,j] = transform(P)[k,j] values into
// shared memory ONCE (as a "producer tile"), then reuses them across the BM_c consumer rows
// (the i dimension). This reduces the producer recompute factor from TM (direct kernel, which
// recomputes P[m,n] per E element) to ceil(TM/BM_c): each P[m,n] is computed by at most
// ceil(TM/BM_c) CTAs that share the same j-range.
//
// The needed P[m,n] for the CTA's output tile are {inv_transform(k*TN + j) : j in [j0, j0+BN_c),
// k in [0, TM)}. These are NOT contiguous in P[PM,PN] for this 8-D transform, so the "producer
// tile" in shared memory is a logical batch of (j_local, k) pairs (not a 2-D contiguous P slice).
// The batch is tiled in (BN_p, BM_p) chunks to bound shared-memory size. The BK_p parameter
// tiles the K1 inner accumulation (standard GEMM K-tiling; no effect on correctness).
//
// Each thread owns exactly one output E[i0+ty, j0+tx] (blockDim.x == BM_c * BN_c). The thread
// cooperatively computes shared_P in phase 1, then accumulates D[i,k]*shared_P[jl][kl] in phase 2.
// The accumulator persists in registers across (jb, kb) batches.
extern "C" __global__ void fused_pte_tiled_kernel(
    const c64* A, const c64* B, const c64* D, c64* E,
    int PM, int PN, int K1, int TM, int TN,
    int BM_p, int BN_p, int BK_p, int BM_c, int BN_c,
    const int* outdim, const int* out_stride, const int* rd_stride, const int* tp) {

    int i0 = blockIdx.y * BM_c;
    int j0 = blockIdx.x * BN_c;
    int tid = threadIdx.x;

    // Shared memory: producer tile [BN_p][BM_p] (c64). Dynamic shared mem sized by caller.
    extern __shared__ c64 shared_P[];

    // Thread -> output element. blockDim.x == BM_c * BN_c (one thread per output).
    int ty = tid / BN_c;
    int tx = tid % BN_c;
    int i = i0 + ty;
    int j = j0 + tx;
    bool valid = (i < TM && j < TN);

    const c64* drow = valid ? (D + (long long)i * TM) : 0;
    float accx = 0.f, accy = 0.f;

    // Iterate over (j_batch, k_batch) covering [0, BN_c) x [0, TM) in (BN_p, BM_p) chunks.
    for (int jb = 0; jb < BN_c; jb += BN_p) {
        int batch_j = (BN_p < BN_c - jb) ? BN_p : (BN_c - jb);
        for (int kb = 0; kb < TM; kb += BM_p) {
            int batch_k = (BM_p < TM - kb) ? BM_p : (TM - kb);
            int batch_n = batch_j * batch_k;

            // Phase 1: cooperatively compute producer tile into shared_P[0..batch_n-1].
            for (int idx = tid; idx < batch_n; idx += blockDim.x) {
                int jl = idx / batch_k;  // 0..batch_j-1
                int kl = idx % batch_k;  // 0..batch_k-1
                int j_cur = j0 + jb + jl;
                int k_cur = kb + kl;
                int t_lin = k_cur * TN + j_cur;
                long long p = inv_transform(t_lin, outdim, out_stride, rd_stride, tp);
                int m = (int)(p / PN);
                int n = (int)(p % PN);
                const c64* arow = A + (long long)m * K1;
                float px = 0.f, py = 0.f;
                for (int l0 = 0; l0 < K1; l0 += BK_p) {
                    int l_end = (BK_p < K1 - l0) ? (l0 + BK_p) : K1;
                    for (int l = l0; l < l_end; ++l) {
                        const c64& a = arow[l];
                        const c64& b = B[(long long)l * PN + n];
                        px += a.x * b.x - a.y * b.y;
                        py += a.x * b.y + a.y * b.x;
                    }
                }
                shared_P[jl * BM_p + kl].x = px;
                shared_P[jl * BM_p + kl].y = py;
            }
            __syncthreads();

            // Phase 2: accumulate D[i, kb+kl] * shared_P[tx-jb][kl] for this thread's output.
            if (valid) {
                int jl = tx - jb;  // this thread's j offset relative to batch
                if (jl >= 0 && jl < batch_j) {
                    const c64* srow = &shared_P[jl * BM_p];
                    for (int kl = 0; kl < batch_k; ++kl) {
                        const c64& t = srow[kl];
                        const c64& d = drow[kb + kl];
                        accx += d.x * t.x - d.y * t.y;
                        accy += d.x * t.y + d.y * t.x;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (valid) {
        long long eidx = (long long)i * TN + j;
        E[eidx].x = accx;
        E[eidx].y = accy;
    }
}
