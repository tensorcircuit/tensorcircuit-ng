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
