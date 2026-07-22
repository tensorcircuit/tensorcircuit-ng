// Task 8 CUTLASS SM120 4M kernels. Built via torch.utils.cpp_extension
// (CUDA_HOME=nvcc_spike, -I<CUTLASS_ROOT>/include). Entry points added per task.
#include <torch/extension.h>
#include "cutlass/cutlass.h"

#include "cutlass/gemm/device/gemm.h"
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

// Smoke entry (replaced by real kernels in later tasks). Forces CUTLASS include resolution.
int probe() { return 42; }

// D = alpha*A*B + beta*C, BF16 in, FP32 accumulate, FP32 out. One real GEMM.
using RealGemm = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

static cutlass::Status real_gemm(at::Tensor A, at::Tensor B, at::Tensor D,
                                 float alpha, float beta, cudaStream_t stream) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    RealGemm op;
    typename RealGemm::Arguments args(
        {M, N, K},
        {reinterpret_cast<cutlass::bfloat16_t*>(A.data_ptr()), K},
        {reinterpret_cast<cutlass::bfloat16_t*>(B.data_ptr()), N},
        {reinterpret_cast<float*>(D.data_ptr()), N},
        {reinterpret_cast<float*>(D.data_ptr()), N},
        {alpha, beta}, 1);
    size_t ws = op.get_workspace_size(args);
    at::Tensor workspace;
    void* ws_ptr = nullptr;
    if (ws) {
        workspace = at::empty({(int64_t)ws}, at::dtype(at::kByte).device(at::kCUDA));
        ws_ptr = workspace.data_ptr();
    }
    return op(args, ws_ptr, stream);
}

// 4M complex matmul: ReC=ReA.ReB-ImA.ImB ; ImC=ReA.ImB+ImA.ReB (4 real GEMMs via alpha/beta).
std::tuple<at::Tensor, at::Tensor> cutlass_4m_sm80(
        at::Tensor ReA, at::Tensor ImA, at::Tensor ReB, at::Tensor ImB) {
    TORCH_CHECK(ReA.is_cuda() && ReB.is_cuda(), "tensors must be CUDA");
    int M = ReA.size(0), K = ReA.size(1), N = ReB.size(1);
    auto ReC = at::empty({M, N}, at::dtype(at::kFloat).device(at::kCUDA));
    auto ImC = at::empty({M, N}, at::dtype(at::kFloat).device(at::kCUDA));
    cudaStream_t s = c10::cuda::getCurrentCUDAStream().stream();
    real_gemm(ReA, ReB, ReC, 1.0f, 0.0f, s);   // ReC = ReA.ReB
    real_gemm(ImA, ImB, ReC, -1.0f, 1.0f, s);  // ReC -= ImA.ImB
    real_gemm(ReA, ImB, ImC, 1.0f, 0.0f, s);   // ImC = ReA.ImB
    real_gemm(ImA, ReB, ImC, 1.0f, 1.0f, s);   // ImC += ImA.ReB
    return {ReC, ImC};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("probe", &probe, "CUTLASS build smoke");
    m.def("cutlass_4m_sm80", &cutlass_4m_sm80, "2.x Sm80 planar-complex 4M GEMM");
}
