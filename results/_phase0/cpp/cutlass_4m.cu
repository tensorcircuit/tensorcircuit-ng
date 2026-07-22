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

// Workspace bytes the RealGemm kernel needs for an (M,N,K) problem. Used by the
// probe to report `resource.workspace_bytes` without allocating the kernel's I/O.
int64_t real_gemm_workspace_bytes(int M, int N, int K) {
    RealGemm op;
    typename RealGemm::Arguments args(
        {M, N, K},
        {nullptr, K}, {nullptr, N}, {nullptr, N}, {nullptr, N}, {1.0f, 0.0f}, 1);
    return (int64_t)op.get_workspace_size(args);
}

// ============================================================================
// Task 4 (final-remediation): 3.x Blackwell Sm100 peak 4M attempt.
// Genuine CUTLASS 3.x instantiation using CollectiveBuilder +
// device::GemmUniversalAdapter<kernel::GemmUniversal<...>> (the pattern from
// examples/70_blackwell_gemm/70_blackwell_fp16_gemm.cu), NOT the 2.x
// device::GemmUniversal<...,arch::Sm100,...> sketched in the brief. Compiled
// only when -DCUTLASS_ENABLE_SM100_4M=1 is passed to build_extension(). If this
// fails to compile or instantiate for sm_120, build_extension raises and
// _attempt_sm100_then_sm80 transparently falls back to the proven 2.x Sm80
// path (FEASIBLE_WITH_SM80_FALLBACK) — a fully legitimate outcome.
// ============================================================================
#if defined(CUTLASS_ENABLE_SM100_4M)
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

// CUTLASS_ARCH_MMA_SM100_SUPPORTED is set unconditionally for nvcc >= 12.8
// (host-side macro, see include/cutlass/arch/config.h:87). The actual Sm100
// device-side MMA intrinsics are gated by CUTLASS_ARCH_MMA_SM100_ENABLED which
// fires only for __CUDA_ARCH__ == 1000 — NOT for sm_120 (== 1200). This guard
// therefore typically compiles the host-side template graph but finds no
// usable dispatch when targeting sm_120; that is exactly the question this
// probe answers.
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// BF16 A/B, FP32 accumulate + output. TN input layout (A row-major, B
// col-major = physical row-major (K,N) reinterpreted as col-major (N,K), same
// memory). D declared ROW-major to match the physical at::empty({M,N}) buffer
// (the standard Hopper/Blackwell example uses ColMajor D, but that requires
// either allocating the buffer col-major or transposing — declaring RowMajor
// is the transparent fix).
using Sm100ElementA   = cutlass::bfloat16_t;
using Sm100ElementB   = cutlass::bfloat16_t;
using Sm100ElementD   = float;
using Sm100ElementAcc = float;
using Sm100LayoutA    = cutlass::layout::RowMajor;
using Sm100LayoutB    = cutlass::layout::ColumnMajor;
using Sm100LayoutD    = cutlass::layout::RowMajor;
constexpr int Sm100AlignA = 128 / cutlass::sizeof_bits<Sm100ElementA>::value;
constexpr int Sm100AlignB = 128 / cutlass::sizeof_bits<Sm100ElementB>::value;
constexpr int Sm100AlignD = 128 / cutlass::sizeof_bits<Sm100ElementD>::value;

using Sm100ArchTag      = cutlass::arch::Sm100;
using Sm100OpClass      = cutlass::arch::OpClassTensorOp;
using Sm100MmaTileShape = cute::Shape<cute::_256, cute::_128, cute::_64>;
using Sm100ClusterShape = cute::Shape<cute::_2,  cute::_2,   cute::_1>;

using Sm100Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    Sm100ArchTag, Sm100OpClass,
    Sm100MmaTileShape, Sm100ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    Sm100ElementAcc, Sm100ElementAcc,
    Sm100ElementD, Sm100LayoutD, Sm100AlignD,
    Sm100ElementD, Sm100LayoutD, Sm100AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using Sm100Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    Sm100ArchTag, Sm100OpClass,
    Sm100ElementA, Sm100LayoutA, Sm100AlignA,
    Sm100ElementB, Sm100LayoutB, Sm100AlignB,
    Sm100ElementAcc,
    Sm100MmaTileShape, Sm100ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Sm100Epilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using Sm100GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,  // ProblemShape = (M, N, K, batch)
    Sm100Mainloop,
    Sm100Epilogue,
    void>;  // default CLC tile scheduler

using Sm100Gemm = cutlass::gemm::device::GemmUniversalAdapter<Sm100GemmKernel>;

// Single real GEMM via the 3.x universal adapter. alpha*A*B + beta*D (D is
// both source C and destination — beta accumulates into the existing buffer).
// Throws on any non-success status so _attempt_sm100_then_sm80 records the
// verbatim blocker and falls back to the 2.x Sm80 path.
//
// Hoisted to file scope (unguarded) so both the Sm100 and Sm120 compile-guarded
// blocks can name it. CUTLASS's Status enum is always available via cutlass.h.
static const char* cutlass_status_name(cutlass::Status s) {
    switch (s) {
        case cutlass::Status::kSuccess:              return "kSuccess";
        case cutlass::Status::kErrorMisalignedOperand: return "kErrorMisalignedOperand";
        case cutlass::Status::kErrorInvalidDataType: return "kErrorInvalidDataType";
        case cutlass::Status::kErrorInvalidLayout:   return "kErrorInvalidLayout";
        case cutlass::Status::kErrorInvalidProblem:  return "kErrorInvalidProblem";
        case cutlass::Status::kErrorNotSupported:    return "kErrorNotSupported";
        case cutlass::Status::kErrorWorkspaceNull:   return "kErrorWorkspaceNull";
        case cutlass::Status::kErrorInternal:        return "kErrorInternal";
        case cutlass::Status::kErrorArchMismatch:    return "kErrorArchMismatch";
        case cutlass::Status::kErrorInsufficientDriver: return "kErrorInsufficientDriver";
        case cutlass::Status::kErrorMemoryAllocation: return "kErrorMemoryAllocation";
        default: return "kInvalid";
    }
}

// Single real GEMM via the 3.x Sm100 universal adapter. Used by cutlass_4m_sm100.
static cutlass::Status real_gemm_sm100(at::Tensor A, at::Tensor B, at::Tensor D,
                                       float alpha, float beta,
                                       cudaStream_t stream) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    // A row-major (M,K) → packed stride over (M,K,1).
    // B physical row-major (K,N), declared col-major (N,K) → packed stride over (N,K,1).
    // D physical row-major (M,N), declared row-major (M,N) → packed stride over (M,N,1).
    auto stride_A = cutlass::make_cute_packed_stride(
        typename Sm100Gemm::GemmKernel::StrideA{}, cute::make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Sm100Gemm::GemmKernel::StrideB{}, cute::make_shape(N, K, 1));
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Sm100Gemm::GemmKernel::StrideD{}, cute::make_shape(M, N, 1));
    typename Sm100Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {reinterpret_cast<Sm100ElementA*>(A.data_ptr()), stride_A,
         reinterpret_cast<Sm100ElementB*>(B.data_ptr()), stride_B},
        {{alpha, beta},
         reinterpret_cast<Sm100ElementD*>(D.data_ptr()), stride_D,
         reinterpret_cast<Sm100ElementD*>(D.data_ptr()), stride_D},
    };
    Sm100Gemm gemm;
    size_t ws = Sm100Gemm::get_workspace_size(args);
    at::Tensor workspace;
    void* ws_ptr = nullptr;
    if (ws) {
        workspace = at::empty({(int64_t)ws}, at::dtype(at::kByte).device(at::kCUDA));
        ws_ptr = workspace.data_ptr();
    }
    cutlass::Status st = gemm.can_implement(args);
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "Sm100 can_implement failed: ", cutlass_status_name(st),
                " (M=", M, ", N=", N, ", K=", K, ")");
    st = gemm.initialize(args, ws_ptr);
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "Sm100 initialize failed: ", cutlass_status_name(st),
                " — cudaFuncSetAttribute on device_kernel<Sm100GemmKernel> "
                "fails on sm_120 (Sm100 device MMA gated by __CUDA_ARCH__==1000)");
    st = gemm.run(stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "Sm100 run failed: ", cutlass_status_name(st));
    return st;
}

// 4M complex matmul on the Sm100 path. Same alpha/beta structure as
// cutlass_4m_sm80: ReC = +ReA.ReB - ImA.ImB ; ImC = +ReA.ImB + ImA.ReB,
// accumulated via (alpha, beta) into the ReC/ImC buffers.
std::tuple<at::Tensor, at::Tensor> cutlass_4m_sm100(
        at::Tensor ReA, at::Tensor ImA, at::Tensor ReB, at::Tensor ImB) {
    TORCH_CHECK(ReA.is_cuda() && ReB.is_cuda(), "tensors must be CUDA");
    int M = ReA.size(0), K = ReA.size(1), N = ReB.size(1);
    auto ReC = at::empty({M, N}, at::dtype(at::kFloat).device(at::kCUDA));
    auto ImC = at::empty({M, N}, at::dtype(at::kFloat).device(at::kCUDA));
    cudaStream_t s = c10::cuda::getCurrentCUDAStream().stream();
    real_gemm_sm100(ReA, ReB, ReC,  1.0f, 0.0f, s);  // ReC  = ReA.ReB
    real_gemm_sm100(ImA, ImB, ReC, -1.0f, 1.0f, s);  // ReC -= ImA.ImB
    real_gemm_sm100(ReA, ImB, ImC,  1.0f, 0.0f, s);  // ImC  = ReA.ImB
    real_gemm_sm100(ImA, ReB, ImC,  1.0f, 1.0f, s);  // ImC += ImA.ReB
    return {ReC, ImC};
}

#define HAS_CUTLASS_4M_SM100 1
#else
#define HAS_CUTLASS_4M_SM100 0
#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED
#else
#define HAS_CUTLASS_4M_SM100 0
#endif  // CUTLASS_ENABLE_SM100_4M

// ============================================================================
// Task 4b (final-remediation): native Sm120 (CONSUMER Blackwell, RTX 5070 Ti)
// peak 4M attempt. CUTLASS_ARCH_MMA_SM120_ENABLED fires at __CUDA_ARCH__==1200
// (exactly our GPU), so unlike Sm100 this is the *correct* native arch tag.
// Same CollectiveBuilder + GemmUniversalAdapter wiring as Sm100, swapping only
// ArchTag -> arch::Sm120 and ClusterShape -> <1,1,1> (the Sm120 builder
// requires this: sm120_mma_builder.inl:84 "no programmatic multicast on this
// arch"). CUTLASS's Sm120 collective builder is documented as F8F6F4-only
// (sm120_mma_builder.inl:80,115 — "Non-blockscaled collective builder only
// supports F8F6F4 MMA"); this attempt probes whether a BF16 instantiation
// slips through anyway (e.g. via a generic fallback or a relaxed dispatch).
// ============================================================================
#if defined(CUTLASS_ENABLE_SM120_4M)
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)

using Sm120ElementA   = cutlass::bfloat16_t;
using Sm120ElementB   = cutlass::bfloat16_t;
using Sm120ElementD   = float;
using Sm120ElementAcc = float;
using Sm120LayoutA    = cutlass::layout::RowMajor;     // TN
using Sm120LayoutB    = cutlass::layout::ColumnMajor;
using Sm120LayoutD    = cutlass::layout::RowMajor;
constexpr int Sm120AlignA = 128 / cutlass::sizeof_bits<Sm120ElementA>::value;
constexpr int Sm120AlignB = 128 / cutlass::sizeof_bits<Sm120ElementB>::value;
constexpr int Sm120AlignD = 128 / cutlass::sizeof_bits<Sm120ElementD>::value;

using Sm120ArchTag      = cutlass::arch::Sm120;
using Sm120OpClass      = cutlass::arch::OpClassTensorOp;
using Sm120MmaTileShape = cute::Shape<cute::_128, cute::_128, cute::_64>;
using Sm120ClusterShape = cute::Shape<cute::_1,  cute::_1,   cute::_1>;  // required

using Sm120Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    Sm120ArchTag, Sm120OpClass,
    Sm120MmaTileShape, Sm120ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    Sm120ElementAcc, Sm120ElementAcc,
    Sm120ElementD, Sm120LayoutD, Sm120AlignD,
    Sm120ElementD, Sm120LayoutD, Sm120AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using Sm120Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    Sm120ArchTag, Sm120OpClass,
    Sm120ElementA, Sm120LayoutA, Sm120AlignA,
    Sm120ElementB, Sm120LayoutB, Sm120AlignB,
    Sm120ElementAcc,
    Sm120MmaTileShape, Sm120ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Sm120Epilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using Sm120GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    Sm120Mainloop,
    Sm120Epilogue,
    void>;

using Sm120Gemm = cutlass::gemm::device::GemmUniversalAdapter<Sm120GemmKernel>;

static cutlass::Status real_gemm_sm120(at::Tensor A, at::Tensor B, at::Tensor D,
                                       float alpha, float beta,
                                       cudaStream_t stream) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto stride_A = cutlass::make_cute_packed_stride(
        typename Sm120Gemm::GemmKernel::StrideA{}, cute::make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Sm120Gemm::GemmKernel::StrideB{}, cute::make_shape(N, K, 1));
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Sm120Gemm::GemmKernel::StrideD{}, cute::make_shape(M, N, 1));
    typename Sm120Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {reinterpret_cast<Sm120ElementA*>(A.data_ptr()), stride_A,
         reinterpret_cast<Sm120ElementB*>(B.data_ptr()), stride_B},
        {{alpha, beta},
         reinterpret_cast<Sm120ElementD*>(D.data_ptr()), stride_D,
         reinterpret_cast<Sm120ElementD*>(D.data_ptr()), stride_D},
    };
    Sm120Gemm gemm;
    size_t ws = Sm120Gemm::get_workspace_size(args);
    at::Tensor workspace;
    void* ws_ptr = nullptr;
    if (ws) {
        workspace = at::empty({(int64_t)ws}, at::dtype(at::kByte).device(at::kCUDA));
        ws_ptr = workspace.data_ptr();
    }
    cutlass::Status st = gemm.can_implement(args);
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "Sm120 can_implement failed: ", cutlass_status_name(st),
                " (M=", M, ", N=", N, ", K=", K, ")");
    st = gemm.initialize(args, ws_ptr);
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "Sm120 initialize failed: ", cutlass_status_name(st));
    st = gemm.run(stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "Sm120 run failed: ", cutlass_status_name(st));
    return st;
}

std::tuple<at::Tensor, at::Tensor> cutlass_4m_sm120(
        at::Tensor ReA, at::Tensor ImA, at::Tensor ReB, at::Tensor ImB) {
    TORCH_CHECK(ReA.is_cuda() && ReB.is_cuda(), "tensors must be CUDA");
    int M = ReA.size(0), K = ReA.size(1), N = ReB.size(1);
    auto ReC = at::empty({M, N}, at::dtype(at::kFloat).device(at::kCUDA));
    auto ImC = at::empty({M, N}, at::dtype(at::kFloat).device(at::kCUDA));
    cudaStream_t s = c10::cuda::getCurrentCUDAStream().stream();
    real_gemm_sm120(ReA, ReB, ReC,  1.0f, 0.0f, s);
    real_gemm_sm120(ImA, ImB, ReC, -1.0f, 1.0f, s);
    real_gemm_sm120(ReA, ImB, ImC,  1.0f, 0.0f, s);
    real_gemm_sm120(ImA, ReB, ImC,  1.0f, 1.0f, s);
    return {ReC, ImC};
}

#define HAS_CUTLASS_4M_SM120 1
#else
#define HAS_CUTLASS_4M_SM120 0
#endif  // CUTLASS_ARCH_MMA_SM120_SUPPORTED
#else
#define HAS_CUTLASS_4M_SM120 0
#endif  // CUTLASS_ENABLE_SM120_4M

// Exposed to Python so _attempt_sm100_then_sm80 can distinguish "build
// succeeded but the Sm100 path was compiled out" (guard never set) from a
// real Sm100 kernel being present.
bool has_sm100() { return HAS_CUTLASS_4M_SM100; }
bool has_sm120() { return HAS_CUTLASS_4M_SM120; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("probe", &probe, "CUTLASS build smoke");
    m.def("cutlass_4m_sm80", &cutlass_4m_sm80, "2.x Sm80 planar-complex 4M GEMM");
    m.def("real_gemm_workspace_bytes", &real_gemm_workspace_bytes, "");
    m.def("has_sm100", &has_sm100,
          "whether the 3.x Sm100 4M kernel compiled into this build");
    m.def("has_sm120", &has_sm120,
          "whether the native 3.x Sm120 4M kernel compiled into this build");
    m.def("cutlass_4m_sm100",
          [](at::Tensor a, at::Tensor b, at::Tensor c, at::Tensor d)
              -> std::tuple<at::Tensor, at::Tensor> {
#if HAS_CUTLASS_4M_SM100
              return cutlass_4m_sm100(a, b, c, d);
#else
              TORCH_CHECK(false, "sm100 4M not enabled in this build");
#endif
          },
          "3.x Sm100 planar-complex 4M GEMM (compile-guarded)");
    m.def("cutlass_4m_sm120",
          [](at::Tensor a, at::Tensor b, at::Tensor c, at::Tensor d)
              -> std::tuple<at::Tensor, at::Tensor> {
#if HAS_CUTLASS_4M_SM120
              return cutlass_4m_sm120(a, b, c, d);
#else
              TORCH_CHECK(false, "sm120 4M not enabled in this build");
#endif
          },
          "3.x Sm120 planar-complex 4M GEMM (compile-guarded, consumer Blackwell)");
}
