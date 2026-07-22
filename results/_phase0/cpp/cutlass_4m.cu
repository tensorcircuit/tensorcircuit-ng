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

// cutlass::Status -> name helper. Hoisted to true file scope (unguarded) so the
// Sm80 path (cutlass_4m_sm80 below) AND the Sm100/Sm120 compile-guarded blocks
// can all name it. CUTLASS's Status enum is always available via cutlass.h.
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

// 4M complex matmul: ReC=ReA.ReB-ImA.ImB ; ImC=ReA.ImB+ImA.ReB (4 real GEMMs via alpha/beta).
std::tuple<at::Tensor, at::Tensor> cutlass_4m_sm80(
        at::Tensor ReA, at::Tensor ImA, at::Tensor ReB, at::Tensor ImB) {
    TORCH_CHECK(ReA.is_cuda() && ReB.is_cuda(), "tensors must be CUDA");
    int M = ReA.size(0), K = ReA.size(1), N = ReB.size(1);
    auto ReC = at::empty({M, N}, at::dtype(at::kFloat).device(at::kCUDA));
    auto ImC = at::empty({M, N}, at::dtype(at::kFloat).device(at::kCUDA));
    cudaStream_t s = c10::cuda::getCurrentCUDAStream().stream();
    cutlass::Status st;
    st = real_gemm(ReA, ReB, ReC, 1.0f, 0.0f, s);   // ReC = ReA.ReB
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "real_gemm failed: ", cutlass_status_name(st));
    st = real_gemm(ImA, ImB, ReC, -1.0f, 1.0f, s);  // ReC -= ImA.ImB
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "real_gemm failed: ", cutlass_status_name(st));
    st = real_gemm(ReA, ImB, ImC, 1.0f, 0.0f, s);   // ImC = ReA.ImB
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "real_gemm failed: ", cutlass_status_name(st));
    st = real_gemm(ImA, ReB, ImC, 1.0f, 1.0f, s);   // ImC += ImA.ReB
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "real_gemm failed: ", cutlass_status_name(st));
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

// ============================================================================
// Task 5 (final-remediation): CUTLASS 2.x GemmGrouped over a heterogeneous
// shape set — the Task 7 cuBLAS-gap handoff. cuBLAS LtMatmul had no grouped
// planar-complex path for heterogeneous shapes; this probe wires the CUTLASS
// 2.x device::GemmGrouped (arch::Sm80 — the only BF16-viable grouped path on
// sm_120, since 3.x Sm100/Sm120 grouped either won't instantiate at __CUDA_ARCH__
// ==1200 or hard-gate to F8F6F4) to run the 4-real-GEMM complex decomposition
// (ReC=ReA.ReB-ImA.ImB ; ImC=ReA.ImB+ImA.ReB) over G groups of distinct
// (M,K,N) per group. Built only when -DCUTLASS_ENABLE_GROUPED_4M=1 is passed.
// If this fails to compile or instantiate for sm_120, build_extension raises
// and run_grouped returns status=BLOCKED (legitimate verdict per spec §9).
// ============================================================================
#if defined(CUTLASS_ENABLE_GROUPED_4M)
#include <cstring>
#include <vector>
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"

// 2.x GemmGrouped configuration: BF16 A/B, FP32 accumulate + FP32 output,
// RowMajor throughout (matches the proven single-4m cutlass_4m_sm80 layout).
// Alignment 8 for BF16 inputs, 4 for FP32 output (128/sizeof_bits).
constexpr int kGroupedAlignA = 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value;  // 8
constexpr int kGroupedAlignB = 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value;  // 8
constexpr int kGroupedAlignC = 128 / cutlass::sizeof_bits<float>::value;                 // 4

using GroupedEpilogue = cutlass::epilogue::thread::LinearCombination<
    float, kGroupedAlignC, float, float>;

using GroupedGemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone, kGroupedAlignA,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone, kGroupedAlignB,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    GroupedEpilogue,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4>::GemmKernel;

using GroupedGemm = cutlass::gemm::device::GemmGrouped<GroupedGemmKernel>;

// Local status-name helper (cutlass_status_name in the Sm100 block is not
// visible unless CUTLASS_ENABLE_SM100_4M is also defined — this keeps the
// grouped block self-contained).
static const char* grouped_status_name(cutlass::Status s) {
  switch (s) {
    case cutlass::Status::kSuccess:                return "kSuccess";
    case cutlass::Status::kErrorMisalignedOperand: return "kErrorMisalignedOperand";
    case cutlass::Status::kErrorInvalidDataType:   return "kErrorInvalidDataType";
    case cutlass::Status::kErrorInvalidLayout:     return "kErrorInvalidLayout";
    case cutlass::Status::kErrorInvalidProblem:    return "kErrorInvalidProblem";
    case cutlass::Status::kErrorNotSupported:      return "kErrorNotSupported";
    case cutlass::Status::kErrorWorkspaceNull:     return "kErrorWorkspaceNull";
    case cutlass::Status::kErrorInternal:          return "kErrorInternal";
    case cutlass::Status::kErrorArchMismatch:      return "kErrorArchMismatch";
    case cutlass::Status::kErrorMemoryAllocation:  return "kErrorMemoryAllocation";
    default:                                       return "kInvalid";
  }
}

// Copy a host int64 vector to a device int64 tensor (for ptr arrays + lda/ldb/ldc/ldd).
static at::Tensor _grouped_int64_to_device(const std::vector<int64_t>& host) {
  auto cpu = at::empty({(int64_t)host.size()},
                       at::TensorOptions().dtype(at::kLong).device(at::kCPU));
  std::memcpy(cpu.data_ptr<int64_t>(), host.data(), host.size() * sizeof(int64_t));
  return cpu.to(at::kCUDA, /*non_blocking=*/false);
}

// Build the device-side pointer-to-pointer array from a list of device tensors.
// Each int64 holds a device pointer; on the device this re-interprets as Element**.
static at::Tensor _grouped_ptrs_to_device(const std::vector<at::Tensor>& ts) {
  std::vector<int64_t> p(ts.size());
  for (size_t i = 0; i < ts.size(); ++i) p[i] = (int64_t)ts[i].data_ptr();
  return _grouped_int64_to_device(p);
}

// One real GemmGrouped pass: out = alpha * (A_pass . B_pass) + beta * out, across all G groups.
// ptr_A / ptr_B select the input list; ptr_C and ptr_D point at the same output buffer
// (in-place accumulate). alpha/beta are uniform across groups.
static cutlass::Status _grouped_pass(
    int G, int threadblock_count,
    cutlass::gemm::GemmCoord* problem_sizes_device,
    cutlass::gemm::GemmCoord* problem_sizes_host,
    const at::Tensor& ptr_A_dev, const at::Tensor& ptr_B_dev,
    const at::Tensor& ptr_C_dev,
    const at::Tensor& lda_dev, const at::Tensor& ldb_dev,
    const at::Tensor& ldc_dev, const at::Tensor& ldd_dev,
    float alpha, float beta, void* workspace, cudaStream_t stream) {
  GroupedGemm op;
  typename GroupedGemm::EpilogueOutputOp::Params epilogue_op{alpha, beta};
  typename GroupedGemm::Arguments args(
      problem_sizes_device, G, threadblock_count, epilogue_op,
      reinterpret_cast<cutlass::bfloat16_t**>(ptr_A_dev.data_ptr<int64_t>()),
      reinterpret_cast<cutlass::bfloat16_t**>(ptr_B_dev.data_ptr<int64_t>()),
      reinterpret_cast<float**>(ptr_C_dev.data_ptr<int64_t>()),
      reinterpret_cast<float**>(ptr_C_dev.data_ptr<int64_t>()),
      lda_dev.data_ptr<int64_t>(), ldb_dev.data_ptr<int64_t>(),
      ldc_dev.data_ptr<int64_t>(), ldd_dev.data_ptr<int64_t>(),
      problem_sizes_host);
  cutlass::Status st = op.initialize(args, workspace, stream);
  if (st != cutlass::Status::kSuccess) return st;
  return op.run(stream);
}

// Grouped 4M complex matmul over G heterogeneous groups. Per group g:
//   ReC_g = +ReA_g.ReB_g - ImA_g.ImB_g ; ImC_g = +ReA_g.ImB_g + ImA_g.ReB_g
// Implemented as 4 real GemmGrouped passes over all G groups; passes 2/4
// accumulate (beta=+1) into the ReC/ImC buffers filled by passes 1/3.
// Returns (ReC_list, ImC_list) — per-group FP32 CUDA tensors of shape (M_g, N_g).
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>>
cutlass_grouped_4m(
    const std::vector<at::Tensor>& ReA_list,
    const std::vector<at::Tensor>& ImA_list,
    const std::vector<at::Tensor>& ReB_list,
    const std::vector<at::Tensor>& ImB_list) {
  TORCH_CHECK(!ReA_list.empty(), "cutlass_grouped_4m: empty ReA_list");
  int G = (int)ReA_list.size();
  TORCH_CHECK((int)ImA_list.size() == G && (int)ReB_list.size() == G &&
              (int)ImB_list.size() == G,
              "cutlass_grouped_4m: all four input lists must have the same length");
  for (int g = 0; g < G; ++g) {
    TORCH_CHECK(ReA_list[g].is_cuda() && ReA_list[g].dtype() == at::kBFloat16,
                "cutlass_grouped_4m: ReA[", g, "] must be CUDA BF16");
  }
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  // Per-group problem sizes + leading dims + output buffers.
  std::vector<cutlass::gemm::GemmCoord> problem_sizes_host(G);
  std::vector<int64_t> lda_h(G), ldb_h(G), ldc_h(G), ldd_h(G);
  std::vector<at::Tensor> ReC_list(G), ImC_list(G);
  for (int g = 0; g < G; ++g) {
    int M = (int)ReA_list[g].size(0), K = (int)ReA_list[g].size(1);
    int N = (int)ReB_list[g].size(1);
    TORCH_CHECK((int)ImA_list[g].size(0) == M && (int)ImA_list[g].size(1) == K,
                "cutlass_grouped_4m: ImA[", g, "] shape must match ReA");
    TORCH_CHECK((int)ReB_list[g].size(0) == K, "cutlass_grouped_4m: ReB[", g, "] rows must equal K");
    TORCH_CHECK((int)ImB_list[g].size(0) == K && (int)ImB_list[g].size(1) == N,
                "cutlass_grouped_4m: ImB[", g, "] shape must match ReB");
    problem_sizes_host[g] = cutlass::gemm::GemmCoord(M, N, K);
    lda_h[g] = K;  // RowMajor A stride
    ldb_h[g] = N;  // RowMajor B stride
    ldc_h[g] = N;  // RowMajor C stride
    ldd_h[g] = N;
    ReC_list[g] = at::empty({M, N}, at::dtype(at::kFloat).device(at::kCUDA));
    ImC_list[g] = at::empty({M, N}, at::dtype(at::kFloat).device(at::kCUDA));
  }

  // SM occupancy check — returns 0 if the kernel can't run on this device.
  int threadblock_count = GroupedGemm::sufficient(problem_sizes_host.data(), G);
  TORCH_CHECK(threadblock_count > 0,
              "cutlass_grouped_4m: GroupedGemm::sufficient returned 0 "
              "(SM occupancy / hw constraint on this device)");

  // Copy problem_sizes to device (GemmCoord is trivially copyable, 12 bytes).
  auto ps_cpu = at::empty({(int64_t)(G * sizeof(cutlass::gemm::GemmCoord))},
                          at::TensorOptions().dtype(at::kByte).device(at::kCPU));
  std::memcpy(ps_cpu.data_ptr(), problem_sizes_host.data(),
              G * sizeof(cutlass::gemm::GemmCoord));
  at::Tensor ps_dev = ps_cpu.to(at::kCUDA);
  cutlass::gemm::GemmCoord* problem_sizes_device =
      reinterpret_cast<cutlass::gemm::GemmCoord*>(ps_dev.data_ptr());

  at::Tensor lda_dev = _grouped_int64_to_device(lda_h);
  at::Tensor ldb_dev = _grouped_int64_to_device(ldb_h);
  at::Tensor ldc_dev = _grouped_int64_to_device(ldc_h);
  at::Tensor ldd_dev = _grouped_int64_to_device(ldd_h);

  at::Tensor ptr_ReA = _grouped_ptrs_to_device(ReA_list);
  at::Tensor ptr_ImA = _grouped_ptrs_to_device(ImA_list);
  at::Tensor ptr_ReB = _grouped_ptrs_to_device(ReB_list);
  at::Tensor ptr_ImB = _grouped_ptrs_to_device(ImB_list);
  at::Tensor ptr_ReC = _grouped_ptrs_to_device(ReC_list);
  at::Tensor ptr_ImC = _grouped_ptrs_to_device(ImC_list);

  // Workspace: same shapes across all 4 passes, so size once with a probe Arguments.
  typename GroupedGemm::EpilogueOutputOp::Params probe_epilogue{1.0f, 0.0f};
  typename GroupedGemm::Arguments probe_args(
      problem_sizes_device, G, threadblock_count, probe_epilogue,
      reinterpret_cast<cutlass::bfloat16_t**>(ptr_ReA.data_ptr<int64_t>()),
      reinterpret_cast<cutlass::bfloat16_t**>(ptr_ReB.data_ptr<int64_t>()),
      reinterpret_cast<float**>(ptr_ReC.data_ptr<int64_t>()),
      reinterpret_cast<float**>(ptr_ReC.data_ptr<int64_t>()),
      lda_dev.data_ptr<int64_t>(), ldb_dev.data_ptr<int64_t>(),
      ldc_dev.data_ptr<int64_t>(), ldd_dev.data_ptr<int64_t>(),
      problem_sizes_host.data());
  size_t ws_bytes = GroupedGemm::get_workspace_size(probe_args);
  at::Tensor workspace = at::empty({(int64_t)ws_bytes},
                                   at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
  void* ws_ptr = ws_bytes ? workspace.data_ptr() : nullptr;

  // Pass 1: ReC = +1*(ReA.ReB) + 0*ReC
  cutlass::Status st = _grouped_pass(
      G, threadblock_count, problem_sizes_device, problem_sizes_host.data(),
      ptr_ReA, ptr_ReB, ptr_ReC, lda_dev, ldb_dev, ldc_dev, ldd_dev,
      +1.0f, 0.0f, ws_ptr, stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess,
              "cutlass_grouped_4m pass 1 (ReC=ReA.ReB) failed: ", grouped_status_name(st));
  // Pass 2: ReC = -1*(ImA.ImB) + 1*ReC
  st = _grouped_pass(
      G, threadblock_count, problem_sizes_device, problem_sizes_host.data(),
      ptr_ImA, ptr_ImB, ptr_ReC, lda_dev, ldb_dev, ldc_dev, ldd_dev,
      -1.0f, +1.0f, ws_ptr, stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess,
              "cutlass_grouped_4m pass 2 (ReC-=ImA.ImB) failed: ", grouped_status_name(st));
  // Pass 3: ImC = +1*(ReA.ImB) + 0*ImC
  st = _grouped_pass(
      G, threadblock_count, problem_sizes_device, problem_sizes_host.data(),
      ptr_ReA, ptr_ImB, ptr_ImC, lda_dev, ldb_dev, ldc_dev, ldd_dev,
      +1.0f, 0.0f, ws_ptr, stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess,
              "cutlass_grouped_4m pass 3 (ImC=ReA.ImB) failed: ", grouped_status_name(st));
  // Pass 4: ImC = +1*(ImA.ReB) + 1*ImC
  st = _grouped_pass(
      G, threadblock_count, problem_sizes_device, problem_sizes_host.data(),
      ptr_ImA, ptr_ReB, ptr_ImC, lda_dev, ldb_dev, ldc_dev, ldd_dev,
      +1.0f, +1.0f, ws_ptr, stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess,
              "cutlass_grouped_4m pass 4 (ImC+=ImA.ReB) failed: ", grouped_status_name(st));

  return {ReC_list, ImC_list};
}

#define HAS_CUTLASS_GROUPED_4M 1
#else
#define HAS_CUTLASS_GROUPED_4M 0
#endif  // CUTLASS_ENABLE_GROUPED_4M

// Exposed to Python so _attempt_sm100_then_sm80 can distinguish "build
// succeeded but the Sm100 path was compiled out" (guard never set) from a
// real Sm100 kernel being present.
bool has_sm100() { return HAS_CUTLASS_4M_SM100; }
bool has_sm120() { return HAS_CUTLASS_4M_SM120; }
bool has_grouped_4m() { return HAS_CUTLASS_GROUPED_4M; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("probe", &probe, "CUTLASS build smoke");
    m.def("cutlass_4m_sm80", &cutlass_4m_sm80, "2.x Sm80 planar-complex 4M GEMM");
    m.def("real_gemm_workspace_bytes", &real_gemm_workspace_bytes, "");
    m.def("has_sm100", &has_sm100,
          "whether the 3.x Sm100 4M kernel compiled into this build");
    m.def("has_sm120", &has_sm120,
          "whether the native 3.x Sm120 4M kernel compiled into this build");
    m.def("has_grouped_4m", &has_grouped_4m,
          "whether the 2.x GemmGrouped 4M kernel compiled into this build");
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
    m.def("cutlass_grouped_4m",
          [](std::vector<at::Tensor> ReA, std::vector<at::Tensor> ImA,
             std::vector<at::Tensor> ReB, std::vector<at::Tensor> ImB)
              -> std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> {
#if HAS_CUTLASS_GROUPED_4M
              return cutlass_grouped_4m(ReA, ImA, ReB, ImB);
#else
              TORCH_CHECK(false, "grouped 4M not enabled in this build");
#endif
          },
          "2.x GemmGrouped planar-complex 4M over heterogeneous shapes (Task 5)");
}
