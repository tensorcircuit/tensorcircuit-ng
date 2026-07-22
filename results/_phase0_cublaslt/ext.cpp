// pybind11 extension: planar-complex BF16 cublasLt probe for Phase 0 Plan B.
// Build via torch.utils.cpp_extension (see _phase0_cublaslt_build.py).
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace py = pybind11;

static const char* cublaslt_status_str(cublasStatus_t s) {
    switch (s) {
        case CUBLAS_STATUS_SUCCESS: return "SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
        default: return "OTHER";
    }
}

// Smoke fn (Task 1): confirms the extension compiles + cublasLt header resolves.
static int smoke_add(int a, int b) { return a + b; }

// Report the linked cublasLt version + whether planar C16BF enums resolve.
static py::dict cublaslt_info() {
    cublasLtHandle_t h = nullptr;
    cublasStatus_t s = cublasLtCreate(&h);
    py::dict d;
    d["cublasLtCreate"] = cublaslt_status_str(s);
    d["has_plane_offset_attr"] = true;  // CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET is an enum in cublasLt.h
    d["compute_32f_value"] = (int)CUBLAS_COMPUTE_32F;
    d["c_16bf_value"] = (int)CUDA_C_16BF;
    if (s == CUBLAS_STATUS_SUCCESS) cublasLtDestroy(h);
    return d;
}

// ============================================================================
// Planar complex BF16 matmul via ONE cublasLtMatmul call.
//
// Planar-complex layout: one device allocation per operand laid out as
//   [ real_plane | pad-to-256B-align | imag_plane ]
// with CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET set to the imag plane's byte offset.
//
// Row/column-major convention: host numpy arrays are row-major
//   A_h (m,k), B_h (k,n), C_h = A_h . B_h  (m,n).
// cublasLt is column-major, so we use the standard rowmajor<->colmajor swap:
//   rowmajor(A.B) == colmajor(B^T . A^T).
// Therefore the cublasLt operands (column-major) are:
//   A_cublas = B_h^T   (rows=n, cols=k, ld=n)   data bytes <- br/bi
//   B_cublas = A_h^T   (rows=k, cols=m, ld=k)   data bytes <- ar/ai
//   D_cublas = C_h^T   (rows=n, cols=m, ld=n)   data bytes -> cr/ci
// After download, D_cublas's real/imag plane bytes are byte-identical to the
// row-major (m,n) C_h real/imag arrays (same stride n, just transposed shape),
// so no host-side transpose is needed.
//
// cublasLt with CUDA_C_16BF + PLANE_OFFSET performs the full complex matmul
// (4-real-matmul fusion: Cr=Ar.Br-Ai.Bi ; Ci=Ar.Bi+Ai.Br) in a single call.
// ============================================================================

static inline size_t align256(size_t x) { return (x + (size_t)255) & ~(size_t)255; }

// Build the planar layout for a column-major operand with the given
// (rows, cols, ld) and the imag-plane byte offset. Returns the layout handle.
static cublasStatus_t make_planar_layout(
    cublasLtMatrixLayout_t* layout,
    cudaDataType_t dtype,
    int rows, int cols, int ld,
    size_t imag_offset_bytes)
{
    cublasStatus_t s = cublasLtMatrixLayoutCreate(layout, dtype,
                                                   (uint32_t)rows,
                                                   (uint32_t)cols,
                                                   (int)ld);
    if (s != CUBLAS_STATUS_SUCCESS) return s;
    // CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET: byte offset of the imaginary plane
    // relative to the matrix data pointer. Must be 256-byte aligned (we align).
    int64_t off = (int64_t)imag_offset_bytes;
    s = cublasLtMatrixLayoutSetAttribute(*layout,
        CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET, &off, sizeof(off));
    return s;
}

// Planar complex BF16 matmul: C = A . B  (complex).
// A/B are BF16 (input compression — the leverage under test); COMPUTE_32F
// accumulates in FP32. The output dtype is selected by out_dtype:
//   "bf16" (default): C/D = CUDA_C_16BF — spec-compliant end-to-end BF16
//     (output compression is half the leverage). Returns (cr, ci) as raw
//     uint16 BF16 views shaped (m,n); the driver upcasts to float32 for
//     comparison. BF16 output inherently rounds to ~8 mantissa bits, so the
//     correctness gate for this path is max-RELATIVE-error (< 1e-2).
//   "fp32": C/D = CUDA_C_32F — mixed-precision cross-check. Returns (cr, ci)
//     as float32 host arrays; max-abs < 1e-2 (expected ~2e-4).
// Host inputs: ar/ai (m,k), br/bi (k,n) as raw uint16 BF16 views.
static py::tuple planar_complex_matmul_bf16(
    py::array_t<uint16_t, py::array::c_style> ar_u16,
    py::array_t<uint16_t, py::array::c_style> ai_u16,
    py::array_t<uint16_t, py::array::c_style> br_u16,
    py::array_t<uint16_t, py::array::c_style> bi_u16,
    int m, int n, int k,
    std::string out_dtype)
{
    constexpr size_t bf16_elem = 2;   // BF16 = 2 bytes  (A, B planes; and C/D when bf16 out)
    constexpr size_t f32_elem  = 4;   // FP32 = 4 bytes  (C, D planes when fp32 out)
    bool bf16_out = (out_dtype == "bf16");
    size_t out_elem = bf16_out ? bf16_elem : f32_elem;
    cudaDataType_t out_cdtype = bf16_out ? CUDA_C_16BF : CUDA_C_32F;
    size_t bytesA = (size_t)m * k * bf16_elem;  // A_h real/imag plane bytes
    size_t bytesB = (size_t)k * n * bf16_elem;  // B_h real/imag plane bytes
    size_t bytesC = (size_t)m * n * out_elem;   // C_h real/imag plane bytes (out dtype)

    // Plane offsets (256-B aligned). With the colmajor-swap convention:
    //   A_cublas = B_h^T (BF16 planar) -> plane size = bytesB
    //   B_cublas = A_h^T (BF16 planar) -> plane size = bytesA
    //   D_cublas = C_h^T (FP32 planar) -> plane size = bytesC
    size_t off_A = align256(bytesB);
    size_t off_B = align256(bytesA);
    size_t off_C = align256(bytesC);

    auto check_cuda = [&](cudaError_t e, const char* what) {
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string(what) + ": cudaError "
                + std::to_string((int)e));
        }
    };
    auto check_cublas = [&](cublasStatus_t e, const char* what) {
        if (e != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(std::string(what) + ": cublasStatus "
                + cublaslt_status_str(e));
        }
    };

    // 1. Allocate planar device buffers and stage real/imag planes.
    void *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    check_cuda(cudaMalloc(&d_A, off_A + bytesB), "cudaMalloc d_A");
    check_cuda(cudaMalloc(&d_B, off_B + bytesA), "cudaMalloc d_B");
    check_cuda(cudaMalloc(&d_C, off_C + bytesC), "cudaMalloc d_C");

    // A_cublas complex = B_h^T : real plane <- br, imag plane <- bi.
    const uint16_t* br_p = br_u16.data();
    const uint16_t* bi_p = bi_u16.data();
    check_cuda(cudaMemcpy(d_A, br_p, bytesB, cudaMemcpyHostToDevice), "H2D br");
    check_cuda(cudaMemcpy((char*)d_A + off_A, bi_p, bytesB, cudaMemcpyHostToDevice), "H2D bi");
    // B_cublas complex = A_h^T : real plane <- ar, imag plane <- ai.
    const uint16_t* ar_p = ar_u16.data();
    const uint16_t* ai_p = ai_u16.data();
    check_cuda(cudaMemcpy(d_B, ar_p, bytesA, cudaMemcpyHostToDevice), "H2D ar");
    check_cuda(cudaMemcpy((char*)d_B + off_B, ai_p, bytesA, cudaMemcpyHostToDevice), "H2D ai");
    // D plane left uninitialized (beta = 0).

    // 2. cublasLt handle.
    cublasLtHandle_t h = nullptr;
    check_cublas(cublasLtCreate(&h), "cublasLtCreate");

    // 3. Planar layouts. Column-major dimensions per the swap convention.
    //    A/B inputs are BF16; C/D output dtype is out_cdtype (bf16 or fp32).
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    check_cublas(make_planar_layout(&Adesc, CUDA_C_16BF, /*rows=*/n, /*cols=*/k, /*ld=*/n, off_A),
                 "Adesc create/set");
    check_cublas(make_planar_layout(&Bdesc, CUDA_C_16BF, /*rows=*/k, /*cols=*/m, /*ld=*/k, off_B),
                 "Bdesc create/set");
    check_cublas(make_planar_layout(&Cdesc, out_cdtype, /*rows=*/n, /*cols=*/m, /*ld=*/n, off_C),
                 "Cdesc create/set");

    // 4. Matmul descriptor: COMPUTE_32F (FP32 accumulate) + scaleType CUDA_C_32F
    //    (complex FP32 alpha/beta). A/B=CUDA_C_16BF; C/D=out_cdtype (layout dtypes).
    //    TRANSA/TRANSB default to CUBLAS_OP_N (no transpose).
    cublasLtMatmulDesc_t desc = nullptr;
    check_cublas(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_C_32F),
                 "MatmulDescCreate");

    // 5. Preference + heuristic algorithm enumeration.
    cublasLtMatmulPreference_t pref = nullptr;
    check_cublas(cublasLtMatmulPreferenceCreate(&pref), "PreferenceCreate");
    size_t ws_limit = 64ull * 1024 * 1024;  // allow up to 64 MB workspace
    check_cublas(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_limit, sizeof(ws_limit)),
        "PreferenceSetAttribute(max_workspace)");

    cublasLtMatmulHeuristicResult_t heur[8];
    std::memset(heur, 0, sizeof(heur));
    int returned = 0;
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(h, desc,
        Adesc, Bdesc, Cdesc, Cdesc, pref, 8, heur, &returned);
    if (hs != CUBLAS_STATUS_SUCCESS || returned == 0) {
        // Cleanup + surface a descriptive error so the driver records NOT_SUPPORTED.
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatmulDescDestroy(desc);
        cublasLtMatrixLayoutDestroy(Adesc);
        cublasLtMatrixLayoutDestroy(Bdesc);
        cublasLtMatrixLayoutDestroy(Cdesc);
        cublasLtDestroy(h);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        char buf[160];
        std::snprintf(buf, sizeof(buf),
            "cublasLtMatmulAlgoGetHeuristic returned no algo (status=%s, count=%d)",
            cublaslt_status_str(hs), returned);
        throw std::runtime_error(buf);
    }

    // 6. Allocate workspace sized to the chosen algo's requirement.
    void* workspace = nullptr;
    size_t ws_size = heur[0].workspaceSize;
    if (ws_size > 0) check_cuda(cudaMalloc(&workspace, ws_size), "cudaMalloc workspace");

    // 7. Execute: D = 1 * A_c . B_c + 0 * D_c (complex alpha/beta = 1+0j / 0+0j).
    float alpha[2] = {1.0f, 0.0f};
    float beta[2] = {0.0f, 0.0f};
    cublasStatus_t es = cublasLtMatmul(h, desc,
        alpha,
        d_A, Adesc,
        d_B, Bdesc,
        beta,
        d_C, Cdesc,   // C (src for beta term; beta=0 so unused)
        d_C, Cdesc,   // D (destination) == C pointer (in-place)
        &heur[0].algo,
        workspace, ws_size,
        0 /* default stream */);

    // Sync before download so results are visible.
    cudaError_t sync_e = cudaDeviceSynchronize();

    // Download C real/imag planes (each m*n elements of out_elem bytes, laid out
    // as col-major n rows x m cols ld=n — byte-identical to row-major (m,n) C_h,
    // see header). For bf16 output, return raw uint16 BF16 views; for fp32, float32.
    py::object cr_arr, ci_arr;
    if (bf16_out) {
        py::array_t<uint16_t> cr_u16({m, n}), ci_u16({m, n});
        cudaMemcpy(cr_u16.mutable_data(), d_C, bytesC, cudaMemcpyDeviceToHost);
        cudaMemcpy(ci_u16.mutable_data(), (char*)d_C + off_C, bytesC, cudaMemcpyDeviceToHost);
        cr_arr = cr_u16;
        ci_arr = ci_u16;
    } else {
        py::array_t<float> cr_f({m, n}), ci_f({m, n});
        cudaMemcpy(cr_f.mutable_data(), d_C, bytesC, cudaMemcpyDeviceToHost);
        cudaMemcpy(ci_f.mutable_data(), (char*)d_C + off_C, bytesC, cudaMemcpyDeviceToHost);
        cr_arr = cr_f;
        ci_arr = ci_f;
    }

    // Cleanup all GPU/handle resources.
    if (workspace) cudaFree(workspace);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtDestroy(h);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    check_cublas(es, "cublasLtMatmul");
    check_cuda(sync_e, "cudaDeviceSynchronize");

    return py::make_tuple(cr_arr, ci_arr);
}

// Enumerate algorithms for the spec-compliant planar-complex BF16-in / BF16-out
// + COMPUTE_32F config WITHOUT executing. The heuristic's C/D dtype
// (CUDA_C_16BF) matches the cublasLtMatmulAlgoGetIds C/D query (CUDA_C_16BF),
// so algo_count + first_algo_id are consistent for the BF16-output path — the
// config that matters for C3_planar. Returns {algo_count, first_algo_id,
// workspace_bytes, heuristic_status, status}.
static py::dict probe_planar_capability(int m, int n, int k) {
    py::dict d;
    constexpr size_t bf16_elem = 2;
    size_t bytesA = (size_t)m * k * bf16_elem;  // BF16 in
    size_t bytesB = (size_t)k * n * bf16_elem;  // BF16 in
    size_t bytesC = (size_t)m * n * bf16_elem;   // BF16 out (spec-compliant; matches AlgoGetIds C/D=CUDA_C_16BF)
    size_t off_A = align256(bytesB);
    size_t off_B = align256(bytesA);
    size_t off_C = align256(bytesC);

    cublasLtHandle_t h = nullptr;
    cublasStatus_t s = cublasLtCreate(&h);
    if (s != CUBLAS_STATUS_SUCCESS) {
        d["algo_count"] = 0;
        d["first_algo_id"] = -1;
        d["workspace_bytes"] = (long)0;
        d["heuristic_status"] = cublaslt_status_str(s);
        d["status"] = std::string("cublasLtCreate failed: ") + cublaslt_status_str(s);
        return d;
    }

    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    make_planar_layout(&Adesc, CUDA_C_16BF, n, k, n, off_A);
    make_planar_layout(&Bdesc, CUDA_C_16BF, k, m, k, off_B);
    make_planar_layout(&Cdesc, CUDA_C_16BF, n, m, n, off_C);

    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_C_32F);

    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t ws_limit = 64ull * 1024 * 1024;
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_limit, sizeof(ws_limit));

    cublasLtMatmulHeuristicResult_t heur[8];
    std::memset(heur, 0, sizeof(heur));
    int returned = 0;
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(h, desc,
        Adesc, Bdesc, Cdesc, Cdesc, pref, 8, heur, &returned);

    d["algo_count"] = returned;
    d["heuristic_status"] = cublaslt_status_str(hs);
    int first_id = -1;
    long first_ws = 0;
    if (returned > 0) {
        first_ws = (long)heur[0].workspaceSize;
        // cublasLt has no public "algo_t -> id" getter; enumerate IDs for this
        // configuration and report the first as a representative identifier.
        int ids[8] = {0};
        int nb_ids = 0;
        cublasLtMatmulAlgoGetIds(h, CUBLAS_COMPUTE_32F, CUDA_C_32F,
            CUDA_C_16BF, CUDA_C_16BF, CUDA_C_16BF, CUDA_C_16BF,
            8, ids, &nb_ids);
        if (nb_ids > 0) first_id = ids[0];
    }
    d["first_algo_id"] = first_id;
    d["workspace_bytes"] = first_ws;
    if (hs == CUBLAS_STATUS_SUCCESS && returned > 0) {
        d["status"] = "OK";
    } else {
        d["status"] = std::string("no algo: ") + cublaslt_status_str(hs);
    }

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtDestroy(h);
    return d;
}

// ============================================================================
// Kernel-only timing for the spec-compliant planar-complex BF16 path.
//
// The host API ``planar_complex_matmul_bf16`` recreates+destroys the handle,
// layouts, matmul desc, preference, workspace AND device buffers on every call,
// and does H2D(x4)+D2H(x2) per call — so timing it against the c64 kernel-only
// baseline (resident tensors, torch's cached handle) is unfair to planar. This
// function measures ONLY the cublasLtMatmul kernel: all handle/layout/desc/
// preference/algo/workspace setup happens ONCE up front, device buffers are
// allocated ONCE, BF16 inputs are uploaded ONCE (outside timing), and teardown
// happens ONCE at the end. The timed loop is cublasLtMatmul + event sync only.
//
// This is the fair §7.5 "production c64" gate: planar-kernel-only vs
// c64-kernel-only. BF16-output (C/D=C16BF, COMPUTE_32F) is the path timed.
//
// Returns dict {median_ms, algo_id, workspace_bytes, iters, warmup, status}.
// On no-algo, status describes the failure and median_ms=0 (driver records it).
// ============================================================================
static py::dict planar_complex_matmul_bf16_kernelonly_timing(
    py::array_t<uint16_t, py::array::c_style> ar_u16,
    py::array_t<uint16_t, py::array::c_style> ai_u16,
    py::array_t<uint16_t, py::array::c_style> br_u16,
    py::array_t<uint16_t, py::array::c_style> bi_u16,
    int m, int n, int k,
    int iters,
    int warmup)
{
    py::dict out;
    if (iters < 1) iters = 1;
    if (warmup < 0) warmup = 0;

    constexpr size_t bf16_elem = 2;
    size_t bytesA = (size_t)m * k * bf16_elem;  // A_h real/imag plane bytes
    size_t bytesB = (size_t)k * n * bf16_elem;  // B_h real/imag plane bytes
    size_t bytesC = (size_t)m * n * bf16_elem;  // BF16 out (spec-compliant)
    size_t off_A = align256(bytesB);
    size_t off_B = align256(bytesA);
    size_t off_C = align256(bytesC);

    auto check_cuda = [&](cudaError_t e, const char* what) {
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string(what) + ": cudaError "
                + std::to_string((int)e));
        }
    };
    auto check_cublas = [&](cublasStatus_t e, const char* what) {
        if (e != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(std::string(what) + ": cublasStatus "
                + cublaslt_status_str(e));
        }
    };

    // RAII-ish teardown: called on every exit path after setup begins. Uses a
    // flag set once each resource exists so double-free / free-null is impossible.
    void *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *workspace = nullptr;
    cublasLtHandle_t h = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cudaEvent_t ev_start = nullptr, ev_stop = nullptr;

    auto teardown = [&]() {
        if (ev_start) cudaEventDestroy(ev_start);
        if (ev_stop) cudaEventDestroy(ev_stop);
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (desc) cublasLtMatmulDescDestroy(desc);
        if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
        if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
        if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
        if (h) cublasLtDestroy(h);
        if (workspace) cudaFree(workspace);
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
    };

    // 1. Allocate planar device buffers + upload BF16 inputs ONCE (outside timing).
    check_cuda(cudaMalloc(&d_A, off_A + bytesB), "cudaMalloc d_A");
    check_cuda(cudaMalloc(&d_B, off_B + bytesA), "cudaMalloc d_B");
    check_cuda(cudaMalloc(&d_C, off_C + bytesC), "cudaMalloc d_C");
    // A_cublas = B_h^T: real <- br, imag <- bi ; B_cublas = A_h^T: real <- ar, imag <- ai.
    check_cuda(cudaMemcpy(d_A, br_u16.data(), bytesB, cudaMemcpyHostToDevice), "H2D br");
    check_cuda(cudaMemcpy((char*)d_A + off_A, bi_u16.data(), bytesB, cudaMemcpyHostToDevice), "H2D bi");
    check_cuda(cudaMemcpy(d_B, ar_u16.data(), bytesA, cudaMemcpyHostToDevice), "H2D ar");
    check_cuda(cudaMemcpy((char*)d_B + off_B, ai_u16.data(), bytesA, cudaMemcpyHostToDevice), "H2D ai");

    // 2. cublasLt handle ONCE.
    check_cublas(cublasLtCreate(&h), "cublasLtCreate");

    // 3. Planar layouts ONCE (column-major swap convention; BF16 in + BF16 out).
    check_cublas(make_planar_layout(&Adesc, CUDA_C_16BF, /*rows=*/n, /*cols=*/k, /*ld=*/n, off_A), "Adesc");
    check_cublas(make_planar_layout(&Bdesc, CUDA_C_16BF, /*rows=*/k, /*cols=*/m, /*ld=*/k, off_B), "Bdesc");
    check_cublas(make_planar_layout(&Cdesc, CUDA_C_16BF, /*rows=*/n, /*cols=*/m, /*ld=*/n, off_C), "Cdesc");

    // 4. Matmul desc ONCE: COMPUTE_32F (FP32 accumulate) + scaleType CUDA_C_32F.
    check_cublas(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_C_32F), "MatmulDescCreate");

    // 5. Preference + enumerate ONE algorithm ONCE.
    check_cublas(cublasLtMatmulPreferenceCreate(&pref), "PreferenceCreate");
    size_t ws_limit = 64ull * 1024 * 1024;
    check_cublas(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_limit, sizeof(ws_limit)),
        "PreferenceSetAttribute(max_workspace)");

    cublasLtMatmulHeuristicResult_t heur[8];
    std::memset(heur, 0, sizeof(heur));
    int returned = 0;
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(h, desc,
        Adesc, Bdesc, Cdesc, Cdesc, pref, 8, heur, &returned);
    if (hs != CUBLAS_STATUS_SUCCESS || returned == 0) {
        teardown();
        out["status"] = std::string("no algo: ") + cublaslt_status_str(hs);
        out["median_ms"] = 0.0;
        out["algo_id"] = -1;
        out["workspace_bytes"] = (long)0;
        out["iters"] = iters;
        out["warmup"] = warmup;
        return out;
    }

    // 6. Workspace ONCE, sized to the chosen algo's requirement.
    size_t ws_size = heur[0].workspaceSize;
    if (ws_size > 0) check_cuda(cudaMalloc(&workspace, ws_size), "cudaMalloc workspace");

    // Representative algo id (no public algo->id getter; enumerate ids for the
    // BF16-out config and report the first — same identifier probe_planar_capability uses).
    int first_id = -1;
    {
        int ids[8] = {0};
        int nb_ids = 0;
        cublasLtMatmulAlgoGetIds(h, CUBLAS_COMPUTE_32F, CUDA_C_32F,
            CUDA_C_16BF, CUDA_C_16BF, CUDA_C_16BF, CUDA_C_16BF,
            8, ids, &nb_ids);
        if (nb_ids > 0) first_id = ids[0];
    }

    // 7. Events for per-iteration GPU timing (default stream; matches the host
    //    API's matmul stream and torch's resident-data c64 baseline).
    check_cuda(cudaEventCreate(&ev_start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&ev_stop), "cudaEventCreate stop");

    float alpha[2] = {1.0f, 0.0f};
    float beta[2] = {0.0f, 0.0f};

    // 8. Warmup: kernel + sync only (first call may init algo-internal state).
    for (int i = 0; i < warmup; ++i) {
        cublasStatus_t es = cublasLtMatmul(h, desc, alpha,
            d_A, Adesc, d_B, Bdesc, beta,
            d_C, Cdesc, d_C, Cdesc,
            &heur[0].algo, workspace, ws_size, 0 /* default stream */);
        if (es != CUBLAS_STATUS_SUCCESS) {
            teardown();
            throw std::runtime_error(std::string("cublasLtMatmul warmup: ")
                + cublaslt_status_str(es));
        }
    }
    check_cuda(cudaStreamSynchronize(0), "warmup sync");

    // 9. Timed loop: record(start) -> cublasLtMatmul -> record(stop) -> sync.
    //    NO H2D/D2H, NO create/destroy in the loop. Collect per-iter ms, median.
    std::vector<float> times;
    times.reserve((size_t)iters);
    for (int i = 0; i < iters; ++i) {
        check_cuda(cudaEventRecord(ev_start, 0), "record start");
        cublasStatus_t es = cublasLtMatmul(h, desc, alpha,
            d_A, Adesc, d_B, Bdesc, beta,
            d_C, Cdesc, d_C, Cdesc,
            &heur[0].algo, workspace, ws_size, 0);
        check_cuda(cudaEventRecord(ev_stop, 0), "record stop");
        check_cuda(cudaEventSynchronize(ev_stop), "event sync");
        if (es != CUBLAS_STATUS_SUCCESS) {
            teardown();
            throw std::runtime_error(std::string("cublasLtMatmul timed: ")
                + cublaslt_status_str(es));
        }
        float ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&ms, ev_start, ev_stop), "elapsed");
        times.push_back(ms);
    }

    std::sort(times.begin(), times.end());
    float median_ms = times[times.size() / 2];

    // 10. Teardown ONCE.
    teardown();

    out["median_ms"] = (double)median_ms;
    out["algo_id"] = first_id;
    out["workspace_bytes"] = (long)ws_size;
    out["iters"] = iters;
    out["warmup"] = warmup;
    out["status"] = std::string("OK");
    return out;
}

PYBIND11_MODULE(_phase0_cublaslt_ext, m) {
    m.def("smoke_add", &smoke_add);
    m.def("cublaslt_info", &cublaslt_info);
    m.def("planar_complex_matmul_bf16", &planar_complex_matmul_bf16,
          py::arg("ar_u16"), py::arg("ai_u16"),
          py::arg("br_u16"), py::arg("bi_u16"),
          py::arg("m"), py::arg("n"), py::arg("k"),
          py::arg("out_dtype") = std::string("bf16"));
    m.def("probe_planar_capability", &probe_planar_capability,
          py::arg("m"), py::arg("n"), py::arg("k"));
    m.def("planar_complex_matmul_bf16_kernelonly_timing",
          &planar_complex_matmul_bf16_kernelonly_timing,
          py::arg("ar_u16"), py::arg("ai_u16"),
          py::arg("br_u16"), py::arg("bi_u16"),
          py::arg("m"), py::arg("n"), py::arg("k"),
          py::arg("iters") = 5,
          py::arg("warmup") = 3);
}
