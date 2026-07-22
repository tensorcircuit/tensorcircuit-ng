// pybind11 extension: planar-complex BF16 cublasLt probe for Phase 0 Plan B.
// Build via torch.utils.cpp_extension (see _phase0_cublaslt_build.py).
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
// Mixed precision: A/B are BF16 (input compression — the leverage under test);
// C/D are FP32 (output preserved at full precision so the BF16 input
// quantization is the ONLY error source vs the FP32 reference, making the
// 1e-2 correctness gate meaningful). COMPUTE_32F accumulates in FP32.
// Host inputs: ar/ai (m,k), br/bi (k,n) as raw uint16 BF16 views.
// Returns (cr, ci) as float32 host arrays shaped (m,n).
static py::tuple planar_complex_matmul_bf16(
    py::array_t<uint16_t, py::array::c_style> ar_u16,
    py::array_t<uint16_t, py::array::c_style> ai_u16,
    py::array_t<uint16_t, py::array::c_style> br_u16,
    py::array_t<uint16_t, py::array::c_style> bi_u16,
    int m, int n, int k)
{
    constexpr size_t bf16_elem = 2;   // BF16 = 2 bytes  (A, B planes)
    constexpr size_t f32_elem  = 4;   // FP32 = 4 bytes  (C, D planes)
    size_t bytesA = (size_t)m * k * bf16_elem;  // A_h real/imag plane bytes
    size_t bytesB = (size_t)k * n * bf16_elem;  // B_h real/imag plane bytes
    size_t bytesC = (size_t)m * n * f32_elem;   // C_h real/imag plane bytes (FP32 out)

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
    //    A/B inputs are BF16; C/D output is FP32 (mixed precision).
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    check_cublas(make_planar_layout(&Adesc, CUDA_C_16BF, /*rows=*/n, /*cols=*/k, /*ld=*/n, off_A),
                 "Adesc create/set");
    check_cublas(make_planar_layout(&Bdesc, CUDA_C_16BF, /*rows=*/k, /*cols=*/m, /*ld=*/k, off_B),
                 "Bdesc create/set");
    check_cublas(make_planar_layout(&Cdesc, CUDA_C_32F,  /*rows=*/n, /*cols=*/m, /*ld=*/n, off_C),
                 "Cdesc create/set");

    // 4. Matmul descriptor: COMPUTE_32F (FP32 accumulate) + scaleType CUDA_C_32F.
    //    A/B=CUDA_C_16BF, C/D=CUDA_C_32F (declared via the layout dtypes).
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

    // Download C real/imag planes (each m*n FP32 values, laid out as col-major
    // n rows x m cols ld=n — byte-identical to row-major (m,n) C_h, see header).
    py::array_t<float> cr_f({m, n}), ci_f({m, n});
    cudaMemcpy(cr_f.mutable_data(), d_C, bytesC, cudaMemcpyDeviceToHost);
    cudaMemcpy(ci_f.mutable_data(), (char*)d_C + off_C, bytesC, cudaMemcpyDeviceToHost);

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

    return py::make_tuple(cr_f, ci_f);
}

// Enumerate algorithms for planar complex BF16-in / FP32-out + COMPUTE_32F
// WITHOUT executing. Tests the SAME mixed-precision configuration as
// planar_complex_matmul_bf16 so the algo_count reflects what the matmul path
// can actually use. Returns {algo_count, first_algo_id, workspace_bytes,
// heuristic_status, status}.
static py::dict probe_planar_capability(int m, int n, int k) {
    py::dict d;
    constexpr size_t bf16_elem = 2;
    constexpr size_t f32_elem  = 4;
    size_t bytesA = (size_t)m * k * bf16_elem;  // BF16 in
    size_t bytesB = (size_t)k * n * bf16_elem;  // BF16 in
    size_t bytesC = (size_t)m * n * f32_elem;   // FP32 out
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
    make_planar_layout(&Cdesc, CUDA_C_32F,  n, m, n, off_C);

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

PYBIND11_MODULE(_phase0_cublaslt_ext, m) {
    m.def("smoke_add", &smoke_add);
    m.def("cublaslt_info", &cublaslt_info);
    m.def("planar_complex_matmul_bf16", &planar_complex_matmul_bf16,
          py::arg("ar_u16"), py::arg("ai_u16"),
          py::arg("br_u16"), py::arg("bi_u16"),
          py::arg("m"), py::arg("n"), py::arg("k"));
    m.def("probe_planar_capability", &probe_planar_capability,
          py::arg("m"), py::arg("n"), py::arg("k"));
}
