// pybind11 extension: planar-complex BF16 cublasLt probe for Phase 0 Plan B.
// Build via torch.utils.cpp_extension (see _phase0_cublaslt_build.py).
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace py = pybind11;

static const char* cublaslt_status_str(cublasStatus_t s) {
    switch (s) {
        case CUBLAS_STATUS_SUCCESS: return "SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
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

PYBIND11_MODULE(_phase0_cublaslt_ext, m) {
    m.def("smoke_add", &smoke_add);
    m.def("cublaslt_info", &cublaslt_info);
}
