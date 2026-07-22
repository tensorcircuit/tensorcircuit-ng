// Task 8 CUTLASS SM120 4M kernels. Built via torch.utils.cpp_extension
// (CUDA_HOME=nvcc_spike, -I<CUTLASS_ROOT>/include). Entry points added per task.
#include <torch/extension.h>
#include "cutlass/cutlass.h"

// Smoke entry (replaced by real kernels in later tasks). Forces CUTLASS include resolution.
int probe() { return 42; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("probe", &probe, "CUTLASS build smoke");
}
