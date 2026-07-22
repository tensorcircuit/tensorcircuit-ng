// Minimal sm_120 BF16 Tensor Core probe — does nvcc accept wmma bf16 for compute capability 12.0?
#include <cuda_runtime.h>
#include <mma.h>  // wmma
using namespace nvcuda;
__global__ void probe_kernel() {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::load_matrix_sync(a, nullptr, 16); wmma::load_matrix_sync(b, nullptr, 16);
    wmma::fill_fragment(c, 0.0f); wmma::mma_sync(c, a, b, c);
}
int main() { return 0; }
