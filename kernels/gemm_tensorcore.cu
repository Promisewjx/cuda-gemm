#include <mma.h>  // 包含Tensor Core的API
#include <cuda_fp16.h>  // 用half数据类型

using namespace nvcuda; // 使用wmma命名空间

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void gemm_tensorcore(const half* A, const half* B, float* C, int N) {
    // 1. 计算这个warp负责哪一个Tile
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x);

    // 2. 声明fragment
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f); // 初始化累加器为0

    // 3. 遍历K方向，累加多个Tile
    for (int i = 0; i < N; i += WMMA_K) {
        // 检查边界
        if ((warpM * WMMA_M < N) && (warpN * WMMA_N < N) && (i < N)) {
            // Load A子块
            const half* tile_ptr_A = A + warpM * WMMA_M * N + i;
            wmma::load_matrix_sync(a_frag, tile_ptr_A, N);

            // Load B子块
            const half* tile_ptr_B = B + i * N + warpN * WMMA_N;
            wmma::load_matrix_sync(b_frag, tile_ptr_B, N);

            // Tensor Core GEMM: C += A * B
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    // 4. 写回C矩阵
    if (warpM * WMMA_M < N && warpN * WMMA_N < N) {
        float* tile_ptr_C = C + warpM * WMMA_M * N + warpN * WMMA_N;
        wmma::store_matrix_sync(tile_ptr_C, c_frag, N, wmma::mem_row_major);
    }
}
