#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BLOCK_SIZE 128  // block tile size

__global__ void gemm_tensorcore_large_tile(const half* A, const half* B, float* C, int N) {
    extern __shared__ half shared_mem[];  // 声明动态共享内存

    // 把共享内存切分成两部分，分别存A_tile和B_tile
    half* As = shared_mem;                         // A tile起始位置
    half* Bs = shared_mem + BLOCK_SIZE * BLOCK_SIZE; // B tile起始位置

    int block_tile_i = blockIdx.y;
    int block_tile_j = blockIdx.x;

    int warp_id = threadIdx.x / 32;
    // int lane_id = threadIdx.x % 32;

    int warp_row = warp_id / (BLOCK_SIZE / WMMA_N);
    int warp_col = warp_id % (BLOCK_SIZE / WMMA_N);

    int row = block_tile_i * BLOCK_SIZE + warp_row * WMMA_M;
    int col = block_tile_j * BLOCK_SIZE + warp_col * WMMA_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // 每次循环处理K方向上的一小块（每次步进16）
    for (int k_tile = 0; k_tile < N; k_tile += BLOCK_SIZE) {
        // ====== Step 1: 搬数据到Shared Memory ======
        // 所有线程一起合作搬数据，每个线程搬多个元素
        for (int i = threadIdx.x; i < BLOCK_SIZE * BLOCK_SIZE * 2; i += blockDim.x) {
            int matrix_idx = i / (BLOCK_SIZE * BLOCK_SIZE);  // 0=A，1=B
            int idx_in_tile = i % (BLOCK_SIZE * BLOCK_SIZE);
            int tile_row = idx_in_tile / BLOCK_SIZE;
            int tile_col = idx_in_tile % BLOCK_SIZE;

            if (matrix_idx == 0) {  // A矩阵
                int global_row = block_tile_i * BLOCK_SIZE + tile_row;
                int global_col = k_tile + tile_col;
                if (global_row < N && global_col < N)
                    As[tile_row * BLOCK_SIZE + tile_col] = A[global_row * N + global_col];
                else
                    As[tile_row * BLOCK_SIZE + tile_col] = __float2half(0.0f);
            } else {  // B矩阵
                int global_row = k_tile + tile_row;
                int global_col = block_tile_j * BLOCK_SIZE + tile_col;
                if (global_row < N && global_col < N)
                    Bs[tile_row * BLOCK_SIZE + tile_col] = B[global_row * N + global_col];
                else
                    Bs[tile_row * BLOCK_SIZE + tile_col] = __float2half(0.0f);
            }
        }

        __syncthreads();  // 等待所有线程搬完

        // ====== Step 2: Warp内部做Tensor Core计算 ======
        for (int k_inner = 0; k_inner < BLOCK_SIZE; k_inner += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

            // 读取shared memory里的tile
            wmma::load_matrix_sync(a_frag, As + warp_row * WMMA_M * BLOCK_SIZE + k_inner, BLOCK_SIZE);
            wmma::load_matrix_sync(b_frag, Bs + k_inner * BLOCK_SIZE + warp_col * WMMA_N, BLOCK_SIZE);

            // Tensor Core矩阵乘累加
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        __syncthreads();  // 保证一个tile处理完再处理下一个tile
    }

    // ====== Step 3: 写回结果到global memory ======
    if (row < N && col < N) {
        wmma::store_matrix_sync(C + row * N + col, c_frag, N, wmma::mem_row_major);
    }
}
