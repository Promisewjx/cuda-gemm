#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void gemm_tiled_register_blocking_db(const float* A, const float* B, float* C, int N) {
    // Double Buffer：两个Tile缓冲区
    __shared__ float tile_A[2][TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[2][TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y * 2;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x * 2;

    float regC00 = 0.0f, regC01 = 0.0f;
    float regC10 = 0.0f, regC11 = 0.0f;

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Prefetch第一块tile
    int t = 0;
    int load_buffer = 0;
    int compute_buffer = 0;

    if (row + 0 < N && t * TILE_SIZE + threadIdx.x < N)
        tile_A[load_buffer][threadIdx.y * 2 + 0][threadIdx.x] = A[(row + 0) * N + t * TILE_SIZE + threadIdx.x];
    else
        tile_A[load_buffer][threadIdx.y * 2 + 0][threadIdx.x] = 0.0f;

    if (row + 1 < N && t * TILE_SIZE + threadIdx.x < N)
        tile_A[load_buffer][threadIdx.y * 2 + 1][threadIdx.x] = A[(row + 1) * N + t * TILE_SIZE + threadIdx.x];
    else
        tile_A[load_buffer][threadIdx.y * 2 + 1][threadIdx.x] = 0.0f;

    if (col + 0 < N && t * TILE_SIZE + threadIdx.y < N)
        tile_B[load_buffer][threadIdx.y][threadIdx.x * 2 + 0] = B[(t * TILE_SIZE + threadIdx.y) * N + col + 0];
    else
        tile_B[load_buffer][threadIdx.y][threadIdx.x * 2 + 0] = 0.0f;

    if (col + 1 < N && t * TILE_SIZE + threadIdx.y < N)
        tile_B[load_buffer][threadIdx.y][threadIdx.x * 2 + 1] = B[(t * TILE_SIZE + threadIdx.y) * N + col + 1];
    else
        tile_B[load_buffer][threadIdx.y][threadIdx.x * 2 + 1] = 0.0f;

    __syncthreads();

    for (t = 1; t < numTiles; ++t) {
        // 交换buffer
        compute_buffer = load_buffer;
        load_buffer = 1 - load_buffer;

        // Prefetch下一块tile（搬到另一个buffer）
        if (row + 0 < N && t * TILE_SIZE + threadIdx.x < N)
            tile_A[load_buffer][threadIdx.y * 2 + 0][threadIdx.x] = A[(row + 0) * N + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[load_buffer][threadIdx.y * 2 + 0][threadIdx.x] = 0.0f;

        if (row + 1 < N && t * TILE_SIZE + threadIdx.x < N)
            tile_A[load_buffer][threadIdx.y * 2 + 1][threadIdx.x] = A[(row + 1) * N + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[load_buffer][threadIdx.y * 2 + 1][threadIdx.x] = 0.0f;

        if (col + 0 < N && t * TILE_SIZE + threadIdx.y < N)
            tile_B[load_buffer][threadIdx.y][threadIdx.x * 2 + 0] = B[(t * TILE_SIZE + threadIdx.y) * N + col + 0];
        else
            tile_B[load_buffer][threadIdx.y][threadIdx.x * 2 + 0] = 0.0f;

        if (col + 1 < N && t * TILE_SIZE + threadIdx.y < N)
            tile_B[load_buffer][threadIdx.y][threadIdx.x * 2 + 1] = B[(t * TILE_SIZE + threadIdx.y) * N + col + 1];
        else
            tile_B[load_buffer][threadIdx.y][threadIdx.x * 2 + 1] = 0.0f;

        // 计算当前tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            float regA0 = tile_A[compute_buffer][threadIdx.y * 2 + 0][k];
            float regA1 = tile_A[compute_buffer][threadIdx.y * 2 + 1][k];
            float regB0 = tile_B[compute_buffer][k][threadIdx.x * 2 + 0];
            float regB1 = tile_B[compute_buffer][k][threadIdx.x * 2 + 1];

            regC00 += regA0 * regB0;
            regC01 += regA0 * regB1;
            regC10 += regA1 * regB0;
            regC11 += regA1 * regB1;
        }

        __syncthreads();
    }

    // 计算最后一个tile
    compute_buffer = load_buffer;
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        float regA0 = tile_A[compute_buffer][threadIdx.y * 2 + 0][k];
        float regA1 = tile_A[compute_buffer][threadIdx.y * 2 + 1][k];
        float regB0 = tile_B[compute_buffer][k][threadIdx.x * 2 + 0];
        float regB1 = tile_B[compute_buffer][k][threadIdx.x * 2 + 1];

        regC00 += regA0 * regB0;
        regC01 += regA0 * regB1;
        regC10 += regA1 * regB0;
        regC11 += regA1 * regB1;
    }

    // 写回global memory
    if (row + 0 < N && col + 0 < N)
        C[(row + 0) * N + col + 0] = regC00;
    if (row + 0 < N && col + 1 < N)
        C[(row + 0) * N + col + 1] = regC01;
    if (row + 1 < N && col + 0 < N)
        C[(row + 1) * N + col + 0] = regC10;
    if (row + 1 < N && col + 1 < N)
        C[(row + 1) * N + col + 1] = regC11;
}