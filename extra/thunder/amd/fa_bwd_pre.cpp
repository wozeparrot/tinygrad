#include "kittens.cuh"

#ifndef ATTN_B
constexpr int ATTN_B = 16; // batch size
#endif

#ifndef ATTN_H
constexpr int ATTN_H = 64; // number of query heads
#endif

#ifndef ATTN_N
constexpr int ATTN_N = 1024; // sequence length
#endif

constexpr int ATTN_D = 128; // dimension
constexpr int DOT_SLICE_QO = 16;

#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;

template<int D, typename T=bf16, typename L=row_l, typename S=rt_16x32_s> using qo_tile = rt<T, DOT_SLICE_QO, D, L, S>;

template<int D> struct attn_prep_globals { 
    gl<bf16, -1, -1, -1, -1> Og;
    gl<bf16, -1, -1, -1, -1> dOg; 
    gl<float, -1, -1, -1, -1> delta;
    dim3 grid() { return dim3(ATTN_B, ATTN_H, ATTN_N / (DOT_SLICE_QO * NUM_WARPS)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

template<int D> __launch_bounds__(NUM_THREADS, 1)
__global__ void attend_prep_ker(float *delta_ptr, bf16 *dq_ptr, bf16 *O_ptr, bf16 *dO_ptr) {
    gl<float, -1, -1, -1, -1> delta{delta_ptr, ATTN_B, ATTN_H, 1, ATTN_N};
    gl<bf16, -1, -1, -1, -1> dQg{dq_ptr, ATTN_B, ATTN_H, ATTN_N, ATTN_D};
    gl<bf16, -1, -1, -1, -1> Og{O_ptr, ATTN_B, ATTN_N, ATTN_H, ATTN_D};
    gl<bf16, -1, -1, -1, -1> dOg{dO_ptr, ATTN_B, ATTN_N, ATTN_H, ATTN_D};
    attn_prep_globals<D> g{Og, dOg, delta};

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.z;

    const int warpid = kittens::warpid();

    qo_tile<D, bf16, row_l, rt_16x32_s> dO, O;
    qo_tile<D, float, row_l, rt_16x32_s> dO_float, O_float;
    typename qo_tile<D, float, row_l, rt_16x32_s>::col_vec delta_vec;

    load<1>(dO, g.dOg, {batch_idx, seq_idx * NUM_WARPS + warpid, head_idx, 0});
    load<1>(O,  g.Og,  {batch_idx, seq_idx * NUM_WARPS + warpid, head_idx, 0});
    copy(O_float, O);
    copy(dO_float, dO);

    // Δ_i = row_sum(dO ⊙ O)
    mul(dO_float, dO_float, O_float);
    row_sum(delta_vec, dO_float);
    store(g.delta, delta_vec, {batch_idx, head_idx, 0, seq_idx * NUM_WARPS + warpid});

    // Zero out dq
    qo_tile<D, bf16, row_l, rt_16x32_s> dQ_zero;
    zero(dQ_zero);
    store<2>(dQg, dQ_zero, {batch_idx, head_idx, seq_idx * NUM_WARPS + warpid, 0});
}

template __global__ void attend_prep_ker<ATTN_D>(float *delta_ptr, bf16 *dq_ptr, bf16 *O_ptr, bf16 *dO_ptr);
