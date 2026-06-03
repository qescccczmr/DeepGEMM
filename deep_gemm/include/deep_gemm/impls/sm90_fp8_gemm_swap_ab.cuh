#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/scheduler/gemm.cuh>

namespace deep_gemm {

template <cute::UMMA::Major kMajorSFB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleDMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs,
          GemmType kGemmType,
          typename epilogue_type_t>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_m_grouped_fp8_gemm_1d2d_swap_ab_impl(float* sfb, int* grouped_layout,
                                          uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                                          const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                                          const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                                          const __grid_constant__ cute::TmaDescriptor tensor_map_d,
                                          const __grid_constant__ cute::TmaDescriptor tensor_map_sfa) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::MGroupedContiguous,
                     "SM90 swap-AB path supports normal and m-grouped contiguous GEMMs");
    DG_STATIC_ASSERT(BLOCK_N == 32 or BLOCK_N == 64 or BLOCK_N == 128,
                     "SM90 swap-AB path requires BLOCK_N=32, 64, or 128");
    DG_STATIC_ASSERT(BLOCK_M >= 8 and BLOCK_M <= 256 and BLOCK_M % 8 == 0,
                     "SM90 swap-AB path supports BLOCK_M in [8, 256] and divisible by 8");
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT((BLOCK_N <= 64 and kNumMathThreads == 128) or (BLOCK_N == 128 and kNumMathThreads == 256),
                     "Swap-AB path needs one math warpgroup per 64 original-N columns");
    DG_STATIC_ASSERT(kSwizzleAMode == 128 and kSwizzleBMode == 128 and (kSwizzleDMode == 64 or kSwizzleDMode == 128),
                     "Minimal SM90 swap-AB path requires 128B A/B swizzling and 64B/128B D swizzling");
    DG_STATIC_ASSERT(kNumTMAMulticast <= 2, "Scheduler does not support > 2 TMA multicast");

    using WGMMA = typename mma::sm90::FP8MMASelector<BLOCK_M>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    constexpr uint32_t SMEM_D_SIZE = math::constexpr_align(BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(__nv_bfloat16)), 1024u);
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
    constexpr uint32_t ALIGNED_SMEM_SFA_SIZE_PER_STAGE = math::constexpr_align(SMEM_SFA_SIZE_PER_STAGE, 128u);
    constexpr uint32_t kNumElemBytes = sizeof(nv_bfloat16);
    constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode / kNumElemBytes;
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / kNumElemBytes;
    DG_STATIC_ASSERT(TMA_D_BLOCK_N == 32 or TMA_D_BLOCK_N == 64,
                     "Minimal SM90 swap-AB epilogue expects 32 or 64 BF16 cols per swizzle atom");

    const uint32_t shape_k_scales = math::ceil_div(shape_k, BLOCK_K);
    const uint32_t shape_n_sfb = math::ceil_div(shape_n, BLOCK_K);
    const uint32_t smem_sfb_size = math::align<uint32_t>(shape_k_scales * sizeof(float), sizeof(Barrier));
    const uint32_t num_total_k_blocks = math::ceil_div(shape_k, BLOCK_K);
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();

    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_d);
    }
    __syncwarp();

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    constexpr uint32_t SMEM_SF_OFFSET = SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + i * ALIGNED_SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + kNumStages * ALIGNED_SMEM_SFA_SIZE_PER_STAGE);

    auto barrier_start_ptr = reinterpret_cast<Barrier*>(reinterpret_cast<uint8_t*>(smem_sfb) + smem_sfb_size);
    auto full_barriers = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });

    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }
        cutlass::arch::fence_barrier_init();
    }
    (kNumTMAMulticast > 1) ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();

    constexpr uint32_t kNumTMARegisters = 40;
    constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 232;
    cudaGridDependencySynchronize();

    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups,
                                      kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs>(
        shape_m, shape_n, shape_k, grouped_layout);

    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    if (warp_idx >= kNumMathThreads / 32) {
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        if (warp_idx == kNumMathThreads / 32 + 2 and cute::elect_one_sync()) {
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
                const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                const uint32_t num_tma_multicast_b = (not kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;

                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    auto& full_barrier = *full_barriers[stage_idx];
                    const uint32_t k_idx = k_block_idx * BLOCK_K;
                    const uint32_t m_idx = scheduler.get_global_idx<false>(shape_m, BLOCK_M, m_block_idx);
                    const uint32_t n_idx = scheduler.get_global_idx<true>(shape_n, BLOCK_N, n_block_idx, m_block_idx);
                    tma::copy<BLOCK_K, BLOCK_N, kSwizzleAMode, __nv_fp8_e4m3, false>(
                        &tensor_map_b, &full_barrier, smem_a[stage_idx], k_idx, n_idx,
                        num_tma_multicast_b);
                    tma::copy<BLOCK_K, BLOCK_M, kSwizzleBMode, __nv_fp8_e4m3, false>(
                        &tensor_map_a, &full_barrier, smem_b[stage_idx], k_idx, m_idx,
                        num_tma_multicast_a);
                    tma::copy<BLOCK_M, 1, 0>(&tensor_map_sfa, &full_barrier,
                        smem_sfa[stage_idx], m_block_idx * BLOCK_M, k_block_idx,
                        num_tma_multicast_a);
                    full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SFA_SIZE_PER_STAGE);
                }
            }

            if constexpr (kNumTMAMulticast > 1) {
                for (uint32_t i = 0; i < kNumStages; advance_pipeline(i))
                    empty_barriers[stage_idx]->wait(phase ^ 1);
            }
        }
    } else {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        const auto warp_in_wg = __shfl_sync(0xffffffff, (threadIdx.x % 128) / 32, 0);
        auto a_desc = mma::sm90::make_smem_desc(smem_a[0] + math_wg_idx * WGMMA::M * BLOCK_K, 1);
        auto b_desc = mma::sm90::make_smem_desc(smem_b[0], 1);
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);

        auto empty_barrier_arrive = [&]() {
            if constexpr (kNumTMAMulticast == 1) {
                if (lane_idx == 0)
                    empty_barriers[stage_idx]->arrive();
            } else {
                auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                if (lane_idx < kNumTMAMulticast)
                    empty_barriers[stage_idx]->arrive(target_cta);
            }
        };

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            uint32_t group_idx = 0;
            if constexpr (kGemmType == GemmType::MGroupedContiguous)
                group_idx = cute::max(0, grouped_layout[m_block_idx * BLOCK_M]);

            if (threadIdx.x >= 32) {
                const uint32_t previous_group_offset =
                    scheduler.template get_global_idx<true, sched::IndexType::SF_K>(shape_n_sfb * shape_k_scales, 0, 0, m_block_idx);
                const uint32_t stride_n_sfb = kMajorSFB == cute::UMMA::Major::MN ? 1 : shape_k_scales;
                const uint32_t stride_k_sfb = kMajorSFB == cute::UMMA::Major::MN ? shape_n_sfb : 1;
                auto local_sfb = sfb + previous_group_offset + (n_block_idx * BLOCK_N / BLOCK_K) * stride_n_sfb;

                #pragma unroll
                for (uint32_t i = threadIdx.x - 32; i < shape_k_scales; i += kNumMathThreads - 32)
                    ptx::st_shared(smem_sfb + i, local_sfb[i * stride_k_sfb]);
            }
            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

            float final_accum[WGMMA::kNumAccum] = {0};
            if (scheduler.is_computation_valid(m_block_idx, 0)) {
                #pragma unroll 8
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    const auto a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                    const auto b_desc_base_lo = b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);
                    const float scale_b = ptx::ld_shared(smem_sfb + k_block_idx);

                    full_barriers[stage_idx]->wait(phase);

                    float accum[WGMMA::kNumAccum];
                    #pragma unroll
                    for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                        ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_arrive();
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                        a_desc.reg32_[0] = a_desc_base_lo + k * WGMMA::K / 16;
                        b_desc.reg32_[0] = b_desc_base_lo + k * WGMMA::K / 16;
                        WGMMA::wgmma(a_desc, b_desc, accum, k > 0);
                    }
                    ptx::warpgroup_commit_batch();
                    #pragma unroll
                    for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                        ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_wait<0>();

                    #pragma unroll
                    for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                        const uint32_t m0 = (lane_idx % 4) * 2 + i * 8;
                        const uint32_t m1 = m0 + 1;
                        const float scale_a_0 = ptx::ld_shared(smem_sfa[stage_idx] + m0);
                        const float scale_a_1 = ptx::ld_shared(smem_sfa[stage_idx] + m1);
                        final_accum[i * 4 + 0] += accum[i * 4 + 0] * scale_b * scale_a_0;
                        final_accum[i * 4 + 1] += accum[i * 4 + 1] * scale_b * scale_a_1;
                        final_accum[i * 4 + 2] += accum[i * 4 + 2] * scale_b * scale_a_0;
                        final_accum[i * 4 + 3] += accum[i * 4 + 3] * scale_b * scale_a_1;
                    }

                    empty_barrier_arrive();
                }
            } else {
                #pragma unroll
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);
                    empty_barrier_arrive();
                }
            }

            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 1);

            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                const uint32_t m0 = (lane_idx % 4) * 2 + i * 8;
                const uint32_t m1 = m0 + 1;
                const uint32_t global_m0 = m_block_idx * BLOCK_M + m0;
                const uint32_t global_m1 = m_block_idx * BLOCK_M + m1;
                bool valid_m0 = global_m0 < shape_m;
                bool valid_m1 = global_m1 < shape_m;
                if constexpr (kGemmType == GemmType::MGroupedContiguous) {
                    valid_m0 = valid_m0 and grouped_layout[global_m0] == static_cast<int>(group_idx);
                    valid_m1 = valid_m1 and grouped_layout[global_m1] == static_cast<int>(group_idx);
                }
                float d0 = valid_m0 ? final_accum[i * 4 + 0] : 0.0f;
                float d1 = valid_m1 ? final_accum[i * 4 + 1] : 0.0f;
                float d2 = valid_m0 ? final_accum[i * 4 + 2] : 0.0f;
                float d3 = valid_m1 ? final_accum[i * 4 + 3] : 0.0f;

                const uint32_t atom_offset = math_wg_idx;
                const uint32_t row = lane_idx % 8;
                const uint32_t col = warp_in_wg * 2 + lane_idx / 8;
                auto smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +
                    atom_offset * BLOCK_M * kSwizzleDMode +
                    i * 8 * kSwizzleDMode +
                    row * kSwizzleDMode +
                    (col ^ row) * kNumBankGroupBytes;

                ptx::SM90_U32x2_STSM_T<int>::copy(
                    math::cast_into_bf16_and_pack(d0, d1),
                    math::cast_into_bf16_and_pack(d2, d3),
                    smem_ptr);
            }

            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 1);

            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
                auto in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
                auto smem_ptr = smem_d + in_block_n_offset * BLOCK_M;
                auto n_idx = epilogue_type_t::template apply_index_n<TMA_D_BLOCK_N>(n_block_idx * BLOCK_N + in_block_n_offset);
                auto m_idx = scheduler.get_global_idx<false>(shape_m, BLOCK_M, m_block_idx);
                cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_ptr, n_idx, m_idx);
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
