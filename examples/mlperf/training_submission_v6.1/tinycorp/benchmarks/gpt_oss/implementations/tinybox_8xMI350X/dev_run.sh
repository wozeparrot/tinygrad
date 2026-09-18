#!/usr/bin/env bash

export PYTHONPATH="."
export PATH="/opt/rocm-7.1.1/bin:$PATH"
export ROCM_PATH="/opt/rocm-7.1.1"
export DEV=${DEV:-PCI+AMD}
# Qualified GPT-OSS compute/copy scheduling uses HCQ1; both runtime and scheduling remain overrideable.
export HCQ2=${HCQ2:-0}
export HCQ_REORDER=${HCQ_REORDER:-1}
export CHECK_OOB=0
export REWRITE_STACK_LIMIT=5000000 HCQDEV_WAIT_TIMEOUT_MS=240000
export DEVICE_IN_FUNCTION_BUG=1

export DEBUG=${DEBUG:-0}
export HK_FLASH_ATTENTION=${HK_FLASH_ATTENTION:-1}
# Exact GPTOSS SWA128 forward: four-wave workgroups retain O/LSE bits while reducing masked tile work.
export GPTOSS_FA_FWD_W4=${GPTOSS_FA_FWD_W4:-1}
# Keep full-causal KV pipeline boundaries unchanged while using four-wave workgroups; O/LSE remain bit-exact.
export GPTOSS_FA_FULL_FWD_W4=${GPTOSS_FA_FULL_FWD_W4:-1}
export GPTOSS_FA_FULL_FWD_LEAN=${GPTOSS_FA_FULL_FWD_LEAN:-1}
# Exact full-D64 backward retains LDS publication/reuse barriers but omits a redundant wave-local dQ rendezvous.
# Actual production full-output/poison gates and paired GPU timestamps pass; window/D128 callbacks are unchanged.
export GPTOSS_FA_BWD_RELAX_DQ_SYNC=${GPTOSS_FA_BWD_RELAX_DQ_SYNC:-1}
# Alternate dS scratch planes while retaining publication fences and the corrected epilogue VM drain.
# Full-output gates and JITBEAM=3 full-model profiling pass; only exact full-causal GPTOSS uses this path.
export GPTOSS_FA_BWD_DOUBLE_ATTN=${GPTOSS_FA_BWD_DOUBLE_ATTN:-1}
export ASM_GEMM=${ASM_GEMM:-1}
export GROUPED_MOE=${GROUPED_MOE:-1}
export ALL2ALL=${ALL2ALL:-1}
export LATE_ALLREDUCE=${LATE_ALLREDUCE:-0}
export ALLREDUCE_CAST=${ALLREDUCE_CAST:-1}
export USE_ATOMICS=${USE_ATOMICS:-1}
export MASTER_WEIGHTS=${MASTER_WEIGHTS:-1}
export MXFP8=${MXFP8:-1}
export ZERO_OPTIM=${ZERO_OPTIM:-1}
export OFFLOAD_OPTIM=${OFFLOAD_OPTIM:-0}

# Proven B200-style producer/epilogue fusions. Keep each overrideable for focused A/B runs.
export GROUPED_MOE=${GROUPED_MOE:-1}
export MOE_SKIP_EMPTY=${MOE_SKIP_EMPTY:-1}
export FC1_SKIP_EMPTY_MAIN_ZERO=${FC1_SKIP_EMPTY_MAIN_ZERO:-1}
export ZERO2=${ZERO2:-1}
# Converged vocabulary-row reduce-scatter and deferred LM-head gathering.
export GPTOSS_ZERO2_LMHEAD=${GPTOSS_ZERO2_LMHEAD:-1}
export GPTOSS_ZERO2_EMBEDDING=${GPTOSS_ZERO2_EMBEDDING:-1}
export DEFERRED_LMHEAD=${DEFERRED_LMHEAD:-1}
export GPTOSS_EMBEDDING_READY=${GPTOSS_EMBEDDING_READY:-1}
# Two distinct optimizer updates per capture, stopping at evaluation/checkpoint boundaries.
export GPTOSS_STEP_GROUP=${GPTOSS_STEP_GROUP:-2}
export JIT_BATCH_SIZE=${JIT_BATCH_SIZE:-128}
export FUSED_FC1=${FUSED_FC1:-1}
export DIRECT_FC1_H_OUT=${DIRECT_FC1_H_OUT:-0}
export FUSED_COMBINE=${FUSED_COMBINE:-1}
# Exact per-device GPT-OSS combine: one token-owned workgroup caches its four route rows/weights.
# Full-buffer exact/repeat; GPU7 JITBEAM3 ABBA64 176.00us -> 109.36us (-37.87%).
export GPTOSS_COMBINE_FWD_HIP=${GPTOSS_COMBINE_FWD_HIP:-1}
export COMBINE_DW_HIP=${COMBINE_DW_HIP:-1}
export COMBINE_DW_PIPE2_NTZ=${COMBINE_DW_PIPE2_NTZ:-1}
export FAST_E8M0=${FAST_E8M0:-1}
export FP8_DISPATCH=${FP8_DISPATCH:-1}
export FUSED_DISPATCH_QE8=${FUSED_DISPATCH_QE8:-1}
export SAVE_DISPATCH_E8=${SAVE_DISPATCH_E8:-1}
export FUSED_DISPATCH_COLW=${FUSED_DISPATCH_COLW:-1}
export GPTOSS_DISPATCH_THREADS=${GPTOSS_DISPATCH_THREADS:-128}
export DISPATCH_DUAL_M32=${DISPATCH_DUAL_M32:-1}
export FC1_DISPATCH_ROW_SI=${FC1_DISPATCH_ROW_SI:-1}
export FUSED_ROUTER_TOPK=${FUSED_ROUTER_TOPK:-1}
export FUSED_ROUTER_BIAS_GRAD=${FUSED_ROUTER_BIAS_GRAD:-1}
export DIRECT_ROUTER_TOPK_OUT=${DIRECT_ROUTER_TOPK_OUT:-1}
export FUSED_QKV_ROPE=${FUSED_QKV_ROPE:-1}
# One thread owns each exact GPTOSS (KV head, D64 pair), reusing cos/sin across its eight Q heads plus K.
export GPTOSS_QKV_ROPE_PACKED=${GPTOSS_QKV_ROPE_PACKED:-1}
# Fold QKV bias into packed RoPE, retaining the separate BF16 rounding boundary.
export GPTOSS_QKV_ROPE_BIAS=${GPTOSS_QKV_ROPE_BIAS:-1}
# Layer-local exact Q/K/V checkpoints bind saved outputs directly; RoPE/FA math and backward retention are unchanged.
export GPTOSS_QKV_ROPE_DIRECT_SAVE=${GPTOSS_QKV_ROPE_DIRECT_SAVE:-1}
# Group exact GPT-OSS Q heads per workgroup and mark their one-use native-FA gradient loads non-temporal.
export GPTOSS_QKV_ROPE_BWD_QGROUP=${GPTOSS_QKV_ROPE_BWD_QGROUP:-1}
export FUSED_MOE_BIAS=${FUSED_MOE_BIAS:-1}
export FUSED_SWIGLU=${FUSED_SWIGLU:-1}
export FUSED_FC1_COLW=${FUSED_FC1_COLW:-1}
export FC1_SPLIT_ROWS=${FC1_SPLIT_ROWS:-1}
export FC1_NATIVE_ROW=${FC1_NATIVE_ROW:-1}
export FC1_FAST_EXP=${FC1_FAST_EXP:-1}
export FC1_FAST_MATH=${FC1_FAST_MATH:-1}
export FC1_FAST_RCP=${FC1_FAST_RCP:-1}
export FC1_FAST_UNROUNDED=${FC1_FAST_UNROUNDED:-1}
export FC1_HALF_N_TAIL=${FC1_HALF_N_TAIL:-1}
export FC1_SKIP_H_TAIL_ZERO=${FC1_SKIP_H_TAIL_ZERO:-1}
export FC1_SCALAR_LDS_BASES=${FC1_SCALAR_LDS_BASES:-1}
export FC1_AMAX_PINGPONG=${FC1_AMAX_PINGPONG:-1}
export FC1_WARP_SCALE32=${FC1_WARP_SCALE32:-1}
export FC1_DUAL_COLW_LDS=${FC1_DUAL_COLW_LDS:-1}
# Reuse exact bias/SwiGLU/quantization results across fully padded lower 32-row wave bands.
export FC1_PAD_LOWER_REUSE=${FC1_PAD_LOWER_REUSE:-1}
export FC1_MED3_CLAMP=${FC1_MED3_CLAMP:-1}
export FC1_BINARY_EXPERT=${FC1_BINARY_EXPERT:-1}
export GPTOSS_FC1_DS_SWIZZLE=${GPTOSS_FC1_DS_SWIZZLE:-1}
export FC1_DIRECT_BIAS=${FC1_DIRECT_BIAS:-1}
# ROCm 7.2.1 schedules the exact FC1 forward slightly better without read-pointer alias hints. Broader masks also
# trigger a gfx950 compiler miscompile, so retain the bitmask only as an opt-in diagnostic override.
export FC1_RESTRICT_READS=${FC1_RESTRICT_READS:-0}
# Peel the exact 23rd contraction so the FC1 steady-state double-buffer loop has no terminal predicates.
export FC1_PEEL_FINAL=${FC1_PEEL_FINAL:-1}
# Hoist the uniform complete-vs-half N-tile choice outside FC1's K loop. GPU1 JITBEAM3 five-output poison/repeat
# ABBA64: 2103.18us -> 2017.96us (-4.05%); the library callback keeps this exact specialization opt-in.
export GPTOSS_FC1_INTERIOR_FAST=${GPTOSS_FC1_INTERIOR_FAST:-1}
# Exact-count partial expert tiles can leave complete upper 32-row wave bands inactive. Keep full upper halves on
# the original contraction and omit only those inactive bands' A LDS reads and cA/cB MFMAs.
export GPTOSS_FC1_PAD_UPPER_SKIP=${GPTOSS_FC1_PAD_UPPER_SKIP:-1}
# Pass exact-shape byte offsets directly to FC1's raw global-to-LDS loader; randomized ABBA64 saves ~0.5% and
# reduces the production kernel from 79 to 75 SGPR without changing VGPR/LDS or any of its five outputs.
export GPTOSS_FC1_PRECOMPUTED_SOFF=${GPTOSS_FC1_PRECOMPUTED_SOFF:-1}
# Cooperatively stage each FC1 bias tile in dead scale LDS before the register epilogue. The existing final
# contraction barrier publishes it; GPU5 randomized full-output ABBA128: 2015.38us -> 2010.78us (-0.23%).
export GPTOSS_FC1_BIAS_LDS=${GPTOSS_FC1_BIAS_LDS:-1}
# Pair adjacent 32-token columnwise blocks and store their e8m0 scales together. Full five-output ABBA96/64
# is byte-exact and saves 0.52-0.61% across balanced, skew, and adversarial-boundary routing on GPU6.
export GPTOSS_FC1_COLW_ADJ_SCALE2=${GPTOSS_FC1_COLW_ADJ_SCALE2:-1}
# Stage each row's four e8m0 output bytes in unused transpose LDS and commit them with one aligned store. Full
# five-output ABBA96/64 is exact and saves 0.71-0.83% across balanced, skew, and adversarial-boundary routing.
export GPTOSS_FC1_ROW_SCALE_LDS4=${GPTOSS_FC1_ROW_SCALE_LDS4:-1}
# Tune only the `_ifast1` FC1 launch, without changing GROUPED_WGM for the other grouped GEMMs. With CWS2+RS4,
# WGM3/x8/c64 is exact and beats WGM6 on two GPUs: 0.17-0.42% balanced, 0.88-1.02% skew, 0.43-0.54% boundary.
export GPTOSS_FC1_WGM=${GPTOSS_FC1_WGM:-3}
export GPTOSS_FC1_MAP_X8_C64=${GPTOSS_FC1_MAP_X8_C64:-1}
export GPTOSS_FC1_MAP_X4_C48=${GPTOSS_FC1_MAP_X4_C48:-0}
export GPTOSS_FC1_MAP_X2_C96=${GPTOSS_FC1_MAP_X2_C96:-0}
# The exact GPT-OSS FC1 forward shape schedules slightly better with ROCm 7.2.1; the callback falls back to the
# configured ROCm toolchain if that side installation is unavailable.
export GPTOSS_FC1_FWD_ROCM72=${GPTOSS_FC1_FWD_ROCM72:-1}
export GPTOSS_DGRAD_TAIL=${GPTOSS_DGRAD_TAIL:-1}
# ROCm 7.2.1 reduces the DGRAD kernel's register pressure and is ~9-10% faster on MI350X.
export GPTOSS_DGRAD_ROCM72=${GPTOSS_DGRAD_ROCM72:-1}
export GPTOSS_DGRAD_WGM=${GPTOSS_DGRAD_WGM:-4}
export DGRAD_STAGE_OUT_SCALE=${DGRAD_STAGE_OUT_SCALE:-2}
# Select the exact FC1 dgrad double-buffer planes from literal LDS bases. GPU5 JITBEAM3 randomized ABBA96:
# 1699.16us -> 1674.66us (-1.44%), with full poison/repeat and all output-scale byte encodings bit-exact.
export GPTOSS_DGRAD_SCALAR_LDS=${GPTOSS_DGRAD_SCALAR_LDS:-1}
export FUSED_DGRAD_FP8=${FUSED_DGRAD_FP8:-1}
export FUSED_DGRAD_DSWIGLU=${FUSED_DGRAD_DSWIGLU:-1}
export FUSED_DGRAD_DSWIGLU_SPLIT=${FUSED_DGRAD_DSWIGLU_SPLIT:-1}
export FUSED_DH_DUAL_PRODUCER=${FUSED_DH_DUAL_PRODUCER:-1}
export GPTOSS_DSWIGLU_DUAL_WRITE_DH=${GPTOSS_DSWIGLU_DUAL_WRITE_DH:-0}
export GPTOSS_DSWIGLU_RCP_NR=${GPTOSS_DSWIGLU_RCP_NR:-1}
export GPTOSS_DSWIGLU_SKIP_PAD_TAIL=${GPTOSS_DSWIGLU_SKIP_PAD_TAIL:-1}
export GPTOSS_DSWIGLU_SKIP_EMPTY_TAIL=${GPTOSS_DSWIGLU_SKIP_EMPTY_TAIL:-1}
# Stop the dual dSwiGLU producer at each expert's ceil(count/128) boundary. Its row/column companions are consumed
# under the same exact-count contract, while skipped bias partials are explicitly zeroed. NaN-poison ABBA64 is exact;
# balanced/skew/boundary routing improves 3.1-3.9% with unchanged VGPR/LDS and no spill.
export GPTOSS_DSWIGLU_EXPERT_COUNTS=${GPTOSS_DSWIGLU_EXPERT_COUNTS:-1}
# Explicit identity preserves the observed legacy hidden-zero grid order without reading unbound hidden grid args.
export GPTOSS_DSWIGLU_XCD_MAP=${GPTOSS_DSWIGLU_XCD_MAP:-0}
export GPTOSS_DSWIGLU_XCD_CHUNK=${GPTOSS_DSWIGLU_XCD_CHUNK:-112}
export GPTOSS_DSWIGLU_LDS_STRIDE=${GPTOSS_DSWIGLU_LDS_STRIDE:-264}
export GPTOSS_DSWIGLU_MAX_MEMORY_CLAUSE=${GPTOSS_DSWIGLU_MAX_MEMORY_CLAUSE:-1}
# Exact bounded-domain exp2 and one empty-tail row scale per lane; production no-dH path only.
# Direct legacy comparison passes all five outputs and 512 paired rounds on balanced/skew/boundary routing.
export GPTOSS_DSWIGLU_NATIVE_EXP2=${GPTOSS_DSWIGLU_NATIVE_EXP2:-1}
export GPTOSS_DSWIGLU_TAIL_ROW_E8=${GPTOSS_DSWIGLU_TAIL_ROW_E8:-1}
export GPTOSS_DOWN_REQUANT=${GPTOSS_DOWN_REQUANT:-1}
export GPTOSS_DOWN_REQUANT_M32=${GPTOSS_DOWN_REQUANT_M32:-1}
export GPTOSS_DOWN_REQUANT_THREADS=${GPTOSS_DOWN_REQUANT_THREADS:-128}
export GPTOSS_DOWN_REQUANT_PACKED_INPUT=${GPTOSS_DOWN_REQUANT_PACKED_INPUT:-1}
# Preserve the down weight's BF16 dequantization boundary with gfx950's direct scaled FP8-to-BF16 conversion.
export GPTOSS_DOWN_REQUANT_DIRECT_BF16=${GPTOSS_DOWN_REQUANT_DIRECT_BF16:-1}
# Exact packed-scale extraction and exponent construction for the down-weight transpose/requant producer.
export GPTOSS_DOWN_REQUANT_BFE_SCALE=${GPTOSS_DOWN_REQUANT_BFE_SCALE:-1}
# One wave owns paired input columns, using both lanes of scaled FP8-to-BF16 conversion.
export GPTOSS_DOWN_REQUANT_PAIR=${GPTOSS_DOWN_REQUANT_PAIR:-1}
export GPTOSS_DOWN_FWD_SCALAR_LDS=${GPTOSS_DOWN_FWD_SCALAR_LDS:-1}
export GPTOSS_DOWN_FWD_BINARY_EXPERT=${GPTOSS_DOWN_FWD_BINARY_EXPERT:-1}
# Exact down-forward WGM8 launch count activates its qualified chiplet mapping without hidden grid arguments.
export GPTOSS_DOWN_FWD_TRUE_GRID=${GPTOSS_DOWN_FWD_TRUE_GRID:-1}
export GPTOSS_DOWN_DGRAD_SCALAR_LDS=${GPTOSS_DOWN_DGRAD_SCALAR_LDS:-1}
# Twelve-row workgroup grouping improves XCD locality for only the exact non-bias down dInput. GPU6 randomized
# JITBEAM3 ABBA96: 958.30us -> 949.02us (-0.97%), with full physical-output poison/repeat/adversarial exactness.
export GPTOSS_DOWN_DGRAD_WGM=${GPTOSS_DOWN_DGRAD_WGM:-12}
# Exact down dInput B fragments select only two scale bytes; preserve all physical output and reduction bits.
export GPTOSS_DOWN_DGRAD_PACK_B32=${GPTOSS_DOWN_DGRAD_PACK_B32:-1}
export DSWIGLU_REAL_ONLY_TAIL=${DSWIGLU_REAL_ONLY_TAIL:-1}
export FUSED_DH_COLW_SEPARATE=${FUSED_DH_COLW_SEPARATE:-1}
export FUSED_DH_BIAS_PARTIAL=${FUSED_DH_BIAS_PARTIAL:-1}
export FUSED_DH_BIAS_REDUCE_HIP=${FUSED_DH_BIAS_REDUCE_HIP:-1}
# Down-bias partials keep the exact even/odd FP32 accumulation tree while preloading 32 independent rows.
export GPTOSS_DOWN_BIAS_REDUCE_UNROLL=${GPTOSS_DOWN_BIAS_REDUCE_UNROLL:-32}
export FUSED_DZ_COLW_SEPARATE=${FUSED_DZ_COLW_SEPARATE:-1}
export FUSED_TQ_WGRAD_DBIAS=${FUSED_TQ_WGRAD_DBIAS:-1}
export GPTOSS_MOE_TQ_THREADS=${GPTOSS_MOE_TQ_THREADS:-64}
# The exact down-gradient transpose has no padded columns; remove its redundant bounds-select from all 32 BF16 loads.
export GPTOSS_DOWN_TQ_DIRECT_LOAD=${GPTOSS_DOWN_TQ_DIRECT_LOAD:-1}
# The exact QKV-gradient dual-layout transpose likewise spans complete 5120-column tiles.
export GPTOSS_QKV_TQ_DUAL=${GPTOSS_QKV_TQ_DUAL:-1}
export GPTOSS_QKV_TQ_DUAL_DIRECT_LOAD=${GPTOSS_QKV_TQ_DUAL_DIRECT_LOAD:-1}
export GPTOSS_QKV_TQ_DUAL_THREADS=${GPTOSS_QKV_TQ_DUAL_THREADS:-128}
export GPTOSS_QKV_TQ_ROW_SI_BYTE_STORE=${GPTOSS_QKV_TQ_ROW_SI_BYTE_STORE:-1}
export GPTOSS_QKV_TQ_ROW_E8_SI_ONLY=${GPTOSS_QKV_TQ_ROW_E8_SI_ONLY:-1}
export GPTOSS_QKV_TQ_COL_E8_SI_ONLY=${GPTOSS_QKV_TQ_COL_E8_SI_ONLY:-1}
# QKV dense backward: reuse packed forward scales in the exact activation/weight transpose-requant producers.
export GPTOSS_QKV_REQUANT_PACKED=${GPTOSS_QKV_REQUANT_PACKED:-1}
export GPTOSS_QKV_REQUANT_M32=${GPTOSS_QKV_REQUANT_M32:-1}
export GPTOSS_QKV_REQUANT_SI_ONLY=${GPTOSS_QKV_REQUANT_SI_ONLY:-1}
export FUSED_DZ_BIAS_PARTIAL=${FUSED_DZ_BIAS_PARTIAL:-1}
# Emit both dZ quantization directions and bias partials directly in their consumer layouts; no BF16 round-trip.
# Full JITBEAM3 profile verifies the old transpose, column-FP8 copy and output initializers are absent.
export GPTOSS_COMBINE_DZ_QUANT=${GPTOSS_COMBINE_DZ_QUANT:-1}
# The fused dZ producer writes column scales in the exact uint32 layout consumed by down wgrad.
export GPTOSS_COMBINE_COL_SI=${GPTOSS_COMBINE_COL_SI:-1}
# Row-local 32x256 quantization retains exact bias sums and both physical MX layouts.
export GPTOSS_COMBINE_DZ_TILE256=${GPTOSS_COMBINE_DZ_TILE256:-1}
export FUSED_DOWN_BIAS=${FUSED_DOWN_BIAS:-1}
export FUSED_CE=${FUSED_CE:-1}
export FUSED_LINEAR_CE=${FUSED_LINEAR_CE:-1}
# Keep the converged BF16 forward. The dual-MX backward is byte-exact against the established materialized
# dLogits path, but emits both quantized layouts and packed scales in one launch.
export LINEAR_CE_MX_FWD=${LINEAR_CE_MX_FWD:-0}
export LINEAR_CE_BF16_NO_PAD=${LINEAR_CE_BF16_NO_PAD:-1}
export GPTOSS_LMHEAD_CE_WGM=${GPTOSS_LMHEAD_CE_WGM:-16}
export GPTOSS_LMHEAD_CE_ROCM72=${GPTOSS_LMHEAD_CE_ROCM72:-1}
# Rotate consecutive CE staging rows across LDS banks instead of using a 512-byte stride.
export GPTOSS_LMHEAD_CE_LDS_PAD=${GPTOSS_LMHEAD_CE_LDS_PAD:-2}
# Four-BF16 saved-logit stores preserve all GEMM/CE math; exact full-output GPU gates are 2.5-2.6% faster.
export GPTOSS_LMHEAD_CE_STORE4=${GPTOSS_LMHEAD_CE_STORE4:-1}
export MX_CE_EXACT_EXP=${MX_CE_EXACT_EXP:-1}
export LINEAR_CE_MX_BWD=${LINEAR_CE_MX_BWD:-1}
export LINEAR_CE_DUAL_MX_BWD=${LINEAR_CE_DUAL_MX_BWD:-1}
export GPTOSS_CE_DUAL_VEC_ROW_STORE=${GPTOSS_CE_DUAL_VEC_ROW_STORE:-1}
export GPTOSS_CE_DUAL_LDS_BLOCK_PAD8=${GPTOSS_CE_DUAL_LDS_BLOCK_PAD8:-1}
export LINEAR_CE_PADDED_TQ=${LINEAR_CE_PADDED_TQ:-1}
export FP8_LMHEAD=${FP8_LMHEAD:-1}
export FUSED_RMSNORM_MUL=${FUSED_RMSNORM_MUL:-1}
export FUSED_RMSNORM_MX=${FUSED_RMSNORM_MX:-1}
# Eight lanes cooperatively quantize each exact GPT-OSS 32-value block; full q/e8/rrms bits match the serial epilogue.
# GPU1/7 JITBEAM3 ABBA64: 242.40us -> 138.89/137.99us (-42.70/-43.08%).
export GPTOSS_RMSNORM_MX_EP8=${GPTOSS_RMSNORM_MX_EP8:-1}
# ROCm 7.2 lowers the unchanged exact backward kernel from 170 to 90 VGPR; all grad-x/weight-partial bits match 7.1.
export GPTOSS_RMSNORM_MX_BWD_ROCM72=${GPTOSS_RMSNORM_MX_BWD_ROCM72:-1}
export FUSED_RMSNORM_MUL_BWD=${FUSED_RMSNORM_MUL_BWD:-1}
# Consume the router's physical flat gradient without a logical-3D materialization.
export GPTOSS_RMSNORM_BWD_FLAT_GRAD=${GPTOSS_RMSNORM_BWD_FLAT_GRAD:-1}
# Two exact GPTOSS rows share one workgroup reduction/barrier pair; all backward outputs remain bit-identical.
export GPTOSS_RMSNORM_MUL_BWD_PAIR=${GPTOSS_RMSNORM_MUL_BWD_PAIR:-1}
# Fuse the exact 16384x4096 attention-output MXFP8 amax and quantizing cast; retain the faster separate scale pack.
export GPTOSS_ATTN_FUSED_QE8=${GPTOSS_ATTN_FUSED_QE8:-1}
export COMBINE_GATHER=${COMBINE_GATHER:-1}
# The exact GPTOSS gather-backward fully overwrites its three outputs. Replace
# their 1.16 ms/layer zero-fill with three one-element graph-partition guards,
# and stream the 453 MiB BF16 output without allocating it in L2.
export COMBINE_DZ_WRITEONLY=${COMBINE_DZ_WRITEONLY:-1}
export COMBINE_DZ_NT_BF16=${COMBINE_DZ_NT_BF16:-1}
export DISPATCH_GATHER=${DISPATCH_GATHER:-1}
# One wave owns each inverse-dispatch token and sums its four expert rows without generalized modulo addressing.
export GGATHER_SUM_HIP=${GGATHER_SUM_HIP:-1}
export GGATHER_SUM_THREADS=${GGATHER_SUM_THREADS:-64}
export BEAM_GLUE=${BEAM_GLUE:-1}
export NATIVE_MXFP8_CVT=${NATIVE_MXFP8_CVT:-1}
export FUSED_ADAM_MXFP8=${FUSED_ADAM_MXFP8:-1}
# Schedule the four independent per-lane FC1 Adam updates in phases so the compiler overlaps their sqrt latency.
export GPTOSS_ADAM_FC1_PIPELINE=${GPTOSS_ADAM_FC1_PIPELINE:-1}
# Skip Adam math for FC1's invariant zero-padded physical rows while still emitting its full q/e8/SI ABI.
export GPTOSS_ADAM_FC1_ROW_SKIP=${GPTOSS_ADAM_FC1_ROW_SKIP:-1}
# Keep the eleven complete FC1 column blocks in a compact hot launch and handle its 64-value/padding tail separately.
export GPTOSS_ADAM_FC1_SPLIT=${GPTOSS_ADAM_FC1_SPLIT:-1}
# Consume the same-shard FC1 gradient and reproduce the materialized BF16 clip boundary inside fused Adam.
export GPTOSS_ADAM_RAW_CLIP=${GPTOSS_ADAM_RAW_CLIP:-1}
# The down-weight kernel needs a distinct v-moment expression to preserve its materialized BF16 clip boundary.
export GPTOSS_ADAM_DOWN_RAW_CLIP=${GPTOSS_ADAM_DOWN_RAW_CLIP:-1}
# One wave updates four independent down-weight elements while preserving the established FP32 v/update associations.
export GPTOSS_ADAM_DOWN_PIPELINE=${GPTOSS_ADAM_DOWN_PIPELINE:-1}
# Fuse m, v, and FP32-master Adam updates for the two GPT-OSS BF16 vocabulary matrices.
export GPTOSS_ADAM_BF16_VOCAB=${GPTOSS_ADAM_BF16_VOCAB:-1}
# Consume their post-DP-reduction gradient and reproduce the materialized BF16 clip boundary inside fused Adam.
export GPTOSS_ADAM_BF16_RAW_CLIP=${GPTOSS_ADAM_BF16_RAW_CLIP:-1}
# Skip Adam arithmetic and input-state reads for the zero-padded rows/columns of each physical 3072x3072 down
# expert while retaining the physical q/e8 ABI and explicitly writing its zero tail.
export GPTOSS_ADAM_DOWN_LOGICAL=${GPTOSS_ADAM_DOWN_LOGICAL:-1}
export PREPACK_FC1_WSI=${PREPACK_FC1_WSI:-1}
# The FC1 dgrad consumes only the logical 2880x5760 rectangle of its transposed weight. Compact the contraction
# grid while retaining the faster physical-N tile order; padded N lanes return before dead requantization.
export GPTOSS_REQUANT_TRIM_DGRAD=${GPTOSS_REQUANT_TRIM_DGRAD:-1}
# Preserve the FC1 weight's BF16 dequantization boundary with gfx950's direct scaled FP8-to-BF16 conversion.
export GPTOSS_FC1_REQUANT_DIRECT_BF16=${GPTOSS_FC1_REQUANT_DIRECT_BF16:-1}
# Extract each packed FC1 source scale byte with gfx950's single-instruction bitfield extract.
export GPTOSS_FC1_REQUANT_BFE_SCALE=${GPTOSS_FC1_REQUANT_BFE_SCALE:-1}
# Select custom forward/backward embedding together; 0 uses stock nn.Embedding.
export GPTOSS_EMBEDDING=${GPTOSS_EMBEDDING:-1}
export FAST_GRAD_NORM=${FAST_GRAD_NORM:-1}
export GPTOSS_BATCHED_GRAD_NORM=${GPTOSS_BATCHED_GRAD_NORM:-1}
export GPTOSS_DIRECT_FC1_GATHER=${GPTOSS_DIRECT_FC1_GATHER:-0}
export GPTOSS_DIRECT_VOCAB_GATHER=${GPTOSS_DIRECT_VOCAB_GATHER:-0}
export GPTOSS_COMPACT_Q_GATHER=${GPTOSS_COMPACT_Q_GATHER:-1}
export GPTOSS_FC1_READY_TAIL=${GPTOSS_FC1_READY_TAIL:-2}
export GPTOSS_VOCAB_READY=${GPTOSS_VOCAB_READY:-0}
export ROUTER_MFMA=${ROUTER_MFMA:-1}
# Norm outputs are dense row-major views; preserve their buffers through the router input.
export GPTOSS_ROUTER_INPUT_VIEW=${GPTOSS_ROUTER_INPUT_VIEW:-1}
# Double-buffer the exact GPT-OSS router's 64-wide contraction tiles without changing FP32 accumulation order.
export GPTOSS_ROUTER_MFMA_DBUF=${GPTOSS_ROUTER_MFMA_DBUF:-1}
export ROUTER_FP32_WGRAD=${ROUTER_FP32_WGRAD:-1}
export ROUTER_FP32_WGRAD_DUAL_EXPERT=${ROUTER_FP32_WGRAD_DUAL_EXPERT:-1}
# Expose eight input groups at a time without changing the router weight-gradient FP32 accumulation order.
export GPTOSS_ROUTER_WGRAD_PREFETCH=${GPTOSS_ROUTER_WGRAD_PREFETCH:-1}
# Materialize the exact GPT-OSS three-way BF16 residual with a coalesced vector kernel. This replaces the generic
# last-block residual program's strided 0.53 ms pass with an exact ~0.08 ms pass and keeps the same logical shape.
export GPTOSS_RESIDUAL_HIP=${GPTOSS_RESIDUAL_HIP:-1}
# Retain the exact FFN norm input (90 MiB/layer/GPU) to avoid WO forward recomputation during backward.
export GPTOSS_SAVE_FFN_INPUT=${GPTOSS_SAVE_FFN_INPUT:-1}
# Exact local attention-output bias sum: retain the 16x1024 FP32 addition tree and BF16 aggregation boundary.
export GPTOSS_WO_BIAS_SUM=${GPTOSS_WO_BIAS_SUM:-1}
# Retain the final RMSNorm's exact per-row reduction order while exposing four times as many independent waves.
export GPTOSS_FINAL_RMSNORM=${GPTOSS_FINAL_RMSNORM:-1}
# Exact GPT-OSS router dgrad + dispatch-residual join. Eight tokens per workgroup reuse each 32x320 gate tile twice.
export FUSED_ROUTER_DGRAD=${FUSED_ROUTER_DGRAD:-1}
export DBUF=${DBUF:-1}
export WGRAD_DBUF=${WGRAD_DBUF:-1}
export GPTOSS_WGRAD_TAIL=${GPTOSS_WGRAD_TAIL:-1}
# Stop each exact FC1 expert contraction at its saved routed-token count instead of the padded 128-row boundary.
export GPTOSS_WGRAD_EXPERT_COUNTS=${GPTOSS_WGRAD_EXPERT_COUNTS:-1}
# ROCm 7.2 lowers register pressure for the exact GPTOSS FC1 wgrad and is ~2% faster in JITBEAM=3 profiles.
export GPTOSS_WGRAD_ROCM72=${GPTOSS_WGRAD_ROCM72:-1}
# Hide the final four B1 LDS reads under the first interior-tile MFMA. GPU5 randomized JITBEAM3 ABBA64:
# 1743.66us -> 1739.71us (-0.23%), with full-output random/special-value poison/repeat exactness.
export GPTOSS_WGRAD_B1_OVERLAP=${GPTOSS_WGRAD_B1_OVERLAP:-1}
# Issue the small interior scale tiles before A/B traffic. GPU5 JITBEAM3 ABBA64: -0.33% balanced, -0.30% skew,
# and -0.09% dist5 on top of B1 overlap, with unchanged 234 VGPR / 135168-byte LDS / zero scratch.
export GPTOSS_WGRAD_SCALE_FIRST=${GPTOSS_WGRAD_SCALE_FIRST:-1}
# Exact FC1-wgrad locality mapping only. GPU6 randomized ABBA96: 1778.99us -> 1759.08us (-1.12%), with full
# physical-output poison/repeat/adversarial exactness and unchanged VGPR/LDS/scratch.
export GPTOSS_WGRAD_WGM=${GPTOSS_WGRAD_WGM:-6}
export GPTOSS_WGRAD_XCD_CHUNK=${GPTOSS_WGRAD_XCD_CHUNK:-128}
export GPTOSS_DOWN_WGRAD_XCD_MAP=${GPTOSS_DOWN_WGRAD_XCD_MAP:-1}
# Exact down-wgrad locality mapping only; matrix math and edge64 stores are unchanged. GPU6 randomized JITBEAM3
# ABBA96: 1017.76us -> 984.09us (-3.31%), with full physical-output poison/repeat/adversarial exactness.
export GPTOSS_DOWN_WGRAD_WGM=${GPTOSS_DOWN_WGRAD_WGM:-3}
export GPTOSS_DOWN_WGRAD_XCD_CHUNK=${GPTOSS_DOWN_WGRAD_XCD_CHUNK:-64}
# Reuse FC1's saved exact expert counts in down wgrad, omitting wholly padded 128-row contraction tiles. Full-gradient
# NaN-poison/repeat checks are exact; same-allocation ABBA64 improves 2.3-3.0% across balanced/skew/boundary routing.
export GPTOSS_DOWN_WGRAD_EXPERT_COUNTS=${GPTOSS_DOWN_WGRAD_EXPERT_COUNTS:-1}
export DGRAD_DSWIGLU_DBUF=${DGRAD_DSWIGLU_DBUF:-1}
export DGRAD_TILE_N128=${DGRAD_TILE_N128:-0}
export DGRAD_PACKED_IO=${DGRAD_PACKED_IO:-1}
export FUSED_GRAD_QUANTIZE=${FUSED_GRAD_QUANTIZE:-1}
export DENSE_DIRECT_REQUANT=${DENSE_DIRECT_REQUANT:-1}
export FUSED_DENSE_BIAS_GRAD=${FUSED_DENSE_BIAS_GRAD:-1}
export FUSED_DENSE_DUAL_QUANT=${FUSED_DENSE_DUAL_QUANT:-1}
export MX_TILE_M128=${MX_TILE_M128:-1}
export MX_SINGLE_BUFFER=${MX_SINGLE_BUFFER:-$MX_TILE_M128}
# GPTOSS's six dense attention shapes consume only two bytes from each 32-row operand scale pack.
export GPTOSS_DENSE_PACK32=${GPTOSS_DENSE_PACK32:-1}
export GPTOSS_DENSE_TRUE_GRID=${GPTOSS_DENSE_TRUE_GRID:-1}
# Exact QKV dInput: preserve the BF16 GEMM boundary and apply the activation's MX scale in the FP32 store epilogue.
export GPTOSS_QKV_DGRAD_SCALE_EPILOGUE=${GPTOSS_QKV_DGRAD_SCALE_EPILOGUE:-1}
# Exact GPT-OSS LM-head dWeight: WGM16 plus a dedicated 64-column physical-output tail at local K=16384.
export GPTOSS_LMHEAD_WGRAD=${GPTOSS_LMHEAD_WGRAD:-1}
export GPTOSS_LMHEAD_WGRAD_SPLIT=${GPTOSS_LMHEAD_WGRAD_SPLIT:-1}
export GPTOSS_LMHEAD_WGRAD_WGM=${GPTOSS_LMHEAD_WGRAD_WGM:-16}
# Exact local GPT-OSS LM-head dHidden uses a WGM4 source built by ROCm 7.2.1. Two fixed-binary GPU6 ABBA192 runs:
# 8.565 -> 8.434 ms and 8.551 -> 8.428 ms (-1.53%/-1.45%), with full-output poison/repeat bit exactness.
export GPTOSS_LMHEAD_DH_WGM4=${GPTOSS_LMHEAD_DH_WGM4:-1}
export GPTOSS_LMHEAD_DH_ROCM72=${GPTOSS_LMHEAD_DH_ROCM72:-1}
export GPTOSS_LMHEAD_PACK32=${GPTOSS_LMHEAD_PACK32:-1}
# Fuse the two serial 501-tile cross-entropy finalize reductions for the exact local GPT-OSS shape.
export GPTOSS_LMHEAD_CE_FINALIZE_BF16=${GPTOSS_LMHEAD_CE_FINALIZE_BF16:-1}
export REG_EPILOGUE=${REG_EPILOGUE:-1}

export DEFAULT_FLOAT="bfloat16" OPTIM_DTYPE="bfloat16"
export DP=${DP:-8} BS=${BS:-16} EVAL_BS=${EVAL_BS:-8} GRADIENT_ACC_STEPS=${GRADIENT_ACC_STEPS:-1}
export GBS=$((BS * GRADIENT_ACC_STEPS))

export MODEL="gptoss"
export BASEDIR="/raid/datasets/c4-8b/"
export EVAL_TARGET=3.34 EVAL_FREQ=12288
export END_LR="4e-5" WARMUP_STEPS=${WARMUP_STEPS:-128} MAX_STEPS=${MAX_STEPS:-1200000}
export SAMPLES=${SAMPLES:-$((MAX_STEPS * GBS))}
export SEQLEN=${SEQLEN:-8192}

export SEED=${SEED:-$RANDOM}
export DATA_SEED=${DATA_SEED:-5760}

export JITBEAM=${JITBEAM:-3}
export BEAM_UOPS_MAX=6000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5 BEAM_PADTO=0

python3 examples/mlperf/model_train.py
