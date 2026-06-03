// Fused permute + NvFP4-per-token-quant for the FP4 MoE-LoRA gate_up path.
//
// Background (decode bs64, EP8): the plain path runs `permuteKernel` (gather bf16 hidden into the
// padded [max_padded, hidden] permuted buffer) then `nvfp4QuantAndPerTokenScaleKernel` over ALL
// max_padded rows. At decode only num_tokens*top_k of those rows are real (the rest are padding),
// so both kernels waste ~6x of their work on padding, and the bf16 permuted buffer is a full
// HBM round-trip (written by permute, read back by quant).
//
// This kernel fuses the two: it reads the UN-permuted hidden, NvFP4-quantizes each (token,expert)
// pair's row, and scatter-writes fp4 + swizzled block-sf + per-token-sf directly to that pair's
// permuted position. It iterates only the num_tokens*top_k real pairs (skipping pad), and never
// materializes the bf16 permuted buffer.
//
// It is a verbatim copy of `nvfp4QuantAndPerTokenScaleKernel` (quantization.cuh) — same amax,
// same per-token-scale recipe, same `cvt_warp_fp16_to_fp4`, same swizzled-sf offset — with the
// single `rowIdx` split into a READ row (the unpermuted source token) and a WRITE row (the
// permuted destination). For the valid rows the output is therefore BIT-IDENTICAL to the plain
// permute->quant chain (the plain chain's quant reads permuted_hidden[writeRow], which the permute
// filled from hidden[readRow]; we read hidden[readRow] directly). v1 targets the production math
// branch (no FP4-fast-math-disable, no 4-over-6); callers using those env flags must keep the
// plain path.
//
// Two variants (kept side by side, selectable, for cross-scenario perf comparison):
//   - no-dedup: grid over the num_tokens*top_k pairs; each block re-reads+re-quantizes its source
//     token (so each token is quantized top_k times) and writes 1 destination. More blocks ->
//     better occupancy at tiny decode sizes.
//   - dedup:    grid over num_tokens; each block reads+quantizes its token once and scatter-writes
//     to all of that token's (valid) permuted destinations. Fewer blocks, no redundant quant.
#pragma once

#include "nv_internal/tensorrt_llm/kernels/quantization.cuh"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace sgl_fused_permute_quant {

namespace tk = tensorrt_llm::kernels;

// Compute the per-token encode scale exactly like the production (`else`) branch of
// nvfp4QuantAndPerTokenScaleKernel: perTokenScale = globalAmax * globalScaleInv, written out, then
// read back (round-trip through gmem matches the original's determinism), encode = 1/perTokenScale.
template <typename T, uint32_t BLOCK_SIZE, tensorrt_llm::QuantizationSFLayout SF_LAYOUT>
__device__ __forceinline__ void fused_quant_one_row(
    uint32_t n,
    T const* input,
    int readRow,
    int writeRow,
    float globalScaleInv,
    uint8_t* weightOutput,
    uint8_t* scaleOutput,
    float* perTokenScaleOutput) {
  constexpr int SF_VEC_SIZE = 16;
  using InType = tk::PackedVec<T, SF_VEC_SIZE>;
  using PackedFp4Type = std::conditional_t<SF_VEC_SIZE == 16, uint64_t, uint32_t>;
  uint32_t const num_vecs_per_row = (n + SF_VEC_SIZE - 1) / SF_VEC_SIZE;

  // ---- pass 1: per-row amax over the (unpermuted) source row ----
  InType vec_in;
  float localAmax = 0.f;
  for (uint32_t vecIdx = threadIdx.x; vecIdx < num_vecs_per_row; vecIdx += BLOCK_SIZE) {
    int64_t const vecOffset = static_cast<int64_t>(readRow) * num_vecs_per_row + vecIdx;
    tk::loadPackedVec(vec_in, reinterpret_cast<InType const*>(input) + vecOffset);
    std::remove_reference_t<decltype(vec_in.elts[0])> localAmax_(0.f, 0.f);
#pragma unroll
    for (int i = 0; i < SF_VEC_SIZE / 2; ++i) {
      localAmax_ = __hmax2(localAmax_, __habs2(vec_in.elts[i]));
    }
    localAmax = fmaxf(localAmax, static_cast<float>(__hmax(localAmax_.x, localAmax_.y)));
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  float const globalAmax = BlockReduce(tempStorage).Reduce(localAmax, cuda::maximum<>{});

  // ---- per-token scale (production branch) ----
  __shared__ float sPerTokenScale;
  if (threadIdx.x == 0) {
    sPerTokenScale = globalAmax * globalScaleInv;
    perTokenScaleOutput[writeRow] = sPerTokenScale;
  }
  __syncthreads();
  float const perTokenScale = sPerTokenScale;
  float const globalEncodeScale = reciprocal_approximate_ftz(perTokenScale);

  // ---- pass 2: quantize + scatter-write to the permuted destination ----
  for (uint32_t vecIdx = threadIdx.x; vecIdx < num_vecs_per_row; vecIdx += BLOCK_SIZE) {
    int64_t const readOffset = static_cast<int64_t>(readRow) * num_vecs_per_row + vecIdx;
    tk::loadPackedVec(vec_in, reinterpret_cast<InType const*>(input) + readOffset);
    uint8_t fp8Scale;
    auto fp4Vals =
        tk::cvt_warp_fp16_to_fp4<T, SF_VEC_SIZE, SF_VEC_SIZE, false, /*DISABLE_FP4_FAST_MATH=*/false,
                                 std::false_type>(vec_in, globalEncodeScale, &fp8Scale);

    int64_t const writeOffset = static_cast<int64_t>(writeRow) * num_vecs_per_row + vecIdx;
    reinterpret_cast<PackedFp4Type*>(weightOutput)[writeOffset] = fp4Vals;

    int64_t sfOffset;
    if constexpr (SF_LAYOUT == tensorrt_llm::QuantizationSFLayout::LINEAR) {
      sfOffset = static_cast<int64_t>(writeRow) * num_vecs_per_row + vecIdx;
    } else if constexpr (SF_LAYOUT == tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4) {
      sfOffset = tk::get_sf_out_offset_128x4(writeRow, vecIdx, num_vecs_per_row);
    } else {
      sfOffset = tk::get_sf_out_offset_8x4(writeRow, vecIdx, num_vecs_per_row);
    }
    scaleOutput[sfOffset] = fp8Scale;
  }
}

// no-dedup: grid.x = num_tokens*top_k (one block per (token,expert) pair).
template <typename T, uint32_t BLOCK_SIZE, tensorrt_llm::QuantizationSFLayout SF_LAYOUT>
__global__ void fusedPermuteNvfp4QuantKernel(
    uint32_t numPairs,
    uint32_t n,
    uint32_t topK,
    T const* input,
    float globalScaleInv,
    int32_t const* expandedIdxToPermutedIdx,
    uint8_t* weightOutput,
    uint8_t* scaleOutput,
    float* perTokenScaleOutput) {
  uint32_t const expandedIdx = blockIdx.x;
  if (expandedIdx >= numPairs) return;
  int const writeRow = expandedIdxToPermutedIdx[expandedIdx];
  if (writeRow < 0) return;
  int const readRow = static_cast<int>(expandedIdx / topK);
  fused_quant_one_row<T, BLOCK_SIZE, SF_LAYOUT>(
      n, input, readRow, writeRow, globalScaleInv, weightOutput, scaleOutput, perTokenScaleOutput);
}

// dedup: grid.x = num_tokens (one block per source token, scatter to its top_k destinations).
template <typename T, uint32_t BLOCK_SIZE, tensorrt_llm::QuantizationSFLayout SF_LAYOUT>
__global__ void fusedPermuteNvfp4QuantDedupKernel(
    uint32_t numTokens,
    uint32_t n,
    uint32_t topK,
    T const* input,
    float globalScaleInv,
    int32_t const* expandedIdxToPermutedIdx,
    uint8_t* weightOutput,
    uint8_t* scaleOutput,
    float* perTokenScaleOutput) {
  constexpr int SF_VEC_SIZE = 16;
  using InType = tk::PackedVec<T, SF_VEC_SIZE>;
  using PackedFp4Type = std::conditional_t<SF_VEC_SIZE == 16, uint64_t, uint32_t>;
  uint32_t const token = blockIdx.x;
  if (token >= numTokens) return;
  uint32_t const num_vecs_per_row = (n + SF_VEC_SIZE - 1) / SF_VEC_SIZE;

  // pass 1: amax over the source token row (read once).
  InType vec_in;
  float localAmax = 0.f;
  for (uint32_t vecIdx = threadIdx.x; vecIdx < num_vecs_per_row; vecIdx += BLOCK_SIZE) {
    int64_t const vecOffset = static_cast<int64_t>(token) * num_vecs_per_row + vecIdx;
    tk::loadPackedVec(vec_in, reinterpret_cast<InType const*>(input) + vecOffset);
    std::remove_reference_t<decltype(vec_in.elts[0])> localAmax_(0.f, 0.f);
#pragma unroll
    for (int i = 0; i < SF_VEC_SIZE / 2; ++i) {
      localAmax_ = __hmax2(localAmax_, __habs2(vec_in.elts[i]));
    }
    localAmax = fmaxf(localAmax, static_cast<float>(__hmax(localAmax_.x, localAmax_.y)));
  }
  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  float const globalAmax = BlockReduce(tempStorage).Reduce(localAmax, cuda::maximum<>{});

  __shared__ float sPerTokenScale;
  if (threadIdx.x == 0) {
    sPerTokenScale = globalAmax * globalScaleInv;
  }
  __syncthreads();
  float const perTokenScale = sPerTokenScale;
  float const globalEncodeScale = reciprocal_approximate_ftz(perTokenScale);

  // Resolve this token's (valid) permuted destinations once (top_k of them).
  // top_k is small; thread 0 writes the per-token scale for each destination.
  if (threadIdx.x < topK) {
    int const writeRow = expandedIdxToPermutedIdx[token * topK + threadIdx.x];
    if (writeRow >= 0) perTokenScaleOutput[writeRow] = perTokenScale;
  }

  // pass 2: quantize each vec once, scatter to all valid destinations.
  for (uint32_t vecIdx = threadIdx.x; vecIdx < num_vecs_per_row; vecIdx += BLOCK_SIZE) {
    int64_t const readOffset = static_cast<int64_t>(token) * num_vecs_per_row + vecIdx;
    tk::loadPackedVec(vec_in, reinterpret_cast<InType const*>(input) + readOffset);
    uint8_t fp8Scale;
    auto fp4Vals =
        tk::cvt_warp_fp16_to_fp4<T, SF_VEC_SIZE, SF_VEC_SIZE, false, /*DISABLE_FP4_FAST_MATH=*/false,
                                 std::false_type>(vec_in, globalEncodeScale, &fp8Scale);
#pragma unroll 1
    for (uint32_t k = 0; k < topK; ++k) {
      int const writeRow = expandedIdxToPermutedIdx[token * topK + k];
      if (writeRow < 0) continue;
      int64_t const writeOffset = static_cast<int64_t>(writeRow) * num_vecs_per_row + vecIdx;
      reinterpret_cast<PackedFp4Type*>(weightOutput)[writeOffset] = fp4Vals;
      int64_t sfOffset;
      if constexpr (SF_LAYOUT == tensorrt_llm::QuantizationSFLayout::LINEAR) {
        sfOffset = static_cast<int64_t>(writeRow) * num_vecs_per_row + vecIdx;
      } else if constexpr (SF_LAYOUT == tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4) {
        sfOffset = tk::get_sf_out_offset_128x4(writeRow, vecIdx, num_vecs_per_row);
      } else {
        sfOffset = tk::get_sf_out_offset_8x4(writeRow, vecIdx, num_vecs_per_row);
      }
      scaleOutput[sfOffset] = fp8Scale;
    }
  }
}

// Launcher. `dedup` picks the variant. `n` (= hidden) must be a multiple of 16.
template <typename T>
void invokeFusedPermuteNvfp4Quant(
    uint32_t numTokens,
    uint32_t topK,
    uint32_t n,
    T const* input,
    float globalScaleInv,
    int32_t const* expandedIdxToPermutedIdx,
    uint8_t* weightOutput,
    uint8_t* scaleOutput,
    float* perTokenScaleOutput,
    tensorrt_llm::QuantizationSFLayout sfLayout,
    bool dedup,
    cudaStream_t stream) {
  constexpr uint32_t BLOCK_SIZE = 128;
  dim3 const block(BLOCK_SIZE);

  auto dispatch = [&](auto layoutTag) {
    constexpr tensorrt_llm::QuantizationSFLayout LAYOUT = decltype(layoutTag)::value;
    if (dedup) {
      dim3 const grid(numTokens);
      fusedPermuteNvfp4QuantDedupKernel<T, BLOCK_SIZE, LAYOUT><<<grid, block, 0, stream>>>(
          numTokens, n, topK, input, globalScaleInv, expandedIdxToPermutedIdx, weightOutput,
          scaleOutput, perTokenScaleOutput);
    } else {
      dim3 const grid(numTokens * topK);
      fusedPermuteNvfp4QuantKernel<T, BLOCK_SIZE, LAYOUT><<<grid, block, 0, stream>>>(
          numTokens * topK, n, topK, input, globalScaleInv, expandedIdxToPermutedIdx, weightOutput,
          scaleOutput, perTokenScaleOutput);
    }
  };

  if (sfLayout == tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4) {
    dispatch(std::integral_constant<tensorrt_llm::QuantizationSFLayout,
                                    tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4>{});
  } else {
    dispatch(std::integral_constant<tensorrt_llm::QuantizationSFLayout,
                                    tensorrt_llm::QuantizationSFLayout::SWIZZLED_8x4>{});
  }
}

}  // namespace sgl_fused_permute_quant
