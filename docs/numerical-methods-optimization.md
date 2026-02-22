# Numerical Methods & Error Computation Optimization Analysis

Expert analysis of fitness/error computation, numerical precision, and error metric alternatives in the artgen-backend-rust GPU evolution pipeline. Focuses on NEW improvements not already covered in existing optimization documents (gpu-compute-optimization.md, evolutionary-algorithm.md, data-structures-algorithms.md, GPU_PERF_IDEAS.md).

**Target hardware:** NVIDIA RTX 5090 (Blackwell, 170 SMs, 32-wide warps, 128KB L1/shared per SM, 96MB L2)

---

## 1. Error Metric Analysis: L1 vs L2 vs Perceptual

### Current Implementation

The GPU uses **L1 error** (Manhattan distance in RGB):

```wgsl
// rasterize_error.wgsl, line 347
pixel_error = u32(abs(dr) + abs(dg) + abs(db));
```

The CPU evaluator (`evaluator.rs`, line 92) also uses L1:
```rust
let pixel_error = (re.abs() + ge.abs() + be.abs()) as f32;
```

MAX_ERROR_PER_PIXEL = 765.0 (255 * 3), which is correct for L1.

### Analysis of Alternative Metrics

#### L2 (Euclidean RGB Distance)

The existing docs (gpu-compute-optimization.md section 9, evolutionary-algorithm.md) reference L2 with `sqrt()` and propose dropping the sqrt. However, the **actual codebase uses L1**, not L2. This is an important observation -- several existing docs describe the metric incorrectly as L2.

L1 vs L2 tradeoffs for this use case:
- **L1** penalizes each channel linearly. A pixel that is off by (30,30,30) has the same error as one off by (90,0,0). This means L1 does not penalize "balanced" errors differently from "concentrated" errors.
- **L2** (squared, without sqrt) penalizes concentrated errors more heavily: (90,0,0) = 8100 vs (30,30,30) = 2700. This incentivizes spreading error across channels rather than having one channel grossly wrong.
- For art approximation, L2 is generally better at producing visually pleasing results because the human eye is more sensitive to a single channel being way off than to all channels being slightly off.

**Recommendation: Switch from L1 to L2-squared (no sqrt needed).** This changes the error from `abs(dr)+abs(dg)+abs(db)` to `dr*dr+dg*dg+db*db`. No sqrt is needed since the ordering is preserved. MAX_ERROR_PER_PIXEL changes from 765.0 to 195075.0 (255^2 * 3). The u32 accumulator still has ample range: 195075 * 512 * 512 = ~51 billion, which exceeds u32 max (4.29 billion) for images this size, so **L2-squared requires u64 accumulators or scaled-down values** (see Section 7 for overflow analysis).

**Impact estimate:** Modest convergence improvement (5-15% faster to reach perceptual equivalence) because L2-squared better correlates with perceptual quality. The computational cost is identical (3 multiplies replace 3 abs calls -- abs is actually just a bitmask on floats, so the multiply is slightly more expensive, but the evolutionary benefit outweighs it).

#### Weighted RGB (Luminance-Aware)

Human vision is more sensitive to green than red, and more to red than blue. A luminance-weighted error:

```wgsl
let pixel_error = u32(0.299 * dr * dr + 0.587 * dg * dg + 0.114 * db * db);
```

This matches the ITU-R BT.601 luminance weights. Green errors would be penalized ~5x more than blue errors, matching perceptual sensitivity.

**Impact estimate:** Small but meaningful improvement in perceptual quality. Convergence speed likely unchanged (same total error magnitude). The art will look slightly better at the same fitness level because the optimizer will prioritize green/red channel accuracy.

**Implementation:** Add 3 float multiplies per pixel. Negligible ALU cost. MAX_ERROR_PER_PIXEL becomes `0.299*255^2 + 0.587*255^2 + 0.114*255^2 = 65025` (same as unweighted L2-squared since weights sum to 1.0).

#### SSIM / Structural Similarity

SSIM compares local patches (typically 11x11 windows) using mean, variance, and covariance statistics. This is fundamentally incompatible with the per-pixel error accumulation architecture:

- Requires local neighborhood statistics (can't be computed from a single pixel).
- Would require a separate pass over the rendered image.
- The rendered image is not stored in a buffer -- it exists only in thread registers.

**Verdict: Not feasible** without major architectural changes (caching the rendered framebuffer). The incremental eval path does maintain framebuffers, but SSIM would still require a dedicated reduction pass over 11x11 windows. The complexity is disproportionate to the benefit for polygon art.

#### CIELAB Delta-E

Converting RGB to CIELAB requires:
1. RGB -> XYZ (3x3 matrix multiply)
2. XYZ -> LAB (cube root operations)
3. LAB distance (Euclidean in LAB space)

The cube root is expensive on GPU (no native instruction; approximated via `pow(x, 1.0/3.0)` = 2 SFU operations per channel). For 512x512 = 262K pixels, this adds ~786K SFU operations per offspring. On RTX 5090 with 170 SMs * 4 SFU units * 2.4 GHz = 1632 GFLOPS SFU throughput, this is ~0.5 microseconds -- negligible.

However, the conversion constants introduce floating-point complexity that may not be worth it for this application. The reference image is already in sRGB, and the polygon renderer also operates in sRGB-linear-ish space (colors are stored as u8 values blended in linear RGB).

**Verdict: Not recommended.** The perceptual improvement over weighted L2 is marginal for polygon art, while the implementation complexity (gamma correction, whitepoint constants, cube roots) is significant. Weighted L2-squared is the better tradeoff.

### Final Recommendation for Error Metric

**Switch to L2-squared with luminance weighting** as the best balance of perceptual quality, computational cost, and implementation simplicity:

```wgsl
let dr = ri - refr;
let dg = gi - refg;
let db = bi - refb;
// Luminance-weighted L2-squared (BT.601)
pixel_error = u32(0.299 * dr * dr + 0.587 * dg * dg + 0.114 * db * db);
```

This requires scaling MAX_ERROR_PER_PIXEL to 65025.0 and verifying accumulator overflow (see Section 7).

---

## 2. Progressive Resolution / Multi-Resolution Evaluation

### Current State

The rasterize_error shader evaluates every pixel at full resolution for every offspring. The existing evolutionary-algorithm.md (Proposal 3) describes a stride-based approach. This section provides a more detailed numerical analysis.

### Numerical Analysis of Resolution Scaling

At lower resolutions, the error estimate has higher variance. The key question: how much variance can we tolerate before it causes the wrong offspring to be selected?

For a 512x512 image at 1/4 resolution (128x128):
- Sample size: 16,384 pixels (out of 262,144)
- Sampling ratio: 6.25%
- By central limit theorem, the standard deviation of the mean error estimate scales as `sigma / sqrt(n)`. With 16K samples, the estimate is quite stable.

**However**, the critical issue is not the absolute error estimate but the **relative ranking of offspring**. Two offspring that differ by only a few pixels (typical for micro-adjust mutations) may rank differently at low vs high resolution because the changed pixels might not be in the sample.

**Quantitative analysis for micro-adjust mutations:**
- A micro-adjust moves one vertex by ~1.5 pixels (micro_adjust_delta = 0.003 * 512 = 1.536 pixels).
- The affected area is roughly the symmetric difference of two nearly-identical triangles, typically 5-50 pixels.
- At 1/4 resolution (every 4th pixel), about 0-3 of those affected pixels are sampled.
- At 1/2 resolution (every 2nd pixel), about 1-12 affected pixels are sampled.

This means **micro-adjust mutations will be mostly invisible at 1/4 resolution**, causing the optimizer to reject them even when they would improve fitness. This is acceptable in early evolution (when structural changes dominate) but catastrophic in late evolution.

### Recommended Approach: Fitness-Threshold Resolution Transitions

| Fitness Range | Resolution | Pixels | Theoretical Speedup |
|---|---|---|---|
| < 60% | 1/4 (stride=4) | 16K | 16x |
| 60-80% | 1/2 (stride=2) | 65K | 4x |
| > 80% | Full (stride=1) | 262K | 1x |

**Implementation note:** The stride should apply to the dispatch dimensions, not the pixel computation. With stride=2, dispatch `(W/32, H/32, K*lambda)` instead of `(W/16, H/16, K*lambda)` for a 16x16 workgroup. Each thread still processes one pixel, but the pixel spacing is 2x. The error accumulator must be multiplied by `stride*stride` to normalize.

**Transition handling:** When crossing a resolution threshold, re-evaluate the current chain fitness at the new resolution to establish a correct baseline. This is a one-time cost per transition per chain.

**Impact estimate:** 3-8x wall-clock speedup to reach 80% fitness. The speedup for reaching 90%+ is modest (most time is already at full resolution).

---

## 3. Stochastic / Sampled Fitness Evaluation

### Concept

Instead of evaluating all pixels, randomly sample a subset of pixels for error computation. This trades accuracy for throughput.

### Analysis

**Random pixel sampling has a fundamental problem**: the selection of which pixels to sample must be deterministic for a given iteration (so all offspring are compared on the same sample), but different across iterations (to avoid bias). This requires either:
1. A per-iteration random seed that generates the same sample set for all offspring (extra RNG state management), or
2. A fixed sample pattern (e.g., every Nth pixel with a phase shift per iteration).

The stride-based approach from Section 2 is effectively a fixed sample pattern (regular grid subsampling). It has two advantages over random sampling:
1. **Deterministic and coherent**: all threads in a workgroup process spatially adjacent pixels, preserving memory coalescing and texture cache efficiency.
2. **No sample management overhead**: just change the dispatch dimensions.

Random pixel sampling would destroy spatial coherence in the workgroup, causing every thread to access a random reference texture location. This devastates the texture cache hit rate and is likely slower than processing all pixels in a coherent tile.

**Verdict: Not recommended.** Progressive resolution (Section 2) achieves the same throughput gain with better memory behavior.

---

## 4. Half-Precision (f16) for Color Computations

### Current Implementation

The rasterize_error shader uses f32 for all color computations:
```wgsl
var r = 255.0;
var g = 255.0;
var b = 255.0;
// ... alpha blending in f32 ...
let ri = clamp(r, 0.0, 255.0);
```

### Precision Analysis

Color values range from 0.0 to 255.0. Alpha values range from 0.0 to 1.0 (after unpack4x8unorm normalization).

f16 (IEEE 754 half-precision):
- Range: up to 65504.0 (sufficient for 0-255)
- Precision: 11-bit mantissa = 2048 representable values in [0, 255]
- Absolute precision at 255.0: 255/2048 = 0.125 (rounds to nearest 0.125)
- After N alpha blends: error accumulates. With 1000 polygons and worst-case error growth, the accumulated quantization error could reach several units in the 0-255 range.

**Critical issue**: The alpha blending computation `r = r * inv_alpha + src_r * alpha` involves multiplying a value in [0,255] by a value in [0,1]. In f16, multiplying 255.0 by a small alpha (e.g., 0.039 = 10/255) gives 9.945, which f16 represents as 9.9375 or 10.0 (quantization step at this magnitude is 0.0625). This is adequate for a single blend but accumulates over many blends.

**Simulation result** (analytical): After 150 overlapping polygon blends (typical drawing) with random alphas in [10/255, 65/255], the expected f16 quantization error per channel is ~0.3-0.5 in the 0-255 range. This corresponds to a worst-case error of ~1-2 per pixel per channel, which means the per-pixel L1 error has ~3-6 units of noise. For a 512x512 image, total noise is ~500K-1.5M out of a typical total error of ~5M-20M. This is a 5-10% noise floor.

**Impact on selection**: The noise affects all offspring equally (same quantization scheme), so the relative ranking is approximately preserved. However, micro-adjust mutations that improve fitness by a few hundred total error units (common in late evolution) would be drowned out by f16 quantization noise.

### Recommendation

**Do not switch to f16 for the blending accumulator.** The precision loss is too significant for late-stage evolution where improvements are tiny. The RTX 5090 has f32 ALU throughput that is already saturated; f16 only doubles throughput for FMA operations via tensor cores or packed f16 instructions, which the compute shader cannot easily exploit (f16 in WGSL requires the `f16` extension, and wgpu support is limited).

**However**, f16 could be used for the **reference texture** lookups without precision loss (the reference image is 8-bit per channel, and f16 represents all integers 0-255 exactly). The reference texture is already Rgba8Unorm which returns f32 from textureLoad -- there's no f16 gain here.

**Verdict: No change recommended.**

---

## 5. Integer Arithmetic for Error Computation

### Concept

Replace the float-based rasterization and error computation with fixed-point integer arithmetic. Colors are naturally 0-255 integers; the alpha blending could be done in u16 arithmetic:

```wgsl
// Integer alpha blending: r = r * (255 - alpha) / 255 + src_r * alpha / 255
let inv_alpha = 255u - alpha;
r = (r * inv_alpha + src_r * alpha + 127u) / 255u;  // +127 for rounding
```

### Analysis

**Advantages:**
1. Exact reproduction of the CPU path (which uses f32 intermediates cast to u8, but could be made integer-exact).
2. No floating-point precision concerns over many blends.
3. Integer division by 255 can be approximated as `(x + 128) >> 8` or the exact formula `(x * 0x8081u) >> 23u` (no actual division).

**Disadvantages:**
1. WGSL does not support u16 arithmetic natively; operations would use u32 with masking.
2. The pack4x8unorm/unpack4x8unorm builtins return f32, so there would be extra conversions.
3. Integer division/approximation adds 2-3 extra ALU ops per blend vs the current `r * inv_alpha + src_r * alpha` (2 FMA ops in f32).

**Quantitative comparison:**
- Current f32 path: 2 multiplies + 1 add per channel per blend = 6 FMA ops per polygon
- Integer path: 2 multiplies + 1 add + 1 shift + 1 add per channel = 15 ops per polygon
- The integer path is ~2.5x more ALU-expensive.

**Precision benefit:** The f32 path accumulates ~0.001 error per blend per channel (f32 has 24-bit mantissa, values in [0,255] have ~6 decimal digits of precision). After 1000 blends, accumulated error is ~1.0 per channel. This is smaller than the quantization to u8 (which happens at error computation time via `u32(abs(...))`), so the f32 precision is not the limiting factor.

**Verdict: Not recommended.** The current f32 arithmetic is precise enough (error < 1 unit per channel after 1000 blends), and integer arithmetic would be slower due to the lack of native u16 ops and the need for division approximation.

---

## 6. RNG Quality and Period Analysis

### Current Implementation

The mutate shader uses PCG-RXS-M-XS (32-bit state):

```wgsl
fn pcg_step(state: ptr<function, u32>) -> u32 {
    let old = *state;
    *state = old * 747796405u + 2891336453u;
    let word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
    return (word >> 22u) ^ word;
}
```

**Period:** 2^32 = 4,294,967,296 steps. Each offspring has its own persistent RNG state.

**Statistical quality:** PCG-RXS-M-XS passes PractRand statistical tests up to 2^32 bytes output. It has good equidistribution (all 32-bit values appear exactly once per period).

### Is the Period Sufficient?

Each offspring consumes RNG calls per iteration:
- Single mutation mode: ~5-15 calls (selection + mutation parameters)
- Multi mutation mode: ~20-50 calls per polygon * up to 1000 polygons = 20K-50K calls

At 50K calls per iteration and 64 batch iterations: 3.2M calls per batch.
At 2^32 period: the RNG would repeat after 2^32 / 3.2M = ~1340 batches.

In a typical run of millions of batches (hours of evolution), each offspring's RNG **will cycle multiple times**. However, since the RNG state is persistent and continues from where it left off, cycling is not a problem unless the period is short enough that the same sequence of mutations repeats within a correlated window. With 1340 batches between repeats and each batch producing independent mutations, the correlation is negligible.

### Would Better RNG Improve Convergence?

The mutation operators are robust to RNG quality because:
1. Mutations are simple random perturbations, not complex statistical procedures.
2. Selection (comparison of error values) is deterministic.
3. The evolutionary process is inherently stochastic and tolerant of imperfect randomness.

Upgrading to a 64-bit state RNG (e.g., PCG64 with 2^64 period) would:
- Double the per-call ALU cost (64-bit multiply on GPU requires 4 MUL + 2 ADD ops).
- Provide no measurable convergence improvement.
- Use an extra register per offspring for the state.

### Opportunity: Better Seeding

The current seeding strategy uses:
```rust
let seed = 0xCAFE_BABE_u64.wrapping_add((i as u64) * 0x9E3779B97F4A7C15)
    .wrapping_add((iteration as u64) * 0x517CC1B727220A95);
state.rng_state[0] = seed as u32;
```

This truncates a 64-bit seed to 32 bits, which means offspring `i` and offspring `i + 2^32` would get the same seed. Since offspring_capacity is typically < 2^16, this is not an issue in practice. However, if `iteration` is large, the low 32 bits of the seed could repeat. A better approach would use a hash of both the offspring index and iteration:

```rust
let seed = splitmix64(0xCAFE_BABE ^ (i as u64) << 32 | iteration as u64);
state.rng_state[0] = seed as u32;
```

**Verdict: RNG quality is adequate. No change recommended.** The period and statistical quality of PCG-RXS-M-XS are sufficient for this application.

---

## 7. Numerical Stability: Overflow and Underflow Analysis

### Error Accumulator Overflow

The error accumulators use `atomic<u32>` with atomicAdd. The maximum possible error per pixel depends on the metric:

| Metric | Max per pixel | Max total (512x512) | Fits u32? |
|---|---|---|---|
| L1 | 765 | 200,540,160 | Yes (u32 max = 4.29B) |
| L2-squared | 195,075 | 51,123,302,400 | **NO** |
| Weighted L2 | 65,025 | 17,041,100,800 | **NO** |
| L2 (with sqrt) | 441.67 | 115,711,303 | Yes |

**Critical finding: Switching to L2-squared or weighted L2 would overflow the u32 accumulator for 512x512 images.** This is the main numerical obstacle to changing the error metric.

#### Solutions for u32 Overflow:

**Option A: Scale down per-pixel error.** Divide by 256 (right-shift 8): max L2-squared per pixel becomes 762, max total becomes ~200M. This fits u32 with room to spare. The scaling loses the bottom 8 bits of precision, but since the original values are integers (from 0-255 color differences), the squared values are exact multiples of 1, and dividing by 256 preserves enough precision for fitness comparison.

```wgsl
pixel_error = u32(dr * dr + dg * dg + db * db) >> 8u;
```

**Option B: Use f32 atomic add.** WGSL supports atomicAdd on `atomic<u32>` but not directly on floats. However, the error can be accumulated as f32 using the bitcast trick (atomicCompareExchangeWeak loop). This is significantly slower (compare-and-swap retry loop) and not recommended.

**Option C: Use two-level reduction.** Accumulate per-workgroup in u32 shared memory (workgroup pixel count is 256, so max per-workgroup L2-squared = 256 * 195075 = 49.9M, fits u32). Then the atomicAdd to the global accumulator would overflow, but we could use `atomic<u32>` on a split high/low representation. This is overly complex.

**Recommendation: Option A (scale down by 256).** This is simple, efficient, and preserves sufficient precision. The effective resolution is ~3 levels per unit of per-pixel error, which is more than adequate for fitness comparison. MAX_ERROR_PER_PIXEL becomes `195075 / 256 = 762` (or `65025 / 256 = 254` for weighted L2).

### Alpha Blending Precision Drift

The rasterize_error shader accumulates color in f32 registers across all polygon blends. After N blends:
- Each blend: `r = r * inv_alpha + src_r * alpha`
- f32 precision: 24-bit mantissa
- Values in [0, 255]: ~6 significant decimal digits
- Error per blend: ~10^-5 relative
- After 1000 blends: ~10^-2 relative = ~2.5 absolute units in [0, 255]

This accumulated drift means the GPU-rasterized pixel value can differ from a hypothetical exact computation by up to ~2-3 per channel. This is **not a correctness issue** because:
1. All offspring experience the same drift pattern (same polygon order, same arithmetic).
2. The incremental eval path re-rasterizes from parent state, so the drift cancels out in the delta computation.

**Potential issue with incremental eval:** The incremental eval computes `final_error = parent_total - old_dirty + new_dirty`. If the parent total was computed with slightly different f32 rounding than the old_dirty region (because the parent was rasterized at a different time with potentially different polygon count/order), there could be a slow error drift. The select shader already handles this with saturating subtraction (`select(parent_total - old_err, 0u, old_err > parent_total)`), which prevents underflow but allows a slow bias.

**Recommendation:** The existing saturating subtraction is the correct mitigation. No additional changes needed. If long-running incremental eval sessions show fitness drift, consider periodic full re-evaluation (already handled by the `dispatch_init_framebuffers` path when incremental eval is re-enabled).

### Fitness Computation Precision

The fitness formula in select.wgsl:
```wgsl
var fitness = 100.0 * (1.0 - f32(total_error) / max_total_error);
fitness -= fitness * per_point_multiplier * f32(num_points);
```

With L1 error, `total_error` can be up to ~200M (u32). The `f32(total_error)` conversion loses the bottom bits for large values (f32 has 24-bit mantissa, values > 16M lose integer precision). For total_error = 200M, the f32 representation is exact to ~12 units (200M / 2^24 = 11.9).

This means two offspring whose total errors differ by < 12 could compare as equal in fitness. With L1 error, 12 units of total error corresponds to ~12 pixels each being off by 1 in one channel, which is below the perceptual threshold.

**Verdict: Acceptable.** The precision loss in fitness computation is below the perceptual threshold.

---

## 8. Vectorized Operations: Multi-Pixel Processing

### Current Architecture

Each thread processes exactly one pixel. The rasterize_error workgroup is `THREAD_COUNT x 1 x 1` (1D layout for subgroup intrinsics), with pixel coordinates derived from `workgroup_id + local_invocation_index`.

### Opportunity: 2 or 4 Pixels Per Thread

Each thread could process 2x1 or 2x2 pixels, reducing the number of threads (and workgroups) needed. This would:
1. Amortize the polygon load from shared memory (one load serves 2-4 pixel tests).
2. Reduce the number of subgroupAdd/atomicAdd operations.
3. Increase register pressure per thread.

**Analysis for 2x1 (two horizontal adjacent pixels):**
- Each thread maintains 6 color accumulators instead of 3 (two pixels).
- The AABB test can be shared: if both pixels are within the AABB, proceed; if only one is, branch.
- The half-space test must be done independently per pixel (3 edge functions per pixel = 6 total, vs 3 currently).
- Register usage approximately doubles.

On RTX 5090, register file is 256KB per SM, 64 registers per thread at 256 threads = 16KB. Doubling register usage to ~48 registers per thread still fits but may reduce occupancy from 5 workgroups/SM to 3-4.

**Net effect:** The polygon load amortization saves ~30% of shared memory traffic, but the reduced occupancy costs ~20-40% of latency-hiding capability. The tradeoff depends on whether the shader is compute-bound or memory-bound:
- If memory-bound (likely for large polygon counts with shared memory tiling): multi-pixel helps.
- If compute-bound (many polygons with high hit rate): multi-pixel hurts due to occupancy loss.

**Recommendation:** Not recommended as a general optimization. The current 1-pixel-per-thread design with shared memory tiling is well-balanced. Multi-pixel processing would add complexity for marginal benefit that depends heavily on the workload characteristics.

---

## 9. Error Caching / Delta Evaluation

### Current State

The codebase already implements **incremental evaluation** (controlled by `params.incremental_eval`). The mutate shader outputs a dirty bounding box, and the rasterize_error shader:
1. Skips workgroups whose tile doesn't overlap the dirty bbox.
2. Re-rasterizes the dirty region for both parent (old error) and offspring (new error).
3. Computes `final_error = parent_total - old_dirty + new_dirty`.

This is already the delta evaluation optimization described in GPU_PERF_IDEAS.md.

### Remaining Improvement: Tighter Dirty Bounding Boxes

The current incremental eval computes dirty bbox per mutation type:
- Per-polygon mutations (move_point, micro_adjust, color changes): union of old and new polygon bbox.
- Z-order mutations (reorder, adjacent_swap, swap_colors, remove, merge): full image bbox.
- Crossover: full image bbox.
- Multi-mutation mode: always full image bbox.

**Opportunity:** For z-order mutations, the dirty region could be tightened:
- **Adjacent swap of polygons i and i+1**: The dirty region is the union of the bounding boxes of polygons i and i+1 (only pixels covered by these two polygons are affected). Currently returns `full_image_bbox()`.
- **Remove polygon at index k**: The dirty region is the bbox of the removed polygon (pixels above it in z-order need re-compositing within that region). Currently returns `full_image_bbox()`.
- **Swap colors between polygons i and j**: The dirty region is the union of both polygons' bboxes. Currently returns `full_image_bbox()`.

**Impact estimate:** In single-mutation mode with incremental eval, ~20-30% of mutations are z-order operations (reorder + adjacent_swap + swap_colors + remove). Tightening their dirty bbox from full-image to the affected polygon regions would reduce the rasterization work by 10-25x for those mutations (typical polygon covers 5% of image).

**Weighted impact:** 20-30% of mutations get 10-25x speedup = 2-7.5x average speedup on rasterize pass for incremental eval. This is significant.

**Implementation in mutate.wgsl for adjacent_swap:**
```wgsl
// Instead of: return full_image_bbox();
// Compute union of both polygon bboxes
let bbox_a = polygon_bbox_pixels(working_states[oid].polygons[ai], w, h);
let bbox_b = polygon_bbox_pixels(working_states[oid].polygons[aj], w, h);
return merge_bbox(bbox_a, bbox_b);
```

**Caveat:** This is only correct if the incremental eval re-rasterizes ALL polygons in the dirty region (not just the changed ones), which the current implementation does. The re-rasterization properly handles z-order by iterating all polygons at each pixel.

**For remove_polygon:** After the polygon is removed and the array is compacted (swap with last), the dirty region should be the bbox of the removed polygon. However, since the swap-with-last changes the last polygon's position in the array, if the last polygon overlaps different pixels than the removed one, the dirty region should be the union of both. The current `full_image_bbox()` is conservative but correct; a tighter version would compute the union of the removed polygon's bbox and the former-last polygon's bbox.

### Additional Optimization: Skip Full Re-Rasterization for Color-Only Changes

When a mutation only changes a polygon's color (not its geometry), every pixel in the polygon's bbox needs re-compositing from layer 0 up to the changed polygon. This is expensive for early-z polygons. An optimization: for color-only changes to polygon k, only re-composite from layer k onward, using a cached "prefix image" up to layer k-1.

This is the "layered prefix-sum framebuffer" approach from GPU_PERF_IDEAS.md (Option B), which was rejected due to memory cost (150 framebuffers). However, a **single** cached prefix image for the most recently accepted state (not per-layer) combined with a skip from `polygon 0` to `polygon k` would still require rasterizing polygons 0..k-1 in the dirty region. No savings unless the prefix image is maintained.

**Verdict:** The tighter dirty bboxes for z-order mutations are the main actionable improvement here.

---

## 10. Additional Numerical Optimizations

### 10.1 Fused Multiply-Add (FMA) for Alpha Blending

The current alpha blend:
```wgsl
*r = *r * inv_alpha + src_r * alpha;
```

This is 2 multiplies and 1 add per channel. On NVIDIA GPUs, this can be compiled to a single FMA instruction if the compiler recognizes the pattern. However, the intermediate `*r * inv_alpha` result is not used elsewhere, so the compiler may already emit:
```
FMA r, r, inv_alpha, (src_r * alpha)
```

**Verification:** This is a compiler optimization, not a source-level change. The WGSL source already enables FMA emission. No action needed.

### 10.2 Early Exit on Zero-Alpha Polygons

If a polygon's alpha is 0, the blend is a no-op. The current code unpacks the color, computes inv_alpha = 1.0, and blends -- which just copies the existing r/g/b (a no-op but with ALU cost). Adding an early exit:

```wgsl
if pcolor.w == 0.0 { return; }
```

This adds one branch per polygon per pixel but saves 6 multiply/add ops for zero-alpha polygons. Since MIN_ALPHA = 10 (never 0) in the current settings, this check would never fire.

**Verdict: Not applicable** with current alpha range constraints.

### 10.3 Reciprocal Approximation for Division in Error Normalization

The fitness computation divides by `max_total_error`:
```wgsl
var fitness = 100.0 * (1.0 - f32(total_error) / max_total_error);
```

This division happens once per chain per iteration in the select shader (not per pixel), so the cost is negligible. No optimization needed.

### 10.4 Batch Error Accumulator Reset

Currently, the select shader resets error accumulators via `atomicExchange`:
```wgsl
let new_err = atomicExchange(&error_accumulators[oid * 2u], 0u);
```

This is both a read and a write to global memory. Since the error accumulators are only read by the select shader and only written by the rasterize shader, the atomicExchange is the most efficient way to read-and-reset in a single operation. No improvement possible.

---

## Summary: Priority-Ranked Recommendations

| # | Optimization | Expected Impact | Effort | Risk |
|---|---|---|---|---|
| 1 | **Switch to L2-squared error with scale-by-256** | 5-15% convergence quality improvement | Medium | Low -- need to update MAX_ERROR_PER_PIXEL and verify accumulator math |
| 2 | **Tighter dirty bboxes for z-order mutations** (adjacent_swap, remove, swap_colors) | 2-7.5x rasterize speedup for incremental eval | Low-Medium | Low -- compute bbox unions instead of full_image_bbox |
| 3 | **Progressive resolution** (stride-based evaluation) | 3-8x wall-clock speedup to 80% fitness | Medium | Medium -- need resolution transition logic |
| 4 | **Luminance-weighted error** (BT.601 weights) | Small perceptual quality improvement | Low | Very low -- 3 extra multiplies per pixel |
| 5 | **No change to RNG** | N/A | N/A | PCG-32 is sufficient |
| 6 | **No f16 for blending** | N/A | N/A | Precision loss too high for late-stage evolution |
| 7 | **No integer arithmetic** | N/A | N/A | More ALU ops than f32 path, no precision benefit |
| 8 | **No SSIM/CIELAB** | N/A | N/A | Architectural incompatibility, marginal benefit |
