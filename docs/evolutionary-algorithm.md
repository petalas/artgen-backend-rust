# Evolutionary Algorithm Analysis & Improvement Proposals

This document analyzes the current evolutionary strategy in artgen-backend-rust and proposes algorithmic improvements to speed up convergence and improve final image quality. Focuses on ideas not already implemented. Last updated 2026-02-22.

## Current Architecture Summary

The system uses a **(1+lambda)-ES** running entirely on the GPU via wgpu compute shaders:

- **Population**: Up to 1024 independent chains (default 4), each holding a drawing of up to 1000 triangles (16 bytes each = 16KB state per chain).
- **Per iteration**: Each chain spawns lambda offspring (1-64, power-of-2, default 64). Offspring are independently mutated, rasterized, error-scored via L1 pixel error, and the best offspring is compared against the parent. Three GPU passes per iteration: mutate -> rasterize_error -> select.
- **Selection**: Strictly elitist -- offspring replaces parent only if fitness strictly improves, with 50% neutral acceptance for plateau traversal (coin flip using offspring RNG bit).
- **Crossover**: Tournament selection (size 3) across the entire population, uniform crossover that walks both parent polygon arrays in lockstep with per-slot coin flip. Probability 0.2 default.
- **Mutation**: 17 mutation operators. Single-mutation mode (default) uses weighted roulette-wheel selection. Multi-mutation mode applies each independently per polygon. Mutations include: add/remove polygon, reorder (random swap), adjacent swap, scale, rotate, offset polygon, move point (large/medium/micro), change color (full/micro), adjust brightness, adjust saturation, merge polygons, clone+jitter, swap colors.
- **Adaptive mutation scale**: 1.2x on accept (capped at 2.0), pow(0.99, 1/lambda) decay on reject (floor at 0.2), burst to 1.5 on stagnation (>10000/lambda iterations). Only affects offspring indices 1..lambda-1 when enabled; offspring 0 always uses scale 1.0.
- **Fitness**: L1 error per pixel (`abs(dr) + abs(dg) + abs(db)`, max 765 per pixel) summed across all pixels, converted to percentage, with a tiny complexity penalty (`PER_POINT_MULTIPLIER = 1/5000000` per vertex).
- **Degenerate triangle culling**: On acceptance, compacts polygon array by removing triangles with cross-product area < 0.00001.
- **Tile culling**: Optional spatial binning pass assigns polygons to screen tiles; rasterize shader only processes polygons overlapping each tile.
- **Incremental evaluation**: Optional dirty-bbox tracking from mutate shader; rasterize_error skips tiles outside the dirty region and computes delta error against parent's total.
- **Double-buffered staging**: Overlaps GPU execution of batch N+1 with CPU readback of batch N's results.
- **No migration/islands**: The island model was removed. Chains are fully independent (except for crossover, which does tournament selection across all chains, and the global-best tracking via atomicMax in the control flags buffer).

---

## Proposal 1: Stratified Offspring Mutation (Deterministic Diversity Within Lambda)

### Problem

With lambda=64 offspring per chain, all 64 independently sample from the same weighted probability distribution. This leads to high redundancy: many offspring attempt the same mutation category (e.g., multiple micro-adjusts on different vertices that happen to be nearby). The information gain per iteration is suboptimal because offspring explore overlapping regions of the mutation space.

The probability distribution heavily weights per-polygon mutations (they scale with polygon count), so with 150 polygons and lambda=64, the overwhelming majority of offspring apply minor vertex or color tweaks. Structural mutations (add/remove polygon, reorder) are extremely rare per iteration despite having the highest potential impact at certain fitness levels.

### Solution

Partition the lambda offspring into **deterministic strata** so each mutation category is guaranteed representation:

```
offspring 0:        always add_polygon
offspring 1:        always remove_polygon (if count > min)
offspring 2:        always reorder (random swap)
offspring 3:        always crossover
offspring 4-7:      move_point (large delta, random polygon)
offspring 8-11:     medium_move (random polygon)
offspring 12-19:    micro_adjust vertex (random polygon, random vertex)
offspring 20-27:    micro_adjust color (random polygon, random channel)
offspring 28-31:    scale/rotate/offset polygon
offspring 32-63:    standard roulette-wheel selection (as today)
```

The stratification map is configurable: the first N slots are reserved for guaranteed-type mutations, the remaining lambda-N use the current probabilistic selection. This is essentially **Latin hypercube sampling** applied to the discrete mutation type space.

### GPU Implementation

In `mutate.wgsl`, add a branch at the top of the mutation path:

```wgsl
if params.stratified_mutation == 1u && offspring_local_idx < STRATIFIED_SLOTS {
    // Deterministic mutation type based on offspring index
    let mutation_type = stratification_table[offspring_local_idx];
    apply_specific_mutation(rng, oid, count, mutation_scale, mutation_type);
} else {
    // Existing roulette-wheel or multi-mutation path
    ...
}
```

The stratification table can be encoded as a small lookup in shader constants (no buffer needed for lambda <= 64).

### Expected Impact

High. With lambda=64, the current system generates ~0.4 add_polygon attempts per iteration (64 * 0.006 probability weight after normalization). With stratification, exactly 1 add_polygon is evaluated every iteration. For rare but high-impact mutations, this is a 2-3x improvement in discovery rate. Literature on stratified sampling in ES shows 15-25% faster convergence vs. pure random sampling at the same lambda.

### Implementation Complexity

Low-Medium. New branch in mutate.wgsl (no buffer changes). One new boolean param. The tricky part is getting the stratification proportions right -- should be configurable or fitness-adaptive.

---

## Proposal 2: Natural Evolution Strategy (NES) Gradient from Lambda Offspring

### Problem

The (1+lambda)-ES discards fitness information from all but the single best offspring. With lambda=64, that means 63/64 evaluations contribute nothing to the search direction. The algorithm uses each offspring as a binary "better or worse" test, ignoring the magnitude of fitness differences. This is maximally wasteful of the expensive rasterize+error evaluations.

### Solution

Implement a **fitness-weighted recombination** inspired by Natural Evolution Strategies (NES) and CMA-ES. Instead of selecting the single best offspring, use the fitness ranking of all lambda offspring to compute a weighted average mutation direction:

1. After rasterize_error computes errors for all lambda offspring, rank them by fitness.
2. Compute recombination weights using log-linear ranking (standard CMA-ES weights): `w_i = max(0, ln(lambda/2 + 1) - ln(rank_i))`, normalized to sum to 1.
3. The accepted drawing is a **weighted blend** of the top-ranked offspring's polygon states.

For polygon art, "weighted blend" cannot directly interpolate polygon arrays (different polygon counts, different orderings). Instead, apply it to the **mutation delta only**: each offspring records which polygon it mutated and the delta applied. The accepted step is the weighted sum of the top-k deltas.

### Simplified Version: Weighted Multi-Accept

A more practical adaptation: instead of blending, run a second round of mutation. From the top-k offspring, identify which polygon indices they mutated and what type of mutation was applied. On the next iteration, bias mutation probability toward those polygon indices and mutation types. This is a lightweight form of gradient information that feeds back into the mutation distribution.

### Simplest Version: Accept Top-K Sequentially

Even simpler: if the best offspring improves fitness, also check the 2nd-best offspring's mutation against the newly accepted state. If it also improves, accept it too. This "greedy sequential accept" captures correlated improvements that a single-winner selection misses. With single-mutation mode, this is safe because individual mutations are small and typically non-interfering.

In the select shader:

```wgsl
// After accepting best offspring:
if lambda >= 2u {
    let second_best_oid = ...;  // from reduction
    // Copy second-best mutation on top of accepted state
    // Re-evaluate error in dirty region only (incremental eval makes this cheap)
    // Accept if still improving
}
```

### Expected Impact

The full NES approach could deliver 30-50% faster convergence (matching the theoretical O(lambda) improvement over (1+1)-ES from using all offspring information). The sequential multi-accept version is simpler and could capture 10-20% of that benefit.

### Implementation Complexity

Full NES: Very High (blending polygon arrays is fundamentally hard). Sequential multi-accept: Medium (needs a "2nd-best" output from the reduction, plus a mini re-evaluation in select). Mutation-bias feedback: Low-Medium (store winning polygon index + mutation type in chain state, read it in next iteration's mutate).

---

## Proposal 3: Coarse-to-Fine Screening (Two-Phase Evaluation)

### Problem

With lambda=64 offspring and a 512x512 image, each iteration performs 64 full rasterizations (each visiting 150 polygons * 262K pixels = ~39M polygon-pixel tests). Roughly 90-95% of these offspring are rejected. The overwhelming majority of rasterization work is wasted on offspring that will never be accepted.

### Solution

Implement a **two-phase evaluation** where a cheap coarse screen filters most offspring, and only promising survivors get full-resolution evaluation:

**Phase 1: Coarse screen** (1/4 resolution = 128x128 = 16K pixels)
- Dispatch rasterize_error with stride=4 (or on a pre-downsampled reference)
- Each offspring gets a coarse error estimate at 1/16 the cost
- Identify the top-K offspring (e.g., K=4) that have the lowest coarse error

**Phase 2: Full evaluation** (only top-K offspring)
- Dispatch full-resolution rasterize_error for only K offspring
- Select the best from these K

### GPU Implementation

This requires two rasterize_error dispatches per iteration, with a compact/filter step between them. The compact step identifies the top-K offspring and writes their indices to a small buffer.

Alternatively, with incremental evaluation already partially implemented, the coarse screen could simply use a sparser pixel sampling pattern within the existing dispatch by having threads sample every 4th pixel via the local invocation index.

### Expected Impact

Very High. If K=4 and lambda=64, the total rasterization work becomes: `64 * 16K + 4 * 262K = 1M + 1M = 2M pixels` vs. the current `64 * 262K = 16.8M pixels`. That is an **8x reduction** in the dominant bottleneck (rasterize_error typically consumes 60-80% of GPU time).

The screening quality is high because L1 error at 1/4 resolution is strongly correlated with full-resolution L1 error -- the coarse estimate correctly identifies the best offspring with >90% probability for typical polygon art (smooth color gradients, few high-frequency features).

### Implementation Complexity

Medium-High. Requires: a second rasterize dispatch with stride logic, a small buffer for top-K indices, a compact/sort step between phases (could be a mini compute pass), and changes to the select shader to only read K offspring errors instead of lambda.

### Caveats

The coarse screen might miss offspring with localized improvements (e.g., a micro-adjust that only affects a few pixels). For incremental-eval mode, the dirty-bbox information could be used to adaptively skip the coarse screen when the dirty region is already small.

---

## Proposal 4: Error-Guided Polygon Placement and Mutation Targeting

### Problem

New polygons are placed at uniformly random positions with random colors. The acceptance rate of `add_polygon` mutations drops precipitously as fitness improves: at 90%+ fitness, only a tiny fraction of the image has significant error, and a randomly placed polygon has <1% chance of landing in a useful region. Similarly, `rand_u32(rng, count)` selects polygons to mutate uniformly, even though most mutations to well-placed polygons are wasted (they already contribute optimally) while polygons overlapping high-error regions have the most room for improvement.

### Solution

Maintain a lightweight **per-chain error grid** (8x8 = 64 cells) that tracks which spatial regions have the highest error. Use this to:

1. **Bias add_polygon placement**: Sample the origin point from the high-error region with 60% probability, random otherwise.
2. **Bias polygon selection for mutation**: When selecting which polygon to mutate, prefer polygons whose centroid falls in a high-error region.

The error grid is updated cheaply in the select shader: on acceptance, use the offspring's total error decomposed by region (computed during rasterize_error via shared-memory per-tile accumulation). On rejection, the grid is unchanged.

### Lightweight Implementation (4x4 Grid)

The simplest version uses a 4x4 grid (16 cells), requiring only 16 u32s (64 bytes) per chain in the drawing state header. This fits within the existing `GpuDrawingState` by repurposing the unused `rng_state[1..3]` fields or extending the header by one cache line.

In `rasterize_error.wgsl`, each workgroup already knows its tile coordinates and accumulates pixel error. Adding a per-region accumulation is a few lines: each thread 0 also `atomicAdd`s to the chain's grid cell.

In `mutate.wgsl`, when adding a polygon, select the grid cell with highest error using a simple max scan (16 iterations), then place the polygon origin within that cell.

### Expected Impact

High. Error-guided placement should improve `add_polygon` acceptance rate by 3-10x in the 80-95% fitness range, directly accelerating convergence during the mid-to-late phase where polygon addition slows to a crawl. Biased polygon selection should reduce wasted mutations by 10-20%.

### Implementation Complexity

Low-Medium (4x4 grid version). Requires: 16 u32s added to chain state (or a small side buffer), a few lines in rasterize_error for regional accumulation, and bias logic in mutate for placement and polygon selection.

---

## Proposal 5: Simulated Annealing Acceptance on Stagnation

### Problem

The current selection is strictly elitist with 50% neutral acceptance (equal-fitness coin flip). The stagnation response (burst mutation_scale to 1.5 after 10000/lambda rejections) is blunt: it destabilizes the solution without providing any mechanism to accept temporarily worse solutions that might lead to better basins.

In polygon art, the fitness landscape has many narrow valleys separated by small ridges. Reordering two overlapping polygons might decrease fitness by 0.001% but enable subsequent mutations that improve fitness by 0.1%. The current algorithm can never traverse such ridges.

### Solution

Add **stagnation-triggered simulated annealing** in the select shader. When `stagnation_counter` exceeds a threshold, allow acceptance of slightly worse offspring with probability depending on the fitness gap:

```wgsl
// In select shader, when fitness <= current_fitness and stagnation detected:
let stagnation = chain_states[chain_id].stagnation_counter;
if stagnation > stagnation_threshold {
    let delta = current_fitness - fitness;
    // Temperature decays as stagnation grows beyond threshold
    let base_temp = 0.01;  // 0.01% of fitness range
    let temp = base_temp * exp(-f32(stagnation - stagnation_threshold) * 0.001);
    let p_accept = exp(-delta / max(temp, 0.0001));
    let rand_bits = working_states[best_offspring_id].rng_state.x;
    if f32(rand_bits) / 4294967296.0 < p_accept {
        // Accept worse solution
        chain_states[chain_id].stagnation_counter = 0u;  // Reset stagnation
        ...
    }
}
```

The key design choice: SA acceptance **only activates after stagnation**, not during normal operation. This means it cannot hurt convergence during productive phases. It only kicks in when the algorithm is already stuck, providing an escape mechanism that is strictly better than the current stagnation burst.

### Expected Impact

Moderate-High. The benefit is image-dependent: complex images with many overlapping polygons (portraits, detailed scenes) have more local optima to escape. Expected improvement: 3-8% better final fitness at convergence for complex images, with negligible impact on simple images.

### Implementation Complexity

Low. Changes are confined to ~20 lines in `select.wgsl`. No new buffers, no new passes. Could reuse the existing stagnation_counter and threshold.

---

## Proposal 6: Polygon Splitting as a Structured Mutation

### Problem

The `add_polygon` mutation creates a new triangle at a random position with random color. Even with error-guided placement (Proposal 4), the new triangle's shape and color must be independently discovered by evolution. Meanwhile, the existing polygon that covers a high-error region already encodes useful information about approximate position and color. What is needed is not a new random polygon but a **refinement** of an existing one.

### Solution

Add a **split polygon** mutation: select an existing triangle, subdivide it into 2-3 smaller triangles, and let evolution differentiate their colors:

**Centroid split** (1 triangle -> 3):
```
Given triangle ABC with centroid M = (A+B+C)/3:
  -> triangle ABM, triangle BCM, triangle CAM
All three inherit the parent's color.
```

**Edge midpoint split** (1 triangle -> 2):
```
Pick the longest edge, say AB. Midpoint P = (A+B)/2.
  -> triangle APC, triangle PBC
Both inherit the parent's color.
```

After splitting, the polygon count increases by 1 or 2 but the rendered image is nearly unchanged (the sub-triangles cover the same area with the same color). Evolution can then independently adjust each sub-triangle's color and position to capture finer detail.

### When to Split

- Bias toward polygons with large area (they cover more pixels and benefit most from subdivision)
- Bias toward polygons overlapping high-error regions (if Proposal 4 is implemented)
- Only split when polygon_count < max_polygons - 1 (need room for the new triangles)

### Expected Impact

High. This provides a principled way to increase drawing complexity. Unlike random `add_polygon`, splitting is **fitness-neutral at creation** (the image does not change), so it does not require a lucky mutation to be accepted -- the split itself is accepted, and subsequent iterations refine the sub-triangles. This directly addresses the representational bottleneck.

### Implementation Complexity

Medium. New mutation type in `mutate.wgsl` (centroid and vertex computation are already available). Needs to insert 1-2 new polygons at adjacent indices to maintain z-ordering.

---

## Proposal 7: Cauchy-Distributed Mutation Steps (Heavy-Tailed Exploration)

### Problem

All spatial mutations use uniform random perturbations within a fixed delta range scaled by mutation_scale. The uniform distribution is thin-tailed: perturbations larger than delta * mutation_scale never occur. When the optimal move is further than the current delta allows, the algorithm cannot reach it in a single step. Multiple small steps through the fitness landscape are unreliable because each intermediate position must improve fitness.

### Solution

Replace the uniform distribution for exploratory mutations (`move_point`, `offset_polygon`) with a **Cauchy distribution**. Keep uniform for exploitative mutations (`micro_adjust`). The Cauchy distribution produces small perturbations most of the time but has heavy tails that occasionally produce large jumps.

```wgsl
fn cauchy_sample(rng: ptr<function, u32>, gamma: f32) -> f32 {
    let u = rand_f32(rng) - 0.5;  // uniform in (-0.5, 0.5)
    return gamma * tan(3.14159265 * u);
}

// Usage in move_point:
let delta = cauchy_sample(&rng, params.move_point_max_delta * mutation_scale);
v0.x = clamp(v0.x + delta, 0.0, 1.0);
```

### Expected Impact

Moderate. Literature on Fast Evolutionary Programming consistently shows 10-30% convergence speedup on multimodal landscapes. The benefit is most pronounced when the algorithm is trapped in a basin and needs to jump to a better region.

### Implementation Complexity

Low. One new function, change 4-6 call sites in `mutate.wgsl`. No buffer changes, no new parameters.

---

## Proposal 8: Fitness-Proportional Polygon Selection (Biased Targeting)

### Problem

Polygons are selected uniformly at random for mutation (`rand_u32(rng, count)`). But in a layered alpha-blended rendering, later polygons (higher indices) are rendered on top and are more visually significant. Mutating an early polygon that is mostly occluded by later polygons changes the genotype without meaningfully changing the phenotype -- a wasted evaluation.

### Solution

Replace uniform polygon selection with a distribution biased toward later (top-layer) polygons:

```wgsl
fn select_polygon_biased(rng: ptr<function, u32>, count: u32) -> u32 {
    let u = rand_f32(rng);
    let v = rand_f32(rng);
    let idx = u32(max(u, v) * f32(count));
    return min(idx, count - 1u);
}
```

The `max(u, v)` of two uniform samples produces a triangular distribution that linearly increases with index. Polygon at index `count-1` (top layer) is selected twice as often as one at index 0 (bottom layer).

### Expected Impact

Moderate. Reduces wasted mutations by an estimated 10-20%. The benefit increases with polygon count (more layers = more occlusion = more waste from uniform selection).

### Implementation Complexity

Very Low. Replace `rand_u32(rng, count)` with `select_polygon_biased(rng, count)` in the per-polygon mutation paths of `mutate.wgsl`. About 10 lines changed, no new parameters or buffers.

---

## Proposal 9: Differential Mutation (Population-Guided Steps)

### Problem

Each chain mutates independently using random perturbations. The population collectively holds information about the fitness landscape -- the *direction* from one chain's solution to another encodes gradient-like information -- but this is currently unused. Crossover recombines polygon sets but does not use inter-chain *differences* as mutation vectors.

### Solution

For a small fraction of offspring, apply **differential mutation** inspired by DE:

```
offspring_vertex[p] = parent_vertex[p] + F * (chain_A_vertex[p] - chain_B_vertex[p])
```

where `chain_A` and `chain_B` are tournament-selected chains. This is applied at the individual polygon level (picking a polygon index that exists in all three drawings). The difference vector automatically adapts: large when population is diverse, small when converged.

### GPU Implementation

The mutate shader already has read access to `chain_states`. Add this as an alternative mutation path:

```wgsl
if rand_f32(&rng) < params.differential_prob {
    let a = tournament_select(&rng, chain_id, chain_count);
    let b = rand_u32(&rng, chain_count);
    let pi = rand_u32(&rng, min(count, chain_states[a].polygon_count, chain_states[b].polygon_count));
    // Apply vertex and color differences scaled by F=0.5
    ...
}
```

### Expected Impact

Moderate. DE-style mutations are most effective with diverse populations (many chains). With the default 4 chains, the benefit is limited. With 32-128 chains, this could provide 10-20% convergence improvement by leveraging implicit gradient information.

### Implementation Complexity

Medium. New mutation path in `mutate.wgsl`, one new parameter (`differential_prob`). No buffer layout changes.

---

## Proposal 10: Adaptive Operator Selection (Reward-Based Probability Tuning)

### Problem

The mutation probability distribution is static (set by the user or auto-tuner). But the optimal distribution changes dramatically during evolution:
- **Early** (fitness < 70%): `add_polygon` and `change_color` have high acceptance rates.
- **Middle** (70-90%): `move_point` and `offset_polygon` dominate.
- **Late** (> 90%): Only `micro_adjust` (vertex and color) produces improvements.

The existing adaptive mutation scale adjusts step *size* but not which *type* of mutation to apply.

### Solution

Track per-mutation-type acceptance rates and dynamically adjust probabilities. The simplest GPU-friendly implementation:

**Three-category tracking**: structural (add/remove/reorder/merge/clone), spatial (move_point/offset/scale/rotate/medium_move), and fine-tuning (micro_adjust/color/brightness/saturation).

Store in each chain's state: `accept_structural: u16`, `accept_spatial: u16`, `accept_finetune: u16`, `total_accepts: u16` (8 bytes total, fits in existing padding or rng_state[1..3]).

In the select shader, on acceptance, increment the appropriate category counter. In the mutate shader, read the counters and bias the roulette wheel:

```wgsl
let structural_weight = base_structural * (1.0 + reward_factor * (structural_rate / avg_rate - 1.0));
let spatial_weight = base_spatial * (1.0 + reward_factor * (spatial_rate / avg_rate - 1.0));
let finetune_weight = base_finetune * (1.0 + reward_factor * (finetune_rate / avg_rate - 1.0));
```

Counter decay: multiply all counters by 0.99 every K iterations (exponential moving average), preventing stale information from dominating.

### Expected Impact

Moderate-High. AOS is well-studied in the EA literature with typical 15-30% convergence improvements. The main benefit is that the algorithm automatically transitions from structural exploration to fine-tuning without user intervention or scheduled parameter changes.

### Implementation Complexity

Medium. Requires 8 bytes per chain in state, bookkeeping in select shader, probability adjustment in mutate shader. The tricky part is getting the reward factor and decay rate right.

---

## Proposal 11: Color-Aware New Polygon Initialization

### Problem

New polygons (from `add_polygon` and `clone_jitter`) are initialized with uniformly random colors. In a 24-bit RGB space, the probability of randomly selecting a color close to the optimal one is vanishingly small. Most new polygons require many subsequent `change_color` mutations to reach a useful color, during which the polygon's shape may also drift.

### Solution

When adding a new polygon, sample the reference image at the polygon's centroid to initialize its color:

```wgsl
// In add_polygon mutation:
let centroid = (new_v0 + new_v1 + new_v2) / 3.0;
let ref_pixel = textureLoad(reference_image, vec2<i32>(
    i32(centroid.x * f32(params.image_width)),
    i32(centroid.y * f32(params.image_height))
), 0);
let new_color = vec4<f32>(ref_pixel.xyz, clamp(rand_f32(rng), params.min_alpha_norm, params.max_alpha_norm));
```

This requires the mutate shader to have read access to the reference texture, which is currently only bound in rasterize_error. Adding it as a binding in the mutate bind group is trivial (texture is read-only).

### Expected Impact

Moderate-High. Reference-sampled colors are a much better starting point than random colors, especially at high fitness where the rendered image is already close to the reference. The acceptance rate of `add_polygon` mutations should improve 2-5x because the new polygon immediately contributes useful color information instead of introducing error.

### Implementation Complexity

Low. Add one texture binding to the mutate bind group layout. Add 4 lines in the add_polygon mutation path. No new buffers.

---

## Priority Ranking

Ranked by expected impact / implementation complexity ratio:

| Priority | Proposal | Impact | Complexity | Notes |
|---|---|---|---|---|
| 1 | **Biased polygon selection (P8)** | Moderate | Very Low | Nearly free, ~10 lines, no downside |
| 2 | **Color-aware init (P11)** | Moderate-High | Low | One binding + 4 lines, immediate benefit |
| 3 | **Cauchy mutations (P7)** | Moderate | Low | One function, well-studied, no new params |
| 4 | **SA acceptance on stagnation (P5)** | Moderate-High | Low | ~20 lines in select.wgsl, safe (stagnation-gated) |
| 5 | **Stratified offspring (P1)** | High | Low-Medium | Deterministic diversity across lambda |
| 6 | **Error-guided placement (P4)** | High | Low-Medium | 4x4 grid version is achievable |
| 7 | **Polygon splitting (P6)** | High | Medium | Principled complexity increase |
| 8 | **Coarse-to-fine screening (P3)** | Very High | Medium-High | 8x rasterize speedup but needs 2-phase dispatch |
| 9 | **Adaptive operator selection (P10)** | Moderate-High | Medium | Automatic structural-to-finetune transition |
| 10 | **NES gradient / multi-accept (P2)** | High (simplified) | Medium | Sequential accept is the practical version |
| 11 | **Differential mutation (P9)** | Moderate | Medium | Mainly useful at high chain counts |

---

## Quick Wins (Implementable in < 1 Day Each)

1. **Biased polygon selection (P8)**: Replace `rand_u32(rng, count)` with `max(u,v)` distribution. ~10 lines changed.
2. **Cauchy mutations for move_point/offset_polygon (P7)**: Add one function, change 4-6 call sites in `mutate.wgsl`.
3. **SA acceptance on stagnation (P5)**: Add temperature computation and stagnation-gated probabilistic acceptance. ~20 lines in `select.wgsl`.
4. **Color-aware polygon initialization (P11)**: Add reference texture to mutate bind group, sample at centroid. ~15 lines total.

## Medium Projects (1-3 Days Each)

5. **Stratified offspring mutation (P1)**: Deterministic mutation assignment for first N offspring slots. New branch in `mutate.wgsl` + configurable stratification map.
6. **Error-guided placement (P4, 4x4 grid)**: 16 u32s per chain, regional error accumulation in rasterize_error, biased placement in mutate.
7. **Polygon splitting on stagnation (P6)**: New mutation function, centroid-based triangle subdivision, z-order-preserving insertion.

## Larger Projects (3+ Days Each)

8. **Coarse-to-fine two-phase evaluation (P3)**: Second rasterize dispatch at 1/4 resolution, top-K compact step, full-resolution re-evaluation.
9. **Adaptive operator selection (P10)**: Per-chain category acceptance tracking, dynamic probability reweighting, decay logic.
10. **Sequential multi-accept (P2, simplified)**: Second-best tracking in select shader, mini re-evaluation against updated parent.
