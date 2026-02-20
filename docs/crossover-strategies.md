# Crossover Strategies for Layered Polygon Drawings

## The Problem

This project evolves drawings made of up to 1000 ordered, alpha-blended triangles. Polygon ordering is semantically meaningful: polygon `i+1` is rendered on top of polygon `i`, and alpha blending is order-dependent (non-commutative). Naively mixing polygon lists from two parents destroys the carefully evolved layering relationships and produces garbage images.

The current codebase already has two crossover operators in `src/shaders/mutate.wgsl`:
- **Spatial crossover** (`crossover_spatial_offspring`): splits the image along a random axis/position, takes polygons whose centroid falls on side A from parent A and side B from parent B. Preserves relative ordering within each parent's contribution.
- **Uniform crossover** (`crossover_uniform_offspring`): walks both parents' polygon arrays in lockstep, randomly picking each slot from parent A or B. Drops slots randomly when only one parent has a polygon at that index.

Both currently write to offspring slots in the `working_states` buffer and are selected with probability `crossover_prob` (default 0.1) before the normal mutation path runs.

## Current Representation Summary

```
GpuDrawingState (16032 bytes):
  polygon_count: u32
  fitness_bits: u32
  mutation_scale: f32
  stagnation_counter: u32
  rng_state: vec4<u32>
  polygons: array<Polygon, 1000>

Polygon (16 bytes):
  data: vec4<u32>  // [color_packed, v0_packed, v1_packed, v2_packed]
```

- Coordinates are normalized 0.0-1.0, quantized to u16 (65535 levels)
- Colors are RGBA packed as 4x u8 (pack4x8unorm)
- Alpha range is clamped to ~4%-25% (MIN_ALPHA=10, MAX_ALPHA=65)
- Island model with tournament selection for picking the second parent

---

## Strategy Analysis

### 1. Layer-Range Crossover (Single-Point Crossover on Layer Axis)

**Concept:** Pick a cut point `k` in [0, N). Take polygons `[0..k)` from parent A ("background layers") and polygons `[k..N)` from parent B ("foreground layers"). The offspring has `k + (N_B - k')` polygons where `k'` is the cut in B, capped at `max_polygons`.

**Variant (two-point):** Pick `k1, k2`. Take `[0..k1)` from A, `[k1..k2)` from B, `[k2..N)` from A.

**GPU feasibility:** Trivially parallelizable. Thread 0 picks `k`, writes header, then a simple loop copies two contiguous ranges. Could even use all 64 workgroup threads for parallel polygon copy. Minimal register/shared memory pressure.

**Expected benefit:** Moderate. This preserves the *relative* ordering within each parent's contributed range, which is the key property. Background layers (large, low-alpha washes that set the overall tone) and foreground layers (small, detailed features) are structurally different, so mixing them from two parents that have each found good solutions for their respective ranges is meaningful. This is essentially what GP crossover does with tree slices.

**Risk of degenerate offspring:** Low-moderate. The main risk is a "seam" at the cut point where the two parents' layers don't mesh well. For example, parent A's background might expect certain foreground detail that now comes from parent B. However, the offspring gets evaluated immediately and only replaces the parent if it's better, so bad offspring are just rejected. The low alpha range (10-65/255) also means each polygon has limited impact, reducing seam visibility.

**Implementation complexity:** Very low. ~20 lines of WGSL. Pick random `k`, two memcpy-style loops, set polygon_count.

**Auxiliary data structures:** None.

**Verdict:** Strong candidate. Simple, fast, preserves layer semantics.

### 2. Spatial Crossover (Already Implemented)

**Concept:** Split the image along a random axis at a random position. Take polygons whose centroid falls in region A from parent A, and polygons from region B from parent B. This is already implemented as `crossover_spatial_offspring`.

**GPU feasibility:** Already implemented and running.

**Expected benefit:** Good in theory. Polygons are spatially localized (small triangles with `NEW_POINT_MAX_DISTANCE = 0.03`), so spatial partitioning tends to produce coherent offspring where each region looks like one parent's good solution for that area.

**Risk of degenerate offspring:** Moderate. The current implementation has a subtle ordering issue: it appends *all* of parent A's left-side polygons first, then *all* of parent B's right-side polygons. This means the layer ordering between the two contributions is broken: all B-polygons render on top of all A-polygons regardless of their original layer position. A polygon near the split boundary that was originally a background wash in parent B now sits on top of parent A's detailed foreground features.

**Possible improvement — interleaved spatial crossover:** Instead of concatenating A-left then B-right, walk both parents' arrays simultaneously (like merge sort) and maintain the relative z-ordering. For each layer index `i`, if the polygon from A is on side-left, take it from A; if the polygon from B is on side-right, take it from B; otherwise pick based on a spatial-distance heuristic. This preserves z-order across the spatial split.

**Implementation complexity:** The current version is trivial. The interleaved version is moderate (~40 lines of WGSL) but requires walking both arrays in lockstep.

**Auxiliary data structures:** None.

**Verdict:** Already present, but the ordering issue limits its effectiveness. The interleaved variant is worth implementing.

### 3. Fitness-Guided Per-Polygon Crossover

**Concept:** For each polygon, compute its individual contribution to fitness (how much error it reduces), then merge the "best" polygons from both parents. This requires a per-polygon fitness measure.

**Computing per-polygon fitness:** Render the drawing without polygon `i`, measure error delta. This requires N+1 renders per parent (one baseline, N leave-one-out). With 1000 polygons at 256x256 pixels, that's ~65M pixel operations per parent -- prohibitively expensive to do every crossover event.

**Approximation:** Use polygon area * alpha as a proxy for "visual importance." Larger, more opaque polygons contribute more. Sort the merged polygon list by this importance metric. But this doesn't account for occlusion or redundancy.

**GPU feasibility:** The full per-polygon fitness version is infeasible (1000 extra renders). The approximation is trivially GPU-computable but of questionable value.

**Expected benefit:** Low-moderate with the approximation. The proxy misses the key property (occlusion relationships). Two polygons might both have high area*alpha but be completely redundant (same region, similar color).

**Risk of degenerate offspring:** High with the approximation. Without occlusion awareness, the re-ordering heuristic will frequently produce incorrect layer stacking.

**Implementation complexity:** Approximation: moderate. Full: infeasible.

**Auxiliary data structures:** Temporary sort buffer (1000 entries) if sorting on GPU.

**Verdict:** Not recommended. The approximation doesn't capture what matters, and the exact version is too expensive.

### 4. Uniform Crossover with Re-Ordering (Already Partially Implemented)

**Concept:** Randomly select polygons from both parents (coin flip per position), then sort the result by some heuristic to establish a rendering order.

The current `crossover_uniform_offspring` implementation does the coin-flip selection but does NOT re-order: it walks positions 0..max(N_A, N_B) and picks from A or B at each position, preserving positional correspondence. This means polygon `i` from parent A might end up at position `i` in the offspring even though the surrounding context has changed.

**Possible re-ordering heuristics:**
- Sort by average brightness (dark/large background polygons first, bright/small detail polygons last)
- Sort by area (large first, small last — mimics "coarse to fine")
- Sort by alpha (lower alpha first, higher last)
- Sort by centroid distance from center (or from a canonical scan pattern)

**GPU feasibility:** Selection is trivial. Sorting 1000 elements on GPU is awkward but possible (bitonic sort in ~O(N log^2 N) with ~10 passes for N=1024). Each pass needs a workgroup barrier or separate dispatch. Could also just use a simple insertion sort in a single thread since N=1000 is manageable (O(N^2) but with small constants for <1000 elements in registers/storage).

**Expected benefit:** Depends entirely on the re-ordering heuristic. A good heuristic could produce reasonable offspring. The "area descending" heuristic (large background washes first, small details last) matches how these drawings naturally evolve and could work surprisingly well.

**Risk of degenerate offspring:** Moderate-high. No heuristic perfectly captures the occlusion relationships that evolution discovered. Two large polygons from different parents might have similar area but evolved at very different layer positions for good reason (one is meant to be a base wash, the other a mid-layer correction).

**Implementation complexity:** Selection: trivial (already done). Re-ordering: moderate (need a sort pass). Total: moderate.

**Auxiliary data structures:** Scratch buffer for sort keys if doing external sort. Or use a simple key-value insertion sort in-place on the offspring polygons array.

**Verdict:** Marginal improvement over the existing uniform crossover. The re-ordering heuristic would need empirical tuning. Not a top priority.

### 5. Building-Block Crossover (Polygon Group Swapping)

**Concept:** Identify spatially and chromatically coherent "groups" of polygons (e.g., 5-20 polygons that are nearby and similarly colored, forming a visual feature like an eye or a shadow). Swap entire groups between parents.

**Identifying groups:** Cluster polygons by centroid proximity and color similarity. This is essentially a clustering problem (k-means, DBSCAN, or greedy agglomerative).

**GPU feasibility:** Clustering on GPU is expensive and complex. K-means is doable but requires multiple passes with convergence checks. DBSCAN needs neighbor queries. For ~1000 polygons this is feasible but adds significant shader complexity and auxiliary buffer requirements.

**Expected benefit:** Potentially high if the clustering finds meaningful building blocks. This is the most theoretically sound approach from the EA literature (schema theorem: crossover should exchange building blocks, not arbitrary genes).

**Risk of degenerate offspring:** Low-moderate if the groups maintain their internal ordering. The insertion point for the swapped group matters: it should replace the recipient's corresponding spatial region.

**Implementation complexity:** High. Need clustering infrastructure (centroid/color distance computation, group assignment, variable-size group extraction and insertion).

**Auxiliary data structures:** Cluster assignment buffer (1000 x u32), centroid/color buffers for clustering iterations.

**Verdict:** Theoretically the best approach but implementation cost is very high for uncertain practical benefit. Better suited for a CPU-side implementation where dynamic allocation and complex control flow are natural. Could be explored as a CPU-side pre-processing step that runs rarely (every N thousand iterations) and uploads results.

### 6. Segment Crossover (Contiguous Block Swap)

**Concept:** Pick two aligned "segments" (contiguous ranges) from each parent and swap them. Parent A keeps layers [0..a) and [b..N_A), and receives parent B's layers [a'..b') in between. This is analogous to two-point crossover in GAs.

**Key insight:** Contiguous ranges in this representation correspond to a "depth band" of the image. Layers 0-100 might be the overall color foundation, 100-500 the mid-level structure, 500-1000 the fine detail. Swapping a contiguous band preserves internal coherence within the band.

**GPU feasibility:** Trivial. Three memcpy-style loops. Simpler than building-block crossover, roughly as simple as layer-range crossover.

**Expected benefit:** Similar to layer-range crossover but more flexible. Can exchange mid-level structure while keeping both parents' background and foreground.

**Risk of degenerate offspring:** Similar to layer-range crossover — seams at both splice points.

**Implementation complexity:** Low. ~30 lines of WGSL.

**Auxiliary data structures:** None.

**Verdict:** Strong candidate. A natural generalization of layer-range crossover with minimal extra complexity.

### 7. Depth-Normalized Interleaving

**Concept:** Both parents have polygons at different "depth fractions" (position / total_count). Map each polygon to a normalized depth [0, 1], then merge both parents' polygon lists sorted by this normalized depth. At each position in the merged list, pick from whichever parent's polygon maps to that depth.

**Example:** Parent A has 800 polygons, parent B has 600. A's polygon at index 400 has normalized depth 0.5. B's polygon at index 300 also has normalized depth 0.5. In the merged offspring, these compete for position ~0.5, and a coin flip (or fitness-based decision) picks one.

**GPU feasibility:** Good. This is essentially a merge operation on two sorted arrays (by normalized depth), which can be done in O(N_A + N_B) in a single thread. For 1000+1000 elements, this is fast even sequentially.

**Expected benefit:** This is the theoretically cleanest approach because it *respects the layer semantics* — polygons evolved at similar relative depths in different parents are assumed to serve similar structural roles (background, mid, detail). The merge naturally maintains overall depth ordering.

**Risk of degenerate offspring:** Low. The depth normalization handles different polygon counts gracefully. The main risk is that two parents might have evolved fundamentally different depth structures (e.g., parent A front-loads detail, parent B back-loads it), but this is rare in practice because the rendering physics (alpha blending from back to front) biases all parents toward similar depth structures.

**Implementation complexity:** Low-moderate. ~40 lines of WGSL for the merge loop. Need to handle the merge of two variable-length lists into a capped output.

**Auxiliary data structures:** None (can work directly on the polygon arrays).

**Verdict:** Strong candidate. Theoretically well-motivated, preserves depth semantics, handles variable-length parents naturally.

---

## Recommendation

### Primary: Layer-Range Crossover (Strategy 1)

**Implement first.** It is the simplest to add (~20 lines of shader code), has no auxiliary data structure requirements, preserves the most important invariant (relative layer ordering within each parent's contribution), and matches the well-studied one-point crossover operator from GA theory. It can be added as a third option alongside the existing spatial and uniform crossover modes, controlled by the `spatial_crossover_weight` parameter or a new parameter.

Concrete design:
- Pick random cut point `k` in `[0, max(count_A, count_B))`
- Take `min(k, count_A)` polygons from parent A (indices 0..k)
- Take `max(0, count_B - k)` polygons from parent B (indices k..count_B)
- Cap total at `max_polygons`
- Write to offspring slot

### Secondary: Interleaved Spatial Crossover (Strategy 2, improved variant)

**Implement second, as an upgrade to the existing spatial crossover.** The current spatial crossover breaks z-ordering by concatenating all A-side polygons before all B-side polygons. The interleaved variant walks both parents simultaneously and maintains depth ordering across the spatial split. This is a modest change to the existing `crossover_spatial_offspring` function:

Concrete design:
- Pick random axis and split position (same as current)
- Walk indices `i = 0..max(count_A, count_B)`
- At each `i`: check which parent has a polygon at index `i` and whether its centroid is on its "assigned" side of the split
- If parent A's polygon at `i` is on A-side, emit it; if parent B's polygon at `i` is on B-side, emit it; if both qualify, emit both; if neither, flip a coin
- This preserves z-ordering because we process layers in order

### What NOT to Pursue (for now)

- **Fitness-guided crossover** (Strategy 3): the per-polygon fitness computation is too expensive, and the proxy metrics don't capture occlusion
- **Building-block crossover** (Strategy 5): too complex for GPU implementation with uncertain payoff; better explored as a rare CPU-side operation
- **Uniform crossover re-ordering** (Strategy 4): the current uniform crossover already does the easy part; adding a sort pass adds complexity for marginal improvement

### Testing Approach

Any crossover implementation should be A/B tested by running the GPU evolver with crossover enabled vs disabled (or at different crossover rates) on the same reference image, comparing fitness curves over 100k+ iterations. The crossover probability (currently 0.1) may need tuning — too high and crossover dominates mutation (preventing fine-tuning); too low and crossover never gets a chance to discover useful combinations.
