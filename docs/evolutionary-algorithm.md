# Evolutionary Algorithm Analysis & Improvement Proposals

This document analyzes the current evolutionary strategy in artgen-backend-rust and proposes algorithmic improvements to speed up convergence and improve final image quality. It focuses exclusively on **new ideas** not already implemented.

## Current Architecture Summary

The system uses a **(1+lambda)-ES** running entirely on the GPU via wgpu compute shaders:

- **Population**: Up to 1024 chains, each holding a drawing (up to 1000 triangles).
- **Per iteration**: Each chain spawns lambda offspring (1-64, power-of-2). Offspring are independently mutated, rasterized, error-scored, and the best is compared against the parent.
- **Selection**: Strictly elitist -- offspring replaces parent only if fitness improves (with 50% neutral acceptance for plateau traversal).
- **Migration**: Two-level island model with intra-island ring migration (every 50 iterations) and inter-island global ring migration (every 500 iterations).
- **Mutation**: Probabilistic, with 13+ mutation operators (add/remove polygon, reorder, scale, rotate, adjacent swap, offset polygon, move point, micro-adjust point, change color, micro-adjust color, lighten, darken). Single-mutation mode uses weighted roulette-wheel selection. Adaptive mutation scale (1.2x on accept, pow(0.99, 1/lambda) on reject, burst to 1.5 on stagnation).
- **Fitness**: L2 error per pixel (Euclidean RGB distance) summed across all pixels, converted to percentage, with a small penalty proportional to the number of vertices (PER_POINT_MULTIPLIER = 1/5000000).

---

## Proposal 1: Cauchy-Distributed Mutation Steps (Heavy-Tailed Exploration)

### Problem

All spatial mutations currently use **uniform random** perturbations within a fixed delta range. Uniform distributions are thin-tailed: they never produce a perturbation larger than `move_point_max_delta * mutation_scale`. This causes the algorithm to get stuck in basins of attraction where the optimal move is farther than the maximum delta allows.

### Solution

Replace the uniform distribution for vertex and offset mutations with a **Cauchy distribution** (or a mixture of Cauchy and Gaussian). The Cauchy distribution has much heavier tails than Gaussian, enabling occasional large jumps that escape local optima while still producing small perturbations most of the time.

On the GPU this can be implemented without transcendental functions beyond `tan()`:

```wgsl
// Cauchy sample: median=0, scale=gamma
fn cauchy_sample(rng: ptr<function, vec4<u32>>, gamma: f32) -> f32 {
    let u = rand_f32(rng) - 0.5;  // uniform in (-0.5, 0.5)
    return gamma * tan(3.14159265 * u);
}
```

Apply this selectively: use Cauchy for `move_point` and `offset_polygon` (the exploratory mutations), keep uniform for `micro_adjust` (the exploitative mutations). This creates a natural explore/exploit split without additional parameters.

### Expected Impact

Literature on evolutionary strategies consistently shows 10-30% faster convergence on multimodal landscapes when switching from Gaussian to Cauchy mutations (known as "Fast Evolutionary Programming"). The benefit is especially pronounced in later stages when the algorithm needs to escape plateaus.

### Implementation Complexity

Low. Replace 4-6 `rand_f32_range` calls in `mutate.wgsl` with Cauchy samples clamped to [0,1]. One new GPU function, no buffer changes, no new parameters needed.

---

## Proposal 2: Differential Mutation (Leverage Population Diversity)

### Problem

Each chain mutates its parent independently. The population of chains carries implicit information about the fitness landscape -- the *direction* from one chain's solution to another encodes gradient-like information -- but this is currently unused. Crossover partially addresses this but only by recombining polygon sets, not by using inter-chain *differences* as mutation vectors.

### Solution

Implement **differential mutation** inspired by Differential Evolution (DE). For a fraction of offspring, instead of applying random perturbations, compute a mutation as:

```
offspring_vertex = parent_vertex + F * (chain_A_vertex - chain_B_vertex)
```

where `chain_A` and `chain_B` are two randomly selected chains from the same island, and `F` is a scaling factor (typically 0.5-0.8).

This requires that the two donor chains have similar polygon structure (roughly the same polygon count), so it should be applied at the individual polygon level rather than the whole drawing:

1. Select a random polygon index `p` that exists in both the parent and the donor chains.
2. For each vertex of polygon `p`: `offspring_vertex[p] = parent_vertex[p] + F * (donor_A_vertex[p] - donor_B_vertex[p])`.
3. Apply the same logic to the polygon's color.

### GPU Implementation

The mutate shader already reads `chain_states` (all chains). Add a new mutation path activated with a configurable probability (e.g., 5-10%):

```wgsl
// Differential mutation path
if rand_f32(&rng) < differential_prob && island_size >= 3u {
    let a = island_start + rand_u32(&rng, island_size);
    let b = island_start + rand_u32(&rng, island_size);
    // ... apply vertex/color differences scaled by F
}
```

### Expected Impact

DE-style mutations are especially effective when the population has converged to different local optima. The difference vectors automatically adapt their magnitude to the current population spread -- large when diverse, small when converged. This provides a form of implicit step-size adaptation that complements the existing explicit adaptive mutation scale.

### Implementation Complexity

Medium. New mutation path in `mutate.wgsl`, one new parameter (`differential_prob` + scale factor `F`). No buffer layout changes since the mutate shader already has read access to `chain_states`.

---

## Proposal 3: Multi-Resolution Coarse-to-Fine Evolution

### Problem

The rasterize+error shader is the bottleneck (typically 60-80% of GPU time per iteration). Every offspring is rasterized at full resolution (up to 512x512 = 262144 pixels) and every pixel contributes equally to the error. Early in evolution, when drawings are rough approximations, this full-resolution evaluation is wasteful -- coarse pixel-level differences dominate and fine detail is irrelevant.

### Solution

Implement a **progressive resolution schedule** that starts evaluations at low resolution and increases as fitness improves:

| Fitness Range | Resolution Scale | Pixels Evaluated | Speedup |
|---|---|---|---|
| < 70% | 1/4 (128x128) | 16K | ~16x |
| 70-85% | 1/2 (256x256) | 65K | ~4x |
| 85-92% | 3/4 (384x384) | 147K | ~1.8x |
| > 92% | Full (512x512) | 262K | 1x |

Implementation approach: Rather than creating multiple reference textures, modify the rasterize+error shader to **stride** over pixels. At 1/4 resolution, each workgroup evaluates every 4th pixel in both x and y, then multiplies the accumulated error by 16. The reference texture is sampled at the strided coordinates.

```wgsl
// In rasterize_error.wgsl
let stride = params.resolution_stride; // 1, 2, or 4
let px = gid.x * stride;
let py = gid.y * stride;
// ... rest of rasterization unchanged ...
// Thread 0 adds: error * stride * stride
atomicAdd(&error_accumulators[chain_id], shared_errors[0] * stride * stride);
```

The dispatch dimensions would shrink proportionally: `(W/16/stride, H/16/stride, K*lambda)`.

### Expected Impact

Massive throughput improvement in early evolution. At 1/4 resolution, iteration throughput increases ~16x, which means the first 70% of fitness is reached dramatically faster. The total wall-clock time to reach 90%+ fitness could be reduced by 40-60%, since most iterations are spent in the early/middle phases.

### Implementation Complexity

Medium. Requires a new `resolution_stride` parameter in `GpuParams`, changes to dispatch dimensions in `mod.rs`, and stride logic in `rasterize_error.wgsl`. The CPU side needs fitness-based resolution transitions. No buffer layout changes.

### Caveats

Resolution transitions cause fitness discontinuities (the error at 1/4 resolution is an approximation). To handle this: when transitioning to a higher resolution, re-evaluate the current best at the new resolution before continuing. This is a single extra evaluation and is negligible.

---

## Proposal 4: Correlated Mutation Tracking (1/5th Rule on Steroids)

### Problem

The current adaptive mutation scale uses fixed multipliers: 1.2x on accept, pow(0.99, 1/lambda) on reject. These are static ratios that don't account for *which mutation types* are productive. In practice, different mutation types have wildly different acceptance rates at different stages of evolution:

- Early: `add_polygon` and `change_color` are productive (large structural changes).
- Middle: `move_point` and `offset_polygon` dominate (positioning).
- Late: `micro_adjust` and `lighten/darken` are the only productive mutations (fine color tuning).

A single global scale factor cannot capture this.

### Solution

Track **per-mutation-type acceptance rates** using exponential moving averages, and use these to dynamically reweight mutation probabilities. This is a form of **adaptive operator selection** (AOS).

For the GPU implementation, store per-chain counters in the `GpuDrawingState` header:

```
// Add to GpuDrawingState (fits in the existing padding or extend slightly)
mutation_type_used: u32,     // which mutation type the offspring used (0-12)
accept_counts: [u32; 4],    // packed: 4 mutation categories x 8 bits each
attempt_counts: [u32; 4],   // same packing
```

In the select shader, when an offspring is accepted, increment the acceptance counter for the mutation type that was used. Every N iterations, normalize the counters to derive new mutation probabilities:

```
new_prob[i] = base_prob[i] * (1 + reward_factor * (accept_rate[i] / avg_accept_rate - 1))
```

This naturally shifts probability mass toward productive mutations without eliminating any type entirely.

### Simplified Alternative

If per-type tracking is too complex for the GPU, a simpler version: track acceptance rates for just **two categories** (structural: add/remove/reorder vs. parametric: move/color/micro-adjust) and shift probability mass between them. This requires only 2 additional u32s in the drawing state.

### Expected Impact

Literature on adaptive operator selection shows 15-30% improvement in convergence speed. The benefit compounds over long runs because the algorithm automatically transitions from structural to fine-tuning mutations without user intervention.

### Implementation Complexity

Medium-High for full per-type tracking. Low for the two-category simplified version.

---

## Proposal 5: Polygon Insertion Guided by Error Map

### Problem

New polygons are currently placed at **random positions** with random colors. Most random placements produce negligible fitness improvement because they land in areas that are already well-approximated. The probability of a random polygon landing in a high-error region is proportional to the area of high-error regions, which shrinks as fitness improves.

### Solution

Maintain an **error heatmap** (downsampled, e.g., 32x32 grid) and use it to bias new polygon placement toward high-error regions. This is a form of **fitness landscape-informed mutation**.

Implementation options:

**Option A: GPU-side error sampling** (preferred). After the rasterize+error pass, add a lightweight reduction pass that produces a 32x32 error grid (each cell = sum of pixel errors in that region). Store this in a small buffer. In the mutate shader, when adding a new polygon, sample the error grid to choose the origin point:

```wgsl
// Weighted random selection from 32x32 error grid
fn sample_error_grid(rng: ptr<function, vec4<u32>>) -> vec2<f32> {
    // Prefix-sum sampling or rejection sampling over 1024 cells
    let total_error = error_grid[1023]; // assume prefix-sum stored
    let target = rand_f32(rng) * f32(total_error);
    // Binary search for cell... return center of selected cell
}
```

**Option B: Simple quadrant biasing** (simpler). Divide the image into a 4x4 grid. In the select shader, after computing the error, also compute the error for each quadrant. Store the highest-error quadrant index in the chain state. New polygons are placed in the highest-error quadrant with 50% probability, random otherwise.

### Expected Impact

This directly addresses one of the biggest inefficiencies in the algorithm: random polygon placement. Empirically, error-guided placement can improve the acceptance rate of `add_polygon` mutations by 3-10x, which translates to faster convergence especially in the 50-80% fitness range where polygon count is still growing.

### Implementation Complexity

Option A: High (new buffer, new reduction pass, prefix-sum on GPU).
Option B: Low-Medium (4 additional u32s per chain for quadrant errors, bias logic in mutate shader).

---

## Proposal 6: Simulated Annealing-Style Acceptance with Temperature Schedule

### Problem

The current selection is strictly elitist with 50% neutral acceptance. This means the algorithm can only traverse fitness plateaus (through neutral moves) but can never accept a slightly worse solution to escape a local optimum. In rugged fitness landscapes with many local optima (which polygon art definitely has -- reordering polygons can create many equivalent-fitness configurations), this leads to premature convergence.

### Solution

Add a **temperature-based acceptance probability** for slightly inferior offspring, inspired by simulated annealing. The acceptance probability for a fitness decrease of `delta_f` would be:

```
p_accept = exp(-delta_f / temperature)
```

The temperature follows a schedule:
- Start high (e.g., 0.5% of max fitness range = 0.5)
- Decay geometrically per chain based on the chain's stagnation counter
- Reset to medium on migration events (diversity injection)
- Floor at a very small value (e.g., 0.001) to maintain permanent slight exploration

In the select shader:

```wgsl
if fitness > current_fitness {
    accept();
} else {
    let delta = current_fitness - fitness;
    let temperature = compute_temperature(chain_states[chain_id].stagnation_counter);
    // Use offspring's RNG bit for deterministic comparison
    let threshold = exp(-delta / temperature);
    let rand_val = f32(working_states[best_offspring_id].rng_state.x) / 4294967296.0;
    if rand_val < threshold {
        accept();  // Accept worse solution with SA probability
    }
}
```

### Expected Impact

SA-style acceptance is one of the most well-studied techniques for escaping local optima. The key insight for this application is that polygon art has **many near-equivalent optima** (slightly different orderings, slightly different triangle decompositions of the same visual region). Allowing temporary fitness decreases lets the algorithm tunnel through these barriers. Expected improvement: 5-15% better final fitness at convergence, especially for complex images.

### Implementation Complexity

Low. Changes are confined to the select shader. No new buffers, no new passes. One new parameter (initial temperature) or compute it from the current fitness range automatically.

### Risk

If temperature is too high, the algorithm degenerates into random walk. The stagnation-triggered temperature (only allow SA acceptance when stagnation_counter > threshold) mitigates this risk entirely.

---

## Proposal 7: Polygon Sorting by Depth/Area for Rendering Order

### Problem

Polygon ordering significantly affects the rendered image (because of alpha blending). The current mutations that affect ordering (reorder, adjacent swap) are random -- they don't encode any heuristic about what orderings tend to be good. In practice, larger polygons should be painted first (background) and smaller polygons later (detail), similar to a painter's algorithm.

### Solution

Periodically (e.g., every 1000 iterations or on stagnation), apply a **soft sort** mutation that partially reorders polygons by area (largest first). "Soft" means: don't fully sort, but do a limited number of comparison-swap passes (like a few iterations of bubble sort) biased toward area-descending order. This preserves most of the current ordering while nudging it toward a better structure.

Implementation in the mutate shader:

```wgsl
// Soft-sort mutation: 3-5 bubble-sort passes biased by area
fn soft_sort_by_area(oid: u32, count: u32, rng: ptr<function, vec4<u32>>) {
    for (var pass = 0u; pass < 3u; pass++) {
        for (var i = 0u; i < count - 1u; i++) {
            let area_i = triangle_area(working_states[oid].polygons[i]);
            let area_next = triangle_area(working_states[oid].polygons[i + 1u]);
            // Swap if smaller polygon is before larger (with 70% probability to keep stochastic)
            if area_i < area_next && rand_f32(rng) < 0.7 {
                let tmp = working_states[oid].polygons[i];
                working_states[oid].polygons[i] = working_states[oid].polygons[i + 1u];
                working_states[oid].polygons[i + 1u] = tmp;
            }
        }
    }
}
```

### Expected Impact

Moderate. Ordering matters most for images with large background regions and fine foreground detail. The impact is image-dependent but averages 2-5% fitness improvement when the polygon count is high (> 200).

### Implementation Complexity

Low-Medium. New mutation function in `mutate.wgsl`. The `triangle_area` computation is already available (cross product, used in degenerate culling). One new probability parameter.

---

## Proposal 8: Warm Restart with Polygon Splitting

### Problem

When the algorithm stagnates with a low polygon count, the stagnation burst (mutation_scale to 1.5) can destabilize the existing solution without fundamentally changing the representational capacity. The algorithm needs **more polygons** in the right places, not larger random perturbations of existing polygons.

### Solution

When stagnation is detected at high fitness (> 85%) and polygon count is below the maximum, perform a **polygon split** operation: select a polygon, split it into two or more smaller triangles that together cover the same area, then allow evolution to differentiate them:

1. Pick a polygon with above-average area.
2. Compute its centroid.
3. Split into 3 sub-triangles by connecting each edge to the centroid.
4. Give each sub-triangle the same color as the original.
5. Now evolution can evolve the sub-triangles' colors independently, capturing finer detail.

This is analogous to **mesh refinement** in finite element methods -- you refine where you need more resolution.

### Targeted Splitting Using Error

Combine with Proposal 5 (error-guided placement): only split polygons that overlap with high-error regions.

### Expected Impact

High. This directly addresses the fundamental limitation of the representation -- the number and placement of polygons. Splitting introduces new degrees of freedom exactly where they can help most. Expected improvement: 3-8% better final fitness, especially visible as sharper edges and better color gradients in the final image.

### Implementation Complexity

Medium. New mutation type in `mutate.wgsl`. Requires centroid computation (already available) and polygon insertion logic (already exists for `add_polygon`).

---

## Proposal 9: Gradient Approximation via Finite Differences

### Problem

Evolution is a zeroth-order optimization method -- it only uses fitness values, not gradients. Each evaluation is expensive (full rasterization), so using fitness information more efficiently would be valuable.

### Solution

For the **micro-adjust** mutations (which perturb a single vertex or color channel by a small delta), approximate the fitness gradient by comparing the fitness of +delta and -delta perturbations. Then move in the direction of improvement:

```
gradient_estimate = (fitness(x + delta) - fitness(x - delta)) / (2 * delta)
step = learning_rate * gradient_estimate
```

This can be done within the existing lambda framework: use two of the lambda offspring slots as paired +delta/-delta probes on the same parameter, then use the gradient to produce a third offspring.

### Simplified Version: Coordinate Descent

Even simpler: for each micro-adjust mutation, try both +1 and -1. Accept whichever is better. This doubles the micro-adjust evaluation cost but makes every micro-adjust move optimal. Since micro-adjust mutations dominate in late evolution, this could significantly speed up the endgame.

### Expected Impact

High for the endgame (> 90% fitness). The endgame is dominated by pixel-level color refinement where the fitness landscape is locally smooth and gradient information is very useful. Expected improvement: 20-40% faster convergence in the > 90% fitness range.

### Implementation Complexity

High for the general case. Medium for the simplified coordinate-descent version (allocate pairs of offspring slots for +/- probes).

---

## Proposal 10: Fitness-Proportional Polygon Mutation Targeting

### Problem

All polygons within a drawing are equally likely to be mutated (`rand_u32(rng, count)` selects uniformly). But not all polygons contribute equally to the fitness. Large polygons covering high-error regions have more potential for improvement than tiny polygons in well-approximated areas.

### Solution

Weight polygon selection by a proxy for "potential improvement": the polygon's area times its overlap with high-error regions. A simpler approximation: weight by **area alone** (larger polygons affect more pixels, so mutating them has more impact).

Even simpler: weight by **inverse rendering order** (polygons rendered later are "on top" and their mutations are more likely visible, since they aren't occluded by later polygons).

```wgsl
// Weighted polygon selection: bias toward later (top) polygons
fn select_polygon_biased(rng: ptr<function, vec4<u32>>, count: u32) -> u32 {
    // Triangular distribution: favors higher indices
    let u = rand_f32(rng);
    let v = rand_f32(rng);
    let idx = u32(max(u, v) * f32(count));
    return min(idx, count - 1u);
}
```

This `max(u, v)` trick produces a distribution that linearly increases with index -- later polygons are selected more often.

### Expected Impact

Moderate. In the current algorithm, mutations to occluded polygons are wasted evaluations (they change the genotype but not the phenotype). Biasing toward visible (later) polygons reduces wasted mutations by an estimated 10-20%.

### Implementation Complexity

Very Low. Replace `rand_u32(rng, count)` with `select_polygon_biased(rng, count)` in the mutation functions. No new buffers or parameters.

---

## Priority Ranking

Ranked by expected impact / implementation complexity ratio:

| Priority | Proposal | Impact | Complexity | Notes |
|---|---|---|---|---|
| 1 | **Multi-Resolution (P3)** | Very High | Medium | Biggest throughput gain, addresses the #1 bottleneck |
| 2 | **Cauchy Mutations (P1)** | High | Low | Well-studied, minimal code change, no downside |
| 3 | **Polygon Mutation Targeting (P10)** | Moderate | Very Low | Nearly free improvement |
| 4 | **SA-Style Acceptance (P6)** | High | Low | Well-studied, confined to select shader |
| 5 | **Error-Guided Placement (P5B)** | High | Low-Medium | Option B (quadrant) is simple and effective |
| 6 | **Polygon Splitting (P8)** | High | Medium | Addresses fundamental representational bottleneck |
| 7 | **Correlated Mutation (P4)** | Moderate-High | Medium-High | Simplified 2-category version is achievable |
| 8 | **Differential Mutation (P2)** | Moderate | Medium | Novel combination with ES |
| 9 | **Polygon Sorting (P7)** | Moderate | Low-Medium | Image-dependent benefit |
| 10 | **Gradient Approx (P9)** | High (endgame) | Medium-High | Only useful at > 90% fitness |

---

## Quick Wins (Implementable in < 1 Day Each)

1. **Cauchy mutations for move_point/offset_polygon** (P1): Add one function, change 4-6 call sites in `mutate.wgsl`.
2. **Biased polygon selection** (P10): Replace uniform selection with `max(u,v)` distribution. ~10 lines changed.
3. **SA acceptance in select shader** (P6): Add temperature computation and probabilistic acceptance. ~20 lines in `select.wgsl`.

## Medium Projects (1-3 Days Each)

4. **Multi-resolution evaluation** (P3): New parameter, stride logic in rasterize shader, dispatch size changes.
5. **Error-guided polygon placement (quadrant version)** (P5B): 4 extra u32s per chain, quadrant error reduction in select shader, biased placement in mutate shader.
6. **Polygon splitting on stagnation** (P8): New mutation function, centroid-based triangle subdivision.
