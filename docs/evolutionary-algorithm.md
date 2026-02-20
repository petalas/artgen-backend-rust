# Evolutionary Algorithm Analysis & Improvement Proposals

Analysis of the GPU evolution pipeline in `artgen-backend-rust`. All proposals below are for changes **not yet implemented** -- see the project README for already-completed optimizations (ring migration, island model, crossover, tournament selection, L1 error, etc.).

---

## 1. Current Architecture Summary

The GPU pipeline runs **K independent (1+1) evolution chains** (default 128, max 512) across 4 compute passes per iteration:

1. **Mutate** (`src/shaders/mutate.wgsl`): Copy chain state to working state, apply probabilistic mutations (or crossover 10% of the time). One thread per chain.
2. **Rasterize + Error** (`src/shaders/rasterize_error.wgsl`): Fused rasterization and L1 error computation. 16x16 workgroups with tiled polygon prefetch into shared memory. Error reduced via binary tree within each workgroup, then `atomicAdd` to per-chain accumulator.
3. **Select** (`src/shaders/select.wgsl`): Compare candidate fitness to current best. Accept if strictly better (greedy). Track global best via `atomicMax`.
4. **Migrate** (`src/shaders/select.wgsl`): Periodic ring migration within islands (intra) or across all chains (inter). Replace self with neighbor if neighbor is fitter.

Fitness = `100 * (1 - L1_error / max_total_error) - complexity_penalty`, where the complexity penalty is proportional to polygon count (via `PER_POINT_MULTIPLIER = 1/5000000`).

---

## 2. Mutation Operator Analysis

### 2.1 Current Operator Inventory

From `mutate.wgsl` (lines 332-516), the mutation operators are:

| Operator | Default Probability | Scope | Effect |
|----------|-------------------|-------|--------|
| Add polygon | 1/50 (0.02) | Drawing | Insert random triangle at random position |
| Remove polygon | 1/1500 (0.00067) | Drawing | Remove random polygon |
| Reorder (swap) | 1/500 (0.002) | Drawing | Swap two polygons' z-order |
| Offset polygon | 1/500 (0.002) | Per-polygon | Translate all vertices by same delta |
| Change color | 1/750 (0.00133) | Per-channel | Replace one channel with random value |
| Micro-adjust color | 1/100 (0.01) | Per-channel | +/- 1/255 on one channel |
| Lighten | 1/750 (0.00133) | Per-polygon | All RGB channels +1/255 |
| Darken | 1/750 (0.00133) | Per-polygon | All RGB channels -1/255 |
| Move point | 1/500 (0.002) | Per-vertex | Random offset within `move_point_max_delta` (0.1) |
| Micro-adjust point | 1/100 (0.01) | Per-vertex | Random offset within `micro_adjust_delta` (0.01) |

### 2.2 Missing Mutation Operators

**a) Scale/Resize Polygon**

There is no operator to uniformly scale a polygon around its centroid. The only way to change polygon size is moving individual vertices. A scale mutation would preserve shape while exploring size, which is especially useful for fine-tuning coverage of a region.

```wgsl
// Proposed: scale polygon around centroid
if rand_f32(&rng) < params.scale_polygon_prob {
    let c = centroid(poly);
    let scale = rand_f32_range(&rng, 0.8, 1.2); // +/- 20%
    poly.v0 = clamp(c + (poly.v0 - c) * scale, vec2(0.0), vec2(1.0));
    poly.v1 = clamp(c + (poly.v1 - c) * scale, vec2(0.0), vec2(1.0));
    poly.v2 = clamp(c + (poly.v2 - c) * scale, vec2(0.0), vec2(1.0));
    is_dirty = true;
}
```

**b) Rotate Polygon**

No rotation operator exists. Rotation around the centroid would allow exploring orientation without changing size or position.

```wgsl
// Proposed: rotate polygon around centroid
if rand_f32(&rng) < params.rotate_polygon_prob {
    let c = centroid(poly);
    let angle = rand_f32_range(&rng, -0.3, 0.3); // ~+/- 17 degrees
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    // Rotate each vertex around centroid
    let d0 = poly.v0 - c;
    poly.v0 = clamp(c + vec2(d0.x * cos_a - d0.y * sin_a, d0.x * sin_a + d0.y * cos_a), vec2(0.0), vec2(1.0));
    // ... same for v1, v2
    is_dirty = true;
}
```

**c) Duplicate Polygon (with slight mutation)**

Currently, adding a polygon creates a brand new random triangle. A "duplicate nearby polygon" operator would exploit existing good coverage by cloning a polygon with slight perturbation -- this is a form of constructive exploitation.

**d) Gaussian (Non-Uniform) Perturbations**

All vertex and color perturbations use uniform distributions. Gaussian (or Cauchy) perturbation distributions would produce mostly small changes with occasional large jumps, which is well-established as superior for continuous optimization. This could be approximated with the Box-Muller transform or even a simple triangular distribution (sum of two uniform samples).

```wgsl
// Approximate Gaussian via triangular distribution (sum of 2 uniform)
fn rand_gaussian_approx(rng: ptr<function, vec4<u32>>, sigma: f32) -> f32 {
    return (rand_f32(rng) + rand_f32(rng) - 1.0) * sigma;
}
```

### 2.3 Mutation Probability Imbalance

The current probabilities have a significant structural issue: **per-polygon mutations compound with polygon count**. With N polygons, the expected number of mutations per iteration is:

- Drawing-level: `add=0.02 + remove=0.00067 + reorder=0.002` = ~0.023
- Per-polygon (per polygon): `offset=0.002 + 4*color=0.00533 + 4*micro_color=0.04 + 3*move=0.006 + 3*micro_move=0.03` = ~0.083
- **Total per-polygon contribution: 0.083 * N**

At N=500 polygons, there are ~41.5 expected per-polygon mutations per iteration, but only ~0.023 drawing-level mutations. This means:

1. **Almost every iteration applies many micro-adjustments simultaneously.** This creates a high-dimensional random walk that is much harder to evaluate -- any single good change is drowned out by many neutral/bad changes.
2. **The "mutate until dirty" loop (line 328) is misleading** -- with N=500, `is_dirty` becomes true on virtually the first attempt, so the loop always runs exactly once. The loop is only meaningful for very small drawings.

**Recommendation**: Consider a "single-mutation" mode where each iteration applies exactly one mutation operator (chosen by weighted roulette). This is the standard approach in (1+1)-ES for combinatorial/structured problems. It allows the selection pressure to act on individual changes rather than batches of changes. The GPU throughput is high enough that evaluating many single-mutation candidates per second is feasible.

```wgsl
// Single-mutation mode: pick one operator via roulette wheel
let r = rand_f32(&rng);
var cumulative = 0.0;
cumulative += params.add_polygon_prob;
if r < cumulative { /* add polygon */ }
cumulative += params.remove_polygon_prob;
if r < cumulative { /* remove polygon */ }
// ... etc, pick a random polygon index for per-polygon ops
```

---

## 3. Selection Pressure & Diversity

### 3.1 Greedy Selection is Too Strict

The current selection (`select.wgsl` line 111) is strictly greedy: `if fitness > current_fitness`. This means:
- **No neutral moves are accepted.** In combinatorial optimization, accepting moves of equal fitness is critical for escaping plateaus. The fitness landscape for polygon art has vast plateaus (many configurations yield identical L1 error after rounding).
- **No simulated annealing.** There is no mechanism to accept slightly worse solutions to escape local optima.

**Recommendation**: Accept neutral moves (change `>` to `>=`, or accept with 50% probability when equal). Optionally, implement a simple Metropolis criterion where `P(accept) = exp(-delta_fitness / temperature)` for slightly worse solutions, with temperature decaying over iterations.

```wgsl
// Accept improvements always, neutral with 50%, worse with exponential decay
if fitness > current_fitness {
    // Accept
} else if fitness == current_fitness && rand_f32(&rng) < 0.5 {
    // Accept neutral
} else {
    let delta = current_fitness - fitness;
    let temperature = max(0.001, 1.0 / (1.0 + f32(params.iteration_number) * 0.0001));
    if rand_f32(&rng) < exp(-delta / temperature) {
        // Accept worse (simulated annealing)
    }
}
```

Note: Implementing `rand_f32` in the select shader would require passing the RNG state into the select pass. Currently the RNG lives in `DrawingState` and is only used in `mutate.wgsl`.

### 3.2 Diversity Loss in Migration

The current migration strategy (`migrate_from` in `select.wgsl` line 139-160) is purely elitist: if your neighbor is better, you **completely adopt their drawing**. This means:
- After enough migration rounds, all chains within an island converge to the island's best solution.
- Diversity is only maintained by independent mutation streams (different RNG seeds) applied to identical drawings.

**Recommendation**: Probabilistic migration acceptance. Instead of always adopting a better neighbor, accept with probability proportional to fitness difference. Or adopt only a fraction of the neighbor's polygons (partial migration).

```wgsl
// Probabilistic migration: accept better neighbor with p < 1.0
let acceptance_prob = 0.3; // only 30% chance to adopt even if neighbor is better
if neighbor_fitness > my_fitness && rand_f32(&rng) < acceptance_prob {
    // migrate
}
```

### 3.3 Population Diversity Monitoring

There is no mechanism to detect or respond to diversity collapse. All islands can converge to similar solutions without any detection. A simple diversity metric would be to track the variance of fitness across chains within an island -- when variance drops below a threshold, inject random mutations or reset the worst chains.

---

## 4. Adaptive Mutation Rates

### 4.1 The 1/5th Rule (Missing)

The classic (1+1)-ES theory prescribes the **1/5th success rule**: if more than 1/5 of mutations are accepted, increase step size (mutation magnitude); if fewer, decrease it. The current system has completely static mutation parameters.

The infrastructure for this partially exists -- `iteration_number` is passed to the GPU but never used in the mutation shader. This could drive adaptive rates:

```wgsl
// Adaptive move_point_max_delta: decrease over time as fitness improves
let adaptive_delta = params.move_point_max_delta * max(0.01, 1.0 / (1.0 + f32(params.iteration_number) * 0.00001));
```

### 4.2 Per-Chain Self-Adaptive Parameters (CMA-ES Inspired)

A more powerful approach: store per-chain mutation parameters (step sizes) in the `DrawingState` and evolve them alongside the drawing. Chains with well-tuned step sizes will produce better offspring, get selected more often via tournament, and spread their step sizes via migration.

This would require adding fields to `DrawingState` (e.g., `mutation_sigma: f32` for vertex perturbation magnitude) and letting successful mutations reinforce their parameter values. This is a well-proven technique from Evolution Strategies.

### 4.3 Fitness-Proportional Mutation Intensity

Low-fitness chains should mutate more aggressively (they are far from optimal and need exploration), while high-fitness chains should apply fine-grained mutations (they are near optimal and need exploitation). Currently all chains use identical mutation parameters regardless of fitness.

```wgsl
// Scale mutation magnitudes by fitness rank within island
let my_fitness = bitcast<f32>(chain_states[chain_id].fitness_bits);
let island_best = /* max fitness in island */;
let relative_quality = my_fitness / max(island_best, 0.001);
let exploration_factor = 2.0 - relative_quality; // 1.0 for best, ~2.0 for worst
// Apply exploration_factor to move_point_max_delta, offset_polygon_magnitude, etc.
```

---

## 5. Fitness Landscape & Error Metric

### 5.1 L1 vs L2 vs Perceptual

L1 (sum of absolute differences per channel) is currently used on GPU, while CPU uses L2 (Euclidean distance across channels). L1 is faster (no sqrt) but treats all errors equally. Consider:

- **Weighted channel error**: Human perception is more sensitive to green than red or blue. Using weights like `(0.299, 0.587, 0.114)` (ITU-R BT.601 luma) would prioritize perceptually important differences. This is a trivial change in `rasterize_error.wgsl` line 174:

```wgsl
// Perceptually-weighted L1 error
let dr = abs(ri - refr) * 0.299;
let dg = abs(gi - refg) * 0.587;
let db = abs(bi - refb) * 0.114;
pixel_error = u32((dr + dg + db) * 3.0); // scale back up to preserve dynamic range
```

- **SSIM-like structural similarity**: L1/L2 metrics don't capture edge alignment or structural features. A simplified structural metric (comparing local mean/variance in small patches) could be computed as a secondary objective but would be significantly more complex on GPU.

### 5.2 Complexity Penalty Tuning

The current complexity penalty (`PER_POINT_MULTIPLIER = 1/5000000` in `src/settings.rs` line 18) is extremely small. With 1000 triangles (3000 points) and fitness ~90.0:

`penalty = 90.0 * (1/5000000) * 3000 = 0.054`

This 0.054 penalty out of 90.0 fitness is negligible (~0.06%). In practice, the algorithm will always prefer more polygons because the error reduction from an extra polygon almost always exceeds this tiny penalty. Consider whether the penalty should be nonlinear (quadratic in polygon count) or significantly larger if the goal is to find parsimonious solutions.

---

## 6. Convergence Speed Improvements

### 6.1 Warm-Start / Seeded Initialization

Currently all chains start from the same initial drawing (`drawing_to_gpu` in `src/gpu_evolver/buffers.rs` line 155). This means all chains begin identical and only diverge through random mutation.

**Recommendation**: Initialize a fraction of chains with random perturbations of the initial drawing (different polygon counts, shuffled z-orders, varied alpha ranges). This gives the population immediate diversity to explore from.

```rust
// In GpuEvolver::new -- perturb some chains on init
let initial_states: Vec<GpuDrawingState> = (0..max_chains)
    .map(|i| {
        let seed = ...;
        let mut state = drawing_to_gpu(initial_drawing, seed);
        // Perturb 25% of chains: randomize some polygon properties
        if i % 4 != 0 {
            // Shuffle polygon order, drop random polygons, etc.
        }
        state
    })
    .collect();
```

### 6.2 Stagnation Detection & Restart

There is no mechanism to detect when evolution has stalled and take corrective action. The system can spend millions of evaluations making no progress.

**Proposal**: Track the iteration number of the last improvement per chain. If a chain has not improved in N iterations (e.g., 10,000), apply a "mega-mutation" (multiple aggressive mutations) or reset it to a perturbation of the island's best.

This could be implemented by adding a `last_improvement_iter: u32` field to `DrawingState` and checking it in the mutate shader:

```wgsl
let stagnant = params.iteration_number - chain_states[chain_id].last_improvement_iter > 10000u;
if stagnant {
    // Apply extra-aggressive mutations: larger deltas, higher probabilities
    // Or: reset to a random perturbation of the island's best
}
```

### 6.3 Multi-Mutation Candidates Per Chain

Currently each chain evaluates one candidate per iteration. An alternative is to generate multiple candidates per chain (e.g., 4) using different mutation strategies and select the best. This is the (1+lambda) strategy. On GPU, the rasterize+error pass dominates runtime, so generating 4 mutations per chain (cheap) and evaluating 4x candidates (expensive) may not be worthwhile unless the mutation shader becomes more sophisticated.

A cheaper alternative: generate 2 candidates with different mutation strengths (one conservative, one aggressive) and select the better one.

---

## 7. Polygon Ordering Optimization

### 7.1 Adjacent Swap vs Random Swap

The current reorder operator (`mutate.wgsl` line 384) swaps two randomly chosen polygons. In polygon art, z-order matters primarily for overlapping polygons. Swapping two polygons that are far apart in the z-order and don't overlap has no visual effect -- it's a wasted evaluation.

**Recommendation**: Bias toward adjacent swaps (swap polygon `i` with `i+1`). This is more likely to produce a visible change and allows the algorithm to bubble polygons through the z-order incrementally.

```wgsl
// Adjacent swap with 80% probability, random swap otherwise
if rand_f32(&rng) < 0.8 {
    let i1 = rand_u32(&rng, count - 1u);
    let i2 = i1 + 1u;
    // swap
} else {
    // random swap (existing behavior)
}
```

### 7.2 Move-to-Front / Move-to-Back

Add operators that move a polygon to the front (top of z-order) or back (bottom). This allows large jumps in z-order that the swap operator explores very slowly.

---

## 8. Crossover Improvements

### 8.1 Crossover Operates on Copies, Not Offspring

The current crossover implementation (`mutate.wgsl` lines 278-306) skips the mutation loop entirely when crossover fires. This means crossover offspring are never mutated in the same iteration. Applying a light mutation after crossover (a common practice in genetic algorithms) would help differentiate offspring from parents.

### 8.2 Fitness-Weighted Crossover

The spatial crossover splits space 50/50 between parents. A fitness-weighted split (better parent contributes more polygons) could be more effective.

### 8.3 Crossover Without Tournament Creates Identity Crossover

When `tournament_select` picks the same chain as `chain_id` (possible since the chain is included in its own island), the crossover produces a copy of the parent. This is a wasted evaluation. Add a check:

```wgsl
// Ensure parent_b != chain_id
var parent_b = tournament_select(&rng, chain_id, chain_count);
var attempts = 0u;
while parent_b == chain_id && attempts < 5u {
    parent_b = tournament_select(&rng, chain_id, chain_count);
    attempts++;
}
if parent_b == chain_id {
    // Fall through to mutation instead
}
```

---

## 9. Batch Size & Iteration Count

### 9.1 GPU_ITERATIONS_PER_BATCH = 50

Currently 50 iterations are packed into a single command buffer (`src/gpu_evolver/mod.rs` line 145). This is good for amortizing CPU-GPU submission overhead, but it means:
- Control flags are only checked every 50 iterations (a new global best could exist for 49 iterations before being reported to the CPU).
- Migration only triggers at multiples of 50 that align with `GPU_MIGRATION_INTERVAL`.

This is unlikely to be a performance issue, but it means the CPU feedback loop is coarse-grained.

### 9.2 Workgroup Size for Mutate/Select

The mutate and select shaders use `@workgroup_size(1)` -- each chain gets a single thread. This is fine for the current design (each chain is independent), but it means these passes have very low occupancy on the GPU. The GPU has many more SMs than chains.

If the chain count is increased significantly (e.g., 2048+), the single-thread-per-workgroup model would still work. But if the mutation logic becomes more complex, consider using workgroup parallelism within each chain (e.g., different threads mutate different polygons).

---

## 10. Summary of Prioritized Recommendations

Ordered by expected impact-to-effort ratio:

| Priority | Improvement | Expected Impact | Effort |
|----------|-------------|----------------|--------|
| 1 | Accept neutral moves in selection (`>=` or 50% acceptance) | High -- fixes plateau stagnation | Trivial |
| 2 | Single-mutation mode (one operator per iteration) | High -- cleaner selection signal | Medium |
| 3 | Perceptually-weighted L1 error | Medium -- better perceptual quality | Trivial |
| 4 | Adjacent swap bias for reorder | Medium -- more effective z-order search | Trivial |
| 5 | Gaussian/triangular perturbation distributions | Medium -- better exploration/exploitation | Low |
| 6 | Scale and rotate polygon operators | Medium -- fills operator gaps | Low |
| 7 | Stagnation detection with mega-mutation | Medium -- prevents wasted compute | Medium |
| 8 | Avoid self-crossover (parent_b == chain_id) | Low-Medium -- prevents wasted evaluations | Trivial |
| 9 | Crossover + mutation (mutate after crossover) | Low-Medium -- standard GA practice | Low |
| 10 | Probabilistic migration acceptance | Medium -- preserves diversity | Low |
| 11 | Adaptive mutation rates (1/5th rule or per-chain sigma) | High long-term -- self-tuning | High |
| 12 | Warm-start with diverse initial population | Medium -- faster early exploration | Low |
| 13 | Fitness-proportional mutation intensity | Medium -- better resource allocation | Medium |
