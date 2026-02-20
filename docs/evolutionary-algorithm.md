# Evolutionary Algorithm Analysis

Deep analysis of the GPU evolution pipeline in `artgen-backend-rust`, with concrete recommendations for what to build next. Everything listed in "Already Completed Optimizations" is excluded.

---

## 1. Current Architecture Summary

The system runs **512 independent (1+1) evolution strategy chains** on the GPU. Each chain:

1. Copies its current-best drawing to a working buffer
2. Applies probabilistic mutations in a `while !is_dirty` loop (up to 1000 attempts)
3. Rasterizes the mutated drawing (compute shader, 8x8 workgroups per pixel)
4. Computes per-pixel L1 error vs. reference image, reduces via shared memory
5. Selects: if mutant fitness > current-best fitness, accept it
6. Periodically migrates via ring topology (every 50 iterations)

All 512 chains execute in lockstep across 50 iterations per GPU submission. There is **no crossover**, **no adaptive mutation**, and **no diversity maintenance** beyond the ring migration.

### Current VRAM Usage

At 384x384 with 512 chains:
- Chain states: 512 * 48,032 = ~23.5 MB
- Working states: ~23.5 MB
- Render targets: 512 * 384 * 384 * 4 = ~302 MB
- Reference image: 384 * 384 * 4 = ~0.6 MB
- Error accumulators: 512 * 4 = 2 KB
- **Total: ~350 MB** out of 32 GB available (~1.1% utilization)

---

## 2. Mutation Operator Analysis

### 2.1 Current Operators (mutate.wgsl)

| Operator | Probability | Scope | Effect |
|----------|------------|-------|--------|
| Add polygon | 1/50 = 2% | Drawing | New random triangle near random origin |
| Remove polygon | 1/1500 = 0.07% | Drawing | Remove random polygon, shift array |
| Reorder | 1/500 = 0.2% | Drawing | Swap two polygons (z-order) |
| Offset polygon | 1/500 = 0.2% | Per-polygon | Translate all vertices by same delta |
| Change color (per channel) | 1/750 = 0.13% | Per-polygon | Replace single channel with random value |
| Micro-adjust color | 1/100 = 1% | Per-polygon | +/-1 per channel (1/255) |
| Lighten | 1/750 = 0.13% | Per-polygon | All RGB channels +1/255 |
| Darken | 1/750 = 0.13% | Per-polygon | All RGB channels -1/255 |
| Move point | 1/500 = 0.2% | Per-vertex | Random within +/-0.1 of current |
| Micro-adjust point | 1/100 = 1% | Per-vertex | Random within +/-0.01 of current |

### 2.2 Probability Balance Issues

**Micro-adjust dominates.** With ~150 polygons (3 vertices each = 450 vertices), micro-adjust fires on ~4.5 vertices per mutation pass and ~1.5 color channels. Meanwhile, structural mutations (add/remove/reorder) fire on average 0.02 + 0.0007 + 0.002 = ~0.023 times. The mutation mix is ~99% fine-tuning, ~1% structural. This is appropriate for late-stage optimization but too conservative for early exploration.

**Add vs. Remove asymmetry is extreme.** Add probability (1/50) is 30x higher than Remove (1/1500). The polygon count will monotonically increase toward the cap. Once at cap, the system loses the ability to simplify and restructure. This is a major source of stagnation.

### 2.3 Missing Mutation Operators

The following operators exist in the CPU path but are absent from the GPU shader:

1. **Remove point** (`REMOVE_POINT_PROBABILITY = 1/500`) -- present in CPU `polygon.mutate()` but not in `mutate.wgsl`. Since GPU polygons are always triangles (3 vertices), this operator doesn't apply in the current representation. However, if the representation ever changes to support N-gons, this would need adding.

The following operators would be valuable additions:

1. **Scale polygon** -- uniformly scale a triangle around its centroid. Currently the only way to resize a polygon is to move individual vertices, which requires 3 lucky mutations to achieve what one scale mutation could do.

2. **Rotate polygon** -- rotate a triangle around its centroid. Same argument as scale: coordinated vertex movement is extremely unlikely through independent point mutations.

3. **Clone polygon** -- duplicate an existing polygon with slight perturbation. Much more useful than adding a random polygon, since existing polygons have already been optimized to useful positions and colors.

4. **Replace polygon** -- simultaneously remove one polygon and add another. Avoids the 2-step penalty of remove-then-add where the intermediate state is always worse.

5. **Swap adjacent polygons** -- instead of swapping two random polygons (which is usually destructive), swap only adjacent polygons in z-order. Most z-order improvements are local.

6. **Color-from-reference** -- sample the reference image color at the polygon's centroid and use it as the polygon's color. This is a strongly guided mutation that would dramatically improve early convergence.

---

## 3. Crossover Design for GPU

Crossover is the single biggest missing piece. With 512 chains all doing (1+1) ES, the system is running 512 independent hill-climbers that only interact through ring migration (wholesale copying). True crossover could combine beneficial traits from different chains.

### 3.1 Why Crossover is Hard for Polygon Drawings

Polygon-based drawings have two properties that make naive crossover destructive:

1. **Order-dependent rendering**: Polygons are composited front-to-back with alpha blending. Swapping polygon subsets between drawings changes the z-ordering context, invalidating the fitness of both subsets.

2. **Co-adaptation**: A polygon's optimal color depends on what's behind it (other polygons + background). Polygons co-adapt in groups. Breaking these groups apart is usually worse than either parent.

### 3.2 Recommended: Region-Based Crossover

**Core idea**: Divide the image into spatial regions. For each region, pick one parent. Take all polygons whose centroid falls in that region from the chosen parent.

**Implementation in WGSL**:
```
fn crossover(parent_a: chain_id, parent_b: chain_id, child: chain_id, rng):
    // Choose a random split line (vertical or horizontal)
    let split_pos = rand_f32(rng)  // 0.0 to 1.0
    let split_vertical = rand_f32(rng) > 0.5

    var child_count = 0u
    // From parent A: take polygons on one side of the split
    for each polygon p in parent_a:
        centroid = (p.v0 + p.v1 + p.v2) / 3.0
        let coord = select(centroid.x, centroid.y, split_vertical)
        if coord < split_pos:
            child.polygons[child_count++] = p

    // From parent B: take polygons on the other side
    for each polygon p in parent_b:
        centroid = (p.v0 + p.v1 + p.v2) / 3.0
        let coord = select(centroid.x, centroid.y, split_vertical)
        if coord >= split_pos:
            child.polygons[child_count++] = p
```

**Why this works**: Polygons that cover the same image region tend to be co-adapted. By keeping spatially coherent groups together, we preserve the co-adaptation structure while recombining different parts of the image.

**GPU considerations**: This requires a new compute pass between select and migrate, dispatched as `(chain_count/2, 1, 1)` workgroups. Pairs of chains produce one child each. The child replaces the worse parent.

**Buffer requirements**: No additional buffers needed -- the working_states buffer can be repurposed as temporary storage for the child, then copied back to chain_states.

### 3.3 Alternative: Uniform Polygon Crossover with Fitness Sorting

Pick polygons from two parents by index. For each index position, randomly choose parent A or parent B's polygon. This is simpler but more destructive because z-ordering is broken.

**Mitigation**: Sort the child's polygons by area (large first) after crossover. This approximates the natural ordering (big background shapes first, details last) and partially recovers z-order coherence.

### 3.4 Alternative: Headless Chicken Crossover

Cross a good solution with a random solution. The "random parent" is just a newly generated random drawing. This tests whether the good parent's polygons are individually valuable (the ones that survive in the child are likely individually beneficial).

**GPU implementation**: Trivial. Generate a random drawing in the mutate shader, then do region-based crossover between the chain's best and the random drawing. This is essentially a structured "restart from random with memory" operator.

### 3.5 Crossover Frequency and Integration

Crossover should be infrequent relative to mutation -- perhaps every 200-500 iterations, interleaved with the migration pass. Suggested implementation:

- Add a `crossover_interval` parameter to `GpuParams` (e.g., 200)
- Add a new `crossover.wgsl` compute shader
- Add a new compute pipeline and bind group in `pipeline.rs`
- Dispatch conditionally in `mod.rs`, similar to how migration is dispatched
- Pair chains for crossover: `(2i, 2i+1)` or random pairs via RNG

---

## 4. Population Structure Improvements

### 4.1 Current State: Flat Ring

All 512 chains are arranged in a single ring. Migration copies the right neighbor's drawing if it's fitter. This means:

- A good solution propagates around the ring at rate 1 chain per migration event
- Full propagation takes ~512 * 50 = 25,600 iterations
- Until propagation completes, most chains are working on inferior solutions

### 4.2 Recommended: Hierarchical Island Model

Partition the 512 chains into **islands** (e.g., 32 islands of 16 chains each). Within each island, use the existing ring migration. Between islands, add a second migration event at a slower rate.

**Implementation**:
```
// In select.wgsl, add inter_island_migrate_main entry point:

@compute @workgroup_size(1)
fn inter_island_migrate_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let chain_id = gid.x;
    let island_size = 16u;  // from params
    let island_id = chain_id / island_size;
    let local_id = chain_id % island_size;

    // Only island "leader" (local_id == 0) participates
    if local_id != 0u { return; }

    // Ring of islands: compare with next island's leader
    let neighbor_island = (island_id + 1u) % (chain_count / island_size);
    let neighbor_chain = neighbor_island * island_size;

    // Same logic as migrate_main but between island leaders
    ...
}
```

**Benefits**: Islands can explore different regions of the search space independently. Inter-island migration shares the best discoveries without killing diversity within other islands.

**Parameters to add to GpuParams**:
- `island_size: u32` (e.g., 16)
- `inter_island_migration_interval: u32` (e.g., 500 -- 10x slower than intra-island)

### 4.3 Topology Alternatives

Beyond hierarchical islands, other topologies worth considering:

- **2D Torus**: Arrange chains in a 2D grid (e.g., 16x32). Each chain migrates from up/down/left/right neighbors. Provides richer connectivity than a ring without full mixing.
- **Random sparse graph**: Each chain has 2-3 random neighbors. Gives small-world properties (fast propagation of good solutions, but still maintains local diversity).

The 2D torus is particularly GPU-friendly since neighbor computation is just modular arithmetic.

---

## 5. Diversity Maintenance

### 5.1 The Convergence Problem

With (1+1) ES + ring migration, all 512 chains will eventually converge to the same local optimum. Once converged, the system is just 512 copies of the same hill-climber, which is no better than 1 chain.

### 5.2 Recommended: Fitness Sharing / Niching

Introduce a diversity bonus or penalty based on how similar a chain's drawing is to its neighbors.

**Lightweight approach -- phenotypic distance**: Compare rendered images rather than genotypes. After the error_reduce pass, compute the L1 distance between neighboring chains' render targets. Chains that are very similar to their neighbors get a small fitness penalty.

**Implementation complexity**: This requires an additional compute pass that reads `render_targets` for neighboring chains. The per-pixel comparison is embarrassingly parallel (same dispatch as error_reduce). The challenge is that `render_targets` for chain N and chain N+1 are in different z-slices of the dispatch, so you'd need to read from both in the same shader.

**Simpler alternative -- genotypic distance**: Compare polygon counts and average vertex positions between neighboring chains. This is cheaper (only reads chain_states, not render_targets) and can be computed in the select pass. Chains within a small Hamming distance of their neighbors get a diversity bonus to their error threshold (accept slightly worse mutations to escape local optima).

### 5.3 Recommended: Stagnation Detection + Random Restart

Track per-chain stagnation: count iterations since last improvement. If a chain hasn't improved in N iterations (e.g., 5000), reinitialize it with a random drawing.

**Implementation**: Add a `stagnation_counter: u32` field to `DrawingState` (requires struct layout change -- currently `_pad0` and `_pad1` could be repurposed). In `select_main`:

```wgsl
if fitness > current_fitness {
    // Accept
    chain_states[chain_id].stagnation_counter = 0u;
    ...
} else {
    chain_states[chain_id].stagnation_counter += 1u;
    if chain_states[chain_id].stagnation_counter > params.stagnation_limit {
        // Random restart: generate new random drawing
        chain_states[chain_id] = random_drawing(rng);
        chain_states[chain_id].stagnation_counter = 0u;
    }
}
```

This is straightforward and low-risk. The `_pad0` field in `DrawingState` can be repurposed as `stagnation_counter` with no layout change (it's already a u32 at offset 8). The `_pad1` could hold a `last_improvement_fitness` for tracking improvement rate.

---

## 6. Adaptive Mutation Rates (Self-Adaptation)

### 6.1 The Problem with Fixed Rates

The current mutation probabilities are fixed constants. Early in evolution, when fitness is low, large structural mutations (add/remove polygon) are most valuable. Late in evolution, when fitness is high, only micro-adjustments help. But the probabilities never change.

### 6.2 Recommended: 1/5th Rule Adaptation

The classic (1+1) ES self-adaptation rule: if more than 1/5 of mutations are accepted, increase mutation strength. If fewer than 1/5 are accepted, decrease it.

**Per-chain implementation**: Add `success_count: u32` and `trial_count: u32` to chain state (repurpose padding or extend the struct). Every N iterations (e.g., 100), compute the acceptance rate and scale mutation deltas:

```
acceptance_rate = success_count / trial_count
if acceptance_rate > 0.2:
    mutation_scale *= 1.1   // explore more
else:
    mutation_scale *= 0.9   // exploit more
```

The `mutation_scale` multiplies `move_point_max_delta`, `offset_polygon_magnitude`, and `micro_adjust_delta`. It does NOT affect structural mutations (add/remove polygon).

**GPU-friendly variant**: Instead of per-chain adaptation (which requires extra state), use **global adaptation** based on the overall acceptance rate across all chains. This can be computed in the select pass using `atomicAdd` on a shared counter:

```wgsl
// In select_main, after the fitness comparison:
if fitness > current_fitness {
    atomicAdd(&control.accept_count, 1u);
}
// CPU reads accept_count after each batch, adjusts params.move_point_max_delta etc.
```

This is the simplest approach: the CPU adjusts `GpuParams` between batches based on the acceptance ratio from the previous batch.

### 6.3 Alternative: Fitness-Proportional Mutation Strength

Scale mutation strength inversely with fitness:

```
scale = 1.0 - (fitness / 100.0)  // fitness ranges 0-100
effective_delta = base_delta * (0.1 + 0.9 * scale)
```

At fitness 0 (bad), full mutation strength. At fitness 90 (good), 19% of base strength. This requires no additional state -- just modify the mutate shader to read chain fitness and scale accordingly.

---

## 7. Multi-Objective Considerations

### 7.1 Current Approach: Weighted Sum

The fitness function is `100 * (1 - error/max_error) - fitness * per_point_multiplier * num_points`. This is a weighted sum of image fidelity and complexity (polygon count). The weight `PER_POINT_MULTIPLIER = 1/5,000,000` is extremely small -- at fitness 90 with 1000 polygons * 3 points, the penalty is `90 * 0.0000002 * 3000 = 0.054`. This is negligible.

### 7.2 Issue: Complexity Penalty is Too Weak

The penalty is so small that the system will always prefer adding polygons. There is no meaningful pressure to simplify. If the goal is to evolve compact, elegant approximations (fewer polygons), the penalty needs to be 10-100x stronger. If the goal is maximum fidelity regardless of complexity, the penalty should be removed entirely.

### 7.3 Recommended: Pareto-Based Multi-Objective

Instead of a weighted sum, maintain a Pareto front of non-dominated solutions across the chains. A solution A dominates B if A has both better fidelity AND fewer polygons. Non-dominated solutions are preserved; dominated ones are candidates for replacement.

**Simplified version for (1+1) ES**: In the select pass, accept a mutant if it:
- Has better fidelity AND same or fewer polygons, OR
- Has same fidelity AND fewer polygons, OR
- Has sufficiently better fidelity to justify additional polygons (e.g., fidelity improvement > threshold * polygon_count_increase)

This doesn't require full NSGA-II machinery but still provides meaningful complexity pressure.

---

## 8. Population Sizing and VRAM Utilization

### 8.1 Current Utilization: ~1.1% of 32 GB

With ~350 MB used out of 32,768 MB, there is massive headroom.

### 8.2 Scaling Options

| Chains | VRAM (est.) | Notes |
|--------|-------------|-------|
| 512 | 350 MB | Current |
| 2,048 | 1.4 GB | 4x more chains, still <5% VRAM |
| 8,192 | 5.5 GB | 16x more chains, ~17% VRAM |
| 16,384 | 11 GB | 32x more chains, ~34% VRAM |

**However**, more chains does not linearly improve convergence speed. The bottleneck is the quality of individual mutations, not the number of parallel attempts. Diminishing returns set in quickly.

### 8.3 Better Use of VRAM: Increase Image Resolution

The current max resolution is 384x384. Increasing to 512x512 or 768x768 would:
- Give finer detail in the reference image
- Improve fitness signal for small polygons
- Cost more VRAM per chain (render targets scale as W*H*4)

At 512x512 with 512 chains: render targets = 512 * 512 * 512 * 4 = 512 MB. Total ~560 MB -- still well within budget.

At 768x768 with 512 chains: render targets = 512 * 768 * 768 * 4 = 1.2 GB. Total ~1.25 GB -- still only 4% of VRAM.

### 8.4 Better Use of VRAM: Multiple Populations

Instead of one population of 512 chains, run multiple independent populations with different hyperparameters (mutation rates, alpha ranges, polygon limits). This is effectively hyperparameter search for free.

**Implementation**: Partition chains into groups. Each group gets its own `GpuParams` slice. The mutate shader indexes into a params array by `chain_id / group_size` instead of using a single uniform.

---

## 9. Concrete Prioritized Recommendations

Ordered by expected impact vs. implementation effort:

### Tier 1: High Impact, Low Effort

1. **Stagnation detection + random restart** (repurpose `_pad0` as stagnation counter in DrawingState, add check in select_main). Prevents dead chains. ~50 lines of WGSL, ~10 lines of Rust.

2. **Fitness-proportional mutation strength** (scale deltas by `1 - fitness/100` in mutate.wgsl). Zero state changes, just multiply deltas by a scale factor read from `chain_states[chain_id].fitness_bits`. ~15 lines of WGSL.

3. **Clone polygon mutation** (duplicate existing polygon with perturbation instead of random new one). Much higher acceptance rate than random polygon insertion. ~30 lines of WGSL in the add-polygon section of mutate.wgsl.

### Tier 2: High Impact, Medium Effort

4. **Region-based crossover** (new `crossover.wgsl` shader, new compute pipeline, conditional dispatch). The biggest algorithmic improvement possible. ~200 lines of WGSL, ~80 lines of Rust.

5. **Hierarchical island model** (partition 512 chains into 32 islands of 16, add inter-island migration). Better diversity preservation. ~50 lines of WGSL for the new entry point, ~20 lines of Rust for the new params.

6. **CPU-side adaptive mutation via acceptance ratio** (count accepts in select pass via atomicAdd, CPU reads and adjusts GpuParams between batches). ~15 lines of WGSL, ~30 lines of Rust.

### Tier 3: Medium Impact, Higher Effort

7. **Scale/rotate polygon mutations** (compute centroid, apply affine transform to vertices). Better search operators. ~60 lines of WGSL.

8. **Color-from-reference mutation** (sample reference image at polygon centroid, use as color). Requires binding reference_image in the mutate shader. ~40 lines of WGSL, ~30 lines of Rust for bind group changes.

9. **Increase image resolution** (raise MAX_IMAGE_WIDTH/HEIGHT to 512 or 768). Mostly settings changes, but need to verify dispatch limits and test performance.

10. **Multi-population with different hyperparameters** (params array instead of single uniform). Implicit hyperparameter search. ~50 lines of WGSL, ~80 lines of Rust.

---

## 10. Implementation Notes

### Struct Layout Constraints

`GpuDrawingState` is 48,032 bytes with layout:
- Offset 0: `polygon_count` (u32)
- Offset 4: `fitness_bits` (u32)
- Offset 8: `_pad0` (u32) -- **can repurpose as `stagnation_counter`**
- Offset 12: `_pad1` (u32) -- **can repurpose as `mutation_scale_bits` (bitcast f32)**
- Offset 16: `rng_state` (vec4<u32>)
- Offset 32: `polygons` (array<Polygon, 1000>)

Repurposing `_pad0` and `_pad1` requires updating both `buffers.rs` (GpuDrawingState struct) and all WGSL shaders that reference the struct. The layout and total size remain unchanged.

### Dispatch Limits

wgpu `max_compute_workgroups_per_dimension = 65535`. Current dispatches:
- Mutate: (512, 1, 1) -- 512 chains
- Rasterize: (48, 48, 512) at 384x384 -- all within limits
- At 768x768: (96, 96, 512) -- still within limits
- At 1024x1024: (128, 128, 512) -- still within limits
- Chain count of 65535 would hit the z-dimension limit for rasterize

### Buffer Size Limits

The render_targets buffer (chains * W * H * 4) is the binding bottleneck. At the WSL2 dozen adapter's `max_storage_buffer_binding_size` of ~1 GB:
- 384x384: max ~1,700 chains
- 512x512: max ~950 chains
- 768x768: max ~425 chains

This means increasing resolution trades off against chain count. The current auto-capping logic in `pipeline.rs` handles this correctly.
