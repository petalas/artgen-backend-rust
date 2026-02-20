# Evolutionary Algorithm Recommendations

**Expert focus**: Migration policy, crossover operators, adaptive parameters, selection strategies

---

## Executive Summary

The current (1+1)-ES with global-best migration is fundamentally flawed: migration replaces ALL chains with the global best every 50 iterations, completely destroying population diversity. The algorithm is effectively a single hill-climber that restarts every 50 iterations. Fixing this alone could yield a multi-fold improvement.

---

## 1. HIGHEST PRIORITY: Fix Catastrophic Migration Policy

**Current problem**: `migrate_main` in `select.wgsl` replaces *every* chain with the global best. After each migration event, all 64 chains are identical copies exploring from the same point with only RNG differences.

**Fix**: Ring-topology partial migration — each chain only considers its right neighbor:

```wgsl
fn migrate_main(...) {
    let neighbor_id = (chain_id + 1u) % chain_count;
    let neighbor_fitness = bitcast<f32>(chain_states[neighbor_id].fitness_bits);
    let my_fitness = bitcast<f32>(chain_states[chain_id].fitness_bits);

    if neighbor_fitness > my_fitness + 0.001 * abs(my_fitness) {
        // adopt neighbor's drawing, keep own RNG
        ...
    }
}
```

Also increase `GPU_MIGRATION_INTERVAL` from 50 to 200-500. Current 50-iteration interval fires every 5 batches — far too aggressive.

---

## 2. Crossover via Segment Grafting

Full-genome crossover is problematic because polygon ordering matters (alpha compositing). But **segment graft crossover** works well:

Take a contiguous subsequence of polygons from one parent and splice it into the other at a random position, replacing some of its polygons.

- Dispatch: `chain_count/2` workgroups (pairs of adjacent chains)
- Crossover probability: ~5% per iteration
- Segment length: 10-30% of donor's polygons
- Preserves internal polygon ordering within the grafted segment

**Why it works**: Local regions tend to be controlled by groups of nearby, overlapping polygons. A contiguous segment represents a coherent "feature" (shadow, edge, color region). Transplanting preserves its structure.

---

## 3. Adaptive Step Sizes (1/5th Rule)

Fixed mutation step sizes (`move_point_max_delta`, `micro_adjust_delta`) are always wrong for one phase of evolution. Early: large steps find good regions. Late: only tiny refinements help.

**Implementation**: Add `success_count`, `attempt_count`, and `step_scale` fields to `GpuDrawingState` (reuse the `_pad` fields).

In `select.wgsl`, every 50 attempts:
- If success rate > 20%: `scale *= 1.2` (steps too small)
- If success rate < 20%: `scale *= 0.82` (steps too big)
- Clamp to [0.01, 10.0]

In `mutate.wgsl`, multiply all deltas by `step_scale`.

This automatically transitions between exploration and exploitation.

---

## 4. Heterogeneous Mutation Strategies

Instead of all chains using identical mutation probabilities, assign different profiles based on `chain_id`:

| Strategy (chain_id % 4) | Focus |
|---|---|
| 0: "Explorer" | 3x add_polygon_prob, 0.3x micro_adjust |
| 1: "Refiner" | 3x micro_adjust, 0.1x add_polygon |
| 2: "Colorist" | 5x change_color, 0.2x move_point |
| 3: "Default" | Use params as-is (control group) |

Zero new buffers needed — derive strategy from `chain_id` in the shader.

---

## 5. Stochastic Acceptance (Simulated Annealing Hybrid)

Current selection is strictly elitist (`fitness > current_fitness`), which traps in local optima.

Add Metropolis acceptance for per-chain parent replacement (NOT for global best tracking):

```wgsl
let accept = fitness > current_fitness
    || rand_f32(&rng) < exp((fitness - current_fitness) * temperature);
```

Temperature starts at 50.0, decays by 0.9999 per iteration. Add `temperature` to `GpuParams` (replace `_params_pad`).

---

## 6. Error-Guided Polygon Placement

New polygons are placed randomly, but most of the image area already has low error.

Compute a per-chain "worst tile" (highest error 8x8 region) via `atomicMax` in the error reduction pass. In `mutate.wgsl`, 30% of the time bias new polygon placement toward the high-error region:

```wgsl
origin_x = clamp(target_x + rand_f32_range(&rng, -0.1, 0.1), 0.0, 1.0);
origin_y = clamp(target_y + rand_f32_range(&rng, -0.1, 0.1), 0.0, 1.0);
```

Buffer cost: one extra `u32` per chain (packed tile coordinates).

---

## 7. Fix Complexity Penalty

Current penalty `fitness *= (1 - per_point_multiplier * num_points)` gets *larger* as fitness improves — a perverse incentive that punishes good solutions more.

Replace with lexicographic selection:

```wgsl
let accept = (fitness > current_fitness)
    || (abs(fitness - current_fitness) < 0.0001
        && working_states[chain_id].polygon_count < chain_states[chain_id].polygon_count);
```

Primary: minimize error. Secondary: minimize polygon count (only as tiebreaker).

---

## 8. NOT Recommended for This Problem

- **EDA**: 12,000-dimensional with strong epistasis — impractical
- **Differential Evolution**: Vector-difference concept doesn't apply to polygon lists
- **Novelty Search / MAP-Elites**: Image approximation isn't deceptive, QD would waste evaluations

---

## Priority Summary

| Priority | Recommendation | Impact | Effort |
|----------|---------------|--------|--------|
| 1 | Fix migration (ring topology) | Very High | Low |
| 2 | Adaptive step sizes (1/5th rule) | High | Medium |
| 3 | Heterogeneous mutation strategies | Medium-High | Low |
| 4 | Segment graft crossover | Medium-High | Medium |
| 5 | Error-guided polygon placement | Medium | Medium |
| 6 | Stochastic acceptance (SA hybrid) | Medium | Low |
| 7 | Fix complexity penalty (lexicographic) | Medium-Low | Low |
| 8 | Delta fitness / partial re-evaluation | Very High | High |
