# Art evolution

## Language

**Chain**:
An evolving parent drawing and its independent search history.

**Offspring**:
A candidate drawing produced from a chain for evaluation against its parent.

**Evaluation**:
One offspring's rasterization and fitness comparison in the evolutionary search.

**Raster tile**:
A rectangular image region evaluated together. Its pixel dimensions do not specify
how many physical GPU lanes execute it.

**Polygon capacity**:
The triangles one GPU drawing state can store, fixed at compile time. It is
storage, not a search setting.

**Search cap**:
The most triangles mutation may give a drawing (`max_polygons`), a runtime
setting no larger than the polygon capacity.

**Engine capacity**:
The largest chain count and offspring per chain an engine is constructed for;
its storage is allocated for that workload.

**Mutation group**:
The number of logical offspring assigned to one mutation workgroup. It is distinct
from the number of physical lanes in that workgroup.

**Mature drawing restart**:
A new seeded search starting from an existing drawing, without its previous search
history.

**Operator stats**:
Per search operator (each mutation branch, crossover), the offspring it produced
and how many of them were accepted as improvements or at equal fitness, counted
since the chains were initialized.

**Step spread**:
A range of step-size multipliers (`step_spread_min` to `step_spread_max`) spread
over a chain's offspring, so that each iteration tries offset and vertex moves
of several sizes at once. Without a spread every offspring uses one size.

**Step rung**:
One multiplier of a step spread, log-spaced between its ends; offspring i of a
chain uses rung i mod 8. Operator stats count, per rung, the step-scaled
offspring and their accepted winners.

**Change record**:
The polygons an evaluated drawing changed relative to the drawing whose pixels
are cached. An unknown record turns off the shortcuts that skip unchanged pixels.

**Shader variant**:
An alternative program for an evolution stage that preserves the search's fixed
evolutionary rules.

**Pending batch**:
Submitted evaluations whose completion and any resulting drawing have not yet
been collected.

**Trial**:
A search run with a fixed starting drawing, evolution parameters, seed and work
budget.

**Qualification**:
Verification that a trial's start, completed work and results meet its fixed
correctness requirements. Qualification does not itself assign a throughput score.
