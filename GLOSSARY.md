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
