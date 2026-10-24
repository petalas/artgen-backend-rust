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

**Mutation group**:
The number of logical offspring assigned to one mutation workgroup. It is distinct
from the number of physical lanes in that workgroup.

**Mature drawing restart**:
A new seeded search starting from an existing drawing, without its previous search
history.

**Shader variant**:
An alternative program for an evolution stage that preserves the search's fixed
evolutionary rules.

**Pending batch**:
Submitted evaluations whose completion and any resulting drawing have not yet
been collected.
