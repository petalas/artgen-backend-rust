//! Capacity, storage and dispatch rules shared by allocation and execution.
//! Logical raster tiles and logical mutation groups are independent of lanes.
use crate::{
    mutation_params::MutationParams,
    settings::{GPU_MAX_CHAIN_COUNT, GPU_MAX_LAMBDA, TILE_MAX_POLYS},
};

use super::buffers::GPU_DRAWING_STATE_SIZE;
use super::changes::GPU_CHAIN_CHANGE_SIZE;
use super::operators::OPERATOR_STATS_WORDS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RasterGrid {
    pub x: u32,
    pub y: u32,
}

impl RasterGrid {
    pub fn new(image: [u32; 2], tile: [u32; 2]) -> Self {
        Self {
            x: image[0].div_ceil(tile[0]),
            y: image[1].div_ceil(tile[1]),
        }
    }

    pub fn tiles(self) -> u64 {
        u64::from(self.x) * u64::from(self.y)
    }
}

/// The largest workload an engine is constructed for: chains, and offspring per
/// chain. Storage is allocated for it (then clamped to the device's binding
/// limit, see [`AllocationPlan`]); a batch asking for more panics in
/// [`Workload::for_batch`], and benchmark preparation returns an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EngineCapacity {
    chains: u32,
    lambda: u32,
}

impl EngineCapacity {
    /// Every chain count and λ that `MutationParams::sanitize` accepts, as far as
    /// storage allows: the application, whose settings change at runtime.
    pub const ADJUSTABLE: Self = Self {
        chains: GPU_MAX_CHAIN_COUNT,
        lambda: GPU_MAX_LAMBDA,
    };

    /// # Panics
    /// Panics unless `1 <= chains <= GPU_MAX_CHAIN_COUNT` and
    /// `1 <= lambda <= GPU_MAX_LAMBDA`.
    #[must_use]
    pub fn new(chains: u32, lambda: u32) -> Self {
        assert!(
            (1..=GPU_MAX_CHAIN_COUNT).contains(&chains) && (1..=GPU_MAX_LAMBDA).contains(&lambda),
            "engine capacity {chains} chains x {lambda} offspring is outside 1..={GPU_MAX_CHAIN_COUNT} x 1..={GPU_MAX_LAMBDA}"
        );
        Self { chains, lambda }
    }

    /// Exactly the workload of `params`, whose chain count and λ stay fixed.
    ///
    /// # Panics
    /// Panics for parameters that `sanitize` would change.
    #[must_use]
    pub fn for_params(params: &MutationParams) -> Self {
        Self::new(params.chain_count, params.lambda)
    }

    #[must_use]
    pub fn chains(self) -> u32 {
        self.chains
    }

    #[must_use]
    pub fn lambda(self) -> u32 {
        self.lambda
    }
}

/// Workgroup memory the engine requests from the device; every bundled program
/// fits (a CPU test sums each program's workgroup variables through naga). The
/// largest is `init_framebuffers` at a 32x16 tile, 25088 bytes. Dozen/D3D12
/// offers exactly 32 KiB. The polygon capacity does not enter it: mutate keeps
/// no copy of a parent there.
pub(super) const WORKGROUP_STORAGE_BYTES: u32 = 32 * 1024;

/// Bytes of one offspring's tile lists over `grid`: `TILE_MAX_POLYS` indices per tile.
pub(super) fn tile_list_bytes(grid: RasterGrid) -> u64 {
    grid.tiles() * u64::from(TILE_MAX_POLYS) * 4
}

/// The tile-list storage a workload is split over: the bytes of `tile_data`,
/// and an optional cap on a group's offspring that checkers set to exercise
/// grouping on small images.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TileStorage {
    pub bytes: u64,
    pub group_limit: Option<u32>,
}

impl TileStorage {
    /// Offspring whose lists over `grid` fit at once.
    pub fn slots(self, grid: RasterGrid) -> u64 {
        let held = self.bytes / tile_list_bytes(grid);
        self.group_limit.map_or(held, |limit| held.min(u64::from(limit)))
    }
}

/// Consecutive offspring binned and rastered together. `tile_data` holds the
/// tile lists of one group at a time, indexed from the group's first
/// offspring, so a workload whose lists exceed the allocation (large images:
/// 1024x682 at a 32x16 tile needs 361 MB of lists for 256 offspring, the
/// binding limit is 128 MiB) runs in several groups instead of writing past it.
/// Groups are as even as possible; one group when everything fits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TileGroups {
    offspring: u32,
    size: u32,
}

impl TileGroups {
    /// Split `offspring` into the fewest groups of at most `slots`, the
    /// offspring whose lists `tile_data` holds.
    ///
    /// # Panics
    /// Panics if `slots` is zero: the allocation holds no offspring's lists.
    pub fn new(offspring: u32, slots: u64) -> Self {
        assert!(slots >= 1, "tile storage holds no offspring's tile lists at this raster tile");
        let count = u64::from(offspring).div_ceil(slots).max(1);
        let size = u32::try_from(u64::from(offspring).div_ceil(count))
            .expect("group size fits u32");
        Self { offspring, size }
    }

    pub fn count(self) -> u32 {
        self.offspring.div_ceil(self.size.max(1)).max(1)
    }

    /// The first offspring of `group`.
    pub fn first(self, group: u32) -> u32 {
        group * self.size
    }

    /// Offspring in `group`; the last group may be smaller.
    pub fn len(self, group: u32) -> u32 {
        self.size.min(self.offspring - self.first(group))
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Workload {
    pub active: u32,
    pub offspring: u32,
    pub iterations: u32,
    pub raster: RasterGrid,
    pub tile_groups: TileGroups,
    pub mutate_y: u32,
    pub evaluations: u64,
}

impl Workload {
    pub fn for_batch(
        params: &MutationParams,
        image: [u32; 2],
        chain_capacity: u32,
        offspring_capacity: u32,
        mutation_group: u32,
        tile_storage: TileStorage,
    ) -> Self {
        assert!(params.chain_count <= chain_capacity,
            "{} chains exceed the engine's chain capacity ({chain_capacity}); construct it \
             with a larger EngineCapacity", params.chain_count);
        let active = params.chain_count;
        let offspring = active
            .checked_mul(params.lambda)
            .expect("offspring count overflow");
        assert!(offspring <= offspring_capacity,
            "active({active}) * lambda({}) = {offspring} exceeds the engine's offspring capacity \
             ({offspring_capacity}); construct it with a larger EngineCapacity",
            params.lambda);
        let iterations = params.gpu_batch_iters.max(1);
        let raster = RasterGrid::new(image, params.rasterize_wg);
        Self {
            active,
            offspring,
            iterations,
            raster,
            tile_groups: TileGroups::new(offspring, tile_storage.slots(raster)),
            mutate_y: params.lambda.div_ceil(mutation_group),
            evaluations: u64::from(iterations) * u64::from(offspring),
        }
    }
}

/// The raster worklist's header: the entry count.
pub(super) const RASTER_WORK_HEADER_BYTES: u64 = 4;

/// Set in an offspring's new-error accumulator by `rasterize_error` when one of
/// its evaluated tiles holds more than `TILE_MAX_POLYS` polygons, so its list
/// was truncated and its error is not the drawing's. `select` never accepts such
/// an offspring. The flag shares the word with the error sum, so every image's
/// largest error, 765 per pixel, must stay below it (`error_sum_fits`).
pub(super) const TILE_OVERFLOW_ERROR: u32 = 1 << 31;

/// Whether a whole image's byte-L1 error stays below `TILE_OVERFLOW_ERROR`.
pub(super) fn error_sum_fits(image: [u32; 2]) -> bool {
    765 * u64::from(image[0]) * u64::from(image[1]) < u64::from(TILE_OVERFLOW_ERROR)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ResourceSizes {
    pub parents: u64,
    pub working: u64,
    pub errors: u64,
    pub tile_data: u64,
    pub tile_counts: u64,
    pub raster_work: u64,
    pub framebuffers: u64,
    pub parent_errors: u64,
    pub updates: u64,
    pub chain_changes: u64,
    pub fitness: u64,
    pub offspring_ops: u64,
    pub operator_stats: u64,
}

impl ResourceSizes {
    pub fn new(chains: u64, offspring: u64, image: [u32; 2], grid: RasterGrid) -> Self {
        let pixels = u64::from(image[0]) * u64::from(image[1]);
        Self {
            parents: chains * GPU_DRAWING_STATE_SIZE as u64,
            working: offspring * GPU_DRAWING_STATE_SIZE as u64,
            errors: offspring * 8,
            tile_data: offspring * grid.tiles() * u64::from(TILE_MAX_POLYS) * 4,
            // The final word retains diagnostic occupancy across every iteration.
            tile_counts: offspring * grid.tiles() * 4 + 4,
            // The entry count, then one entry per (offspring, tile).
            raster_work: RASTER_WORK_HEADER_BYTES + offspring * grid.tiles() * 4,
            framebuffers: chains * pixels * 4,
            parent_errors: chains * 4,
            updates: chains * 16,
            chain_changes: chains * GPU_CHAIN_CHANGE_SIZE as u64,
            fitness: chains * 4,
            offspring_ops: offspring * 4,
            operator_stats: OPERATOR_STATS_WORDS as u64 * 4,
        }
    }

    pub fn entries(self) -> [(&'static str, u64); 13] {
        [
            ("parent states", self.parents),
            ("working states", self.working),
            ("error accumulators", self.errors),
            ("tile data", self.tile_data),
            ("tile counts", self.tile_counts),
            ("raster worklist", self.raster_work),
            ("parent framebuffers", self.framebuffers),
            ("parent errors", self.parent_errors),
            ("update flags", self.updates),
            ("chain change records", self.chain_changes),
            ("fitness", self.fitness),
            ("offspring operators", self.offspring_ops),
            ("operator stats", self.operator_stats),
        ]
    }

    pub fn validate(self, actual: Self, limit: u64) -> Result<(), String> {
        for ((label, needed), (_, allocated)) in self.entries().into_iter().zip(actual.entries()) {
            if needed > allocated || needed > limit || needed / 4 > u64::from(u32::MAX) {
                return Err(format!(
                    "{label} requires {needed} bytes, allocation {allocated}, device limit {limit}"
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(super) struct AllocationPlan {
    pub chain_count: u32,
    pub offspring_capacity: u32,
    pub max_buffer_size: u64,
    pub resources: ResourceSizes,
}

impl AllocationPlan {
    /// Storage for `capacity`, each buffer clamped to `max_storage`.
    pub fn new(capacity: EngineCapacity, image: [u32; 2], max_storage: u64) -> Self {
        assert!(
            error_sum_fits(image),
            "a {}x{} image's error sum reaches the tile-overflow flag",
            image[0],
            image[1]
        );
        let states = max_storage / GPU_DRAWING_STATE_SIZE as u64;
        // Sized for the smallest supported tile, the most tiles.
        let grid = RasterGrid::new(image, [8, 8]);
        // Tile counts and worklist entries take one word per (offspring, tile)
        // for every offspring of an iteration, so they bound the offspring too.
        let tile_words = (max_storage - RASTER_WORK_HEADER_BYTES) / (grid.tiles() * 4);
        let chains = u64::from(capacity.chains).min(states);
        let offspring = states.min(tile_words).min(chains * u64::from(capacity.lambda));
        let mut resources = ResourceSizes::new(chains, offspring, image, grid);
        // The device may create buffers up to the whole binding limit, whatever
        // the workload: requesting a limit allocates nothing, and diagnostic
        // snapshots stage several resources in one buffer. (A limit derived from
        // the states would refuse small engines their snapshots.) Each resource
        // keeps its independent clamp; usable capacities are checked against
        // actual storage before qualification.
        let max_buffer_size = max_storage;
        // Tile lists beyond the allocation are binned in groups (`TileGroups`).
        resources.tile_data = resources.tile_data.min(max_buffer_size);
        // The diagnostic word is allocated too: an engine sized exactly for its
        // workload has no unused worst-case slack to hold it.
        assert!(resources.tile_counts.max(resources.raster_work) <= max_buffer_size,
            "the offspring capacity bounds tile counts and the raster worklist");
        resources.framebuffers = resources.framebuffers.min(max_buffer_size);
        Self {
            chain_count: u32::try_from(chains).expect("chain count fits requested u32"),
            offspring_capacity: u32::try_from(offspring).expect("offspring capacity fits u32"),
            max_buffer_size,
            resources,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BINDING_LIMIT: u64 = 128 * 1024 * 1024;
    const UNGROUPED: TileStorage = TileStorage { bytes: BINDING_LIMIT, group_limit: None };

    /// Groups of the default 4 x 64 workload at a 32x16 tile, its lists in one
    /// binding-limit allocation.
    fn default_groups(image: [u32; 2]) -> TileGroups {
        let params = MutationParams::default();
        let plan = AllocationPlan::new(EngineCapacity::for_params(&params), image, BINDING_LIMIT);
        let storage = TileStorage { bytes: plan.resources.tile_data, group_limit: None };
        Workload::for_batch(&params, image, plan.chain_count, plan.offspring_capacity, 1, storage)
            .tile_groups
    }

    #[test]
    fn tile_lists_beyond_the_allocation_run_in_groups() {
        // 1376 tiles of 1 KiB: 95 offspring per 128 MiB.
        assert_eq!(tile_list_bytes(RasterGrid::new([1024, 682], [32, 16])), 1_409_024);
        for (image, count, size) in [
            ([384, 384], 1, 256),
            ([512, 340], 1, 256),
            ([768, 512], 2, 128),
            ([1024, 682], 3, 86),
            ([1024, 1024], 4, 64),
        ] {
            let groups = default_groups(image);
            assert_eq!((groups.count(), groups.len(0)), (count, size), "{image:?}");
            let lengths: Vec<u32> = (0..count).map(|g| groups.len(g)).collect();
            assert_eq!(lengths.iter().sum::<u32>(), 256, "{image:?}");
            assert!((1..count).all(|g| groups.first(g) == groups.first(g - 1) + groups.len(g - 1)));
        }
    }

    #[test]
    fn groups_cover_every_offspring_once_within_the_slots() {
        for offspring in [1, 2, 63, 64, 65, 255, 256, 4092] {
            for slots in [1, 2, 3, 37, 95, 256, 10_000] {
                let groups = TileGroups::new(offspring, slots);
                let count = groups.count();
                assert_eq!(count, u32::try_from(u64::from(offspring).div_ceil(slots)).unwrap());
                let mut next = 0;
                for g in 0..count {
                    assert_eq!(groups.first(g), next);
                    assert!((1..=slots).contains(&u64::from(groups.len(g))));
                    next += groups.len(g);
                }
                assert_eq!(next, offspring, "{offspring} offspring, {slots} slots");
            }
        }
    }

    #[test]
    fn a_group_limit_caps_the_slots() {
        let grid = RasterGrid::new([384, 384], [32, 16]);
        assert_eq!(UNGROUPED.slots(grid), 455);
        let limited = TileStorage { group_limit: Some(37), ..UNGROUPED };
        assert_eq!(limited.slots(grid), 37);
        assert_eq!(TileGroups::new(256, limited.slots(grid)).count(), 7);
    }

    #[test]
    #[should_panic(expected = "holds no offspring")]
    fn storage_without_one_offspring_of_lists_fails_loudly() {
        let grid = RasterGrid::new([1024, 1024], [8, 8]);
        let short = TileStorage { bytes: tile_list_bytes(grid) - 4, group_limit: None };
        let _ = TileGroups::new(1, short.slots(grid));
    }

    #[test]
    fn the_overflow_flag_lies_above_every_supported_error_sum() {
        assert!(error_sum_fits([1024, 1024]));
        assert!(error_sum_fits([1, 2_807_168]));
        assert!(!error_sum_fits([1, 2_807_169]));
    }

    #[test]
    fn allocation_preserves_independent_storage_clamps() {
        let plan = AllocationPlan::new(EngineCapacity::ADJUSTABLE, [384, 384], 128 * 1024 * 1024);
        assert_eq!(plan.chain_count, 1024);
        // 128 MiB of 32800-byte states.
        assert_eq!(plan.offspring_capacity, 4092);
        assert_eq!(plan.max_buffer_size, 134_217_728);
        assert_eq!(plan.resources.framebuffers, 134_217_728);
        assert_eq!(plan.resources.tile_data, 134_217_728);
        assert_eq!(plan.resources.tile_counts, 37_711_876);
        assert_eq!(plan.resources.raster_work, 37_711_876);
        let smaller = AllocationPlan::new(EngineCapacity::new(1, 64), [17, 19], 128 * 1024 * 1024);
        assert_eq!(smaller.chain_count, 1);
        assert_eq!(smaller.offspring_capacity, 64);
        assert_eq!(smaller.resources.framebuffers, 1292);
    }

    #[test]
    fn allocation_follows_the_constructed_workload() {
        let params = MutationParams::default();
        let plan = AllocationPlan::new(
            EngineCapacity::for_params(&params),
            [384, 384],
            128 * 1024 * 1024,
        );
        assert_eq!((plan.chain_count, plan.offspring_capacity), (4, 256));
        assert_eq!(plan.max_buffer_size, 128 * 1024 * 1024);
        assert_eq!(plan.resources.parents, 4 * GPU_DRAWING_STATE_SIZE as u64);
        assert_eq!(plan.resources.working, 256 * GPU_DRAWING_STATE_SIZE as u64);
        assert_eq!(plan.resources.framebuffers, 4 * 384 * 384 * 4);
        let storage = TileStorage { bytes: plan.resources.tile_data, group_limit: None };
        let work = Workload::for_batch(&params, [384, 384], plan.chain_count,
            plan.offspring_capacity, 1, storage);
        assert_eq!(work.offspring, 256);
        assert_eq!(work.tile_groups.count(), 1);
    }

    #[test]
    #[should_panic(expected = "offspring capacity")]
    fn a_batch_beyond_the_offspring_capacity_fails_loudly() {
        let params = MutationParams { chain_count: 4, lambda: 64, ..MutationParams::default() };
        let _ = Workload::for_batch(&params, [384, 384], 4, 128, 1, UNGROUPED);
    }

    #[test]
    #[should_panic(expected = "chain capacity")]
    fn a_batch_beyond_the_chain_capacity_fails_loudly() {
        let params = MutationParams { chain_count: 8, lambda: 1, ..MutationParams::default() };
        let _ = Workload::for_batch(&params, [384, 384], 4, 256, 1, UNGROUPED);
    }

    #[test]
    #[should_panic(expected = "outside")]
    fn capacity_rejects_workloads_sanitize_rejects() {
        let _ = EngineCapacity::new(1, GPU_MAX_LAMBDA + 1);
    }

    #[test]
    fn dispatch_preserves_logical_groups_and_odd_image_edges() {
        let params = MutationParams {
            chain_count: 4,
            lambda: 64,
            gpu_batch_iters: 64,
            rasterize_wg: [32, 16],
            ..MutationParams::default()
        };
        for group in [1, 2, 4, 8, 16, 32, 64] {
            let work = Workload::for_batch(&params, [65, 51], 1024, 8192, group, UNGROUPED);
            assert_eq!((work.raster.x, work.raster.y), (3, 4));
            assert_eq!((work.active, work.offspring), (4, 256));
            assert_eq!(work.mutate_y, 64 / group);
            assert_eq!(work.evaluations, 16_384);
        }
    }

    #[test]
    fn storage_validation_covers_each_resource_and_diagnostic_tail() {
        let required = ResourceSizes::new(4, 256, [65, 51], RasterGrid::new([65, 51], [32, 16]));
        assert_eq!(required.tile_counts, 12_292);
        assert!(required.validate(required, 128 * 1024 * 1024).is_ok());
        let undersized = [
            ResourceSizes {
                parents: required.parents - 4,
                ..required
            },
            ResourceSizes {
                working: required.working - 4,
                ..required
            },
            ResourceSizes {
                errors: required.errors - 4,
                ..required
            },
            ResourceSizes {
                tile_data: required.tile_data - 4,
                ..required
            },
            ResourceSizes {
                tile_counts: required.tile_counts - 4,
                ..required
            },
            ResourceSizes {
                raster_work: required.raster_work - 4,
                ..required
            },

            ResourceSizes {
                framebuffers: required.framebuffers - 4,
                ..required
            },
            ResourceSizes {
                parent_errors: required.parent_errors - 4,
                ..required
            },
            ResourceSizes {
                updates: required.updates - 4,
                ..required
            },
            ResourceSizes {
                chain_changes: required.chain_changes - 4,
                ..required
            },
            ResourceSizes {
                fitness: required.fitness - 4,
                ..required
            },
            ResourceSizes {
                offspring_ops: required.offspring_ops - 4,
                ..required
            },
            ResourceSizes {
                operator_stats: required.operator_stats - 4,
                ..required
            },
        ];
        for short in undersized {
            assert!(required.validate(short, 128 * 1024 * 1024).is_err());
        }
        assert!(required.validate(required, required.tile_data - 1).is_err());
    }
}
