//! Capacity, storage and dispatch rules shared by allocation and execution.
//! Logical raster tiles and logical mutation groups are independent of lanes.
use crate::{
    mutation_params::MutationParams,
    settings::{GPU_MAX_CHAIN_COUNT, GPU_MAX_LAMBDA, TILE_MAX_POLYS},
};

use super::buffers::GPU_DRAWING_STATE_SIZE;
use super::changes::GPU_CHAIN_CHANGE_SIZE;

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

#[derive(Clone, Copy, Debug)]
pub(super) struct Workload {
    pub active: u32,
    pub offspring: u32,
    pub iterations: u32,
    pub raster: RasterGrid,
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
        Self {
            active,
            offspring,
            iterations,
            raster: RasterGrid::new(image, params.rasterize_wg),
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
        }
    }

    pub fn entries(self) -> [(&'static str, u64); 11] {
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
        let chains = u64::from(capacity.chains).min(states);
        let offspring = states.min(chains * u64::from(capacity.lambda));
        let mut resources =
            ResourceSizes::new(chains, offspring, image, RasterGrid::new(image, [8, 8]));
        // The device may create buffers up to the whole binding limit, whatever
        // the workload: requesting a limit allocates nothing, and diagnostic
        // snapshots stage several resources in one buffer. (A limit derived from
        // the states would refuse small engines their snapshots.) Each resource
        // keeps its independent clamp; usable capacities are checked against
        // actual storage before qualification.
        let max_buffer_size = max_storage;
        resources.tile_data = resources.tile_data.min(max_buffer_size);
        // The diagnostic word is allocated too: an engine sized exactly for its
        // workload has no unused worst-case slack to hold it.
        resources.tile_counts = resources.tile_counts.min(max_buffer_size);
        resources.raster_work = resources.raster_work.min(max_buffer_size);
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
        let work = Workload::for_batch(&params, [384, 384], plan.chain_count,
            plan.offspring_capacity, 1);
        assert_eq!(work.offspring, 256);
    }

    #[test]
    #[should_panic(expected = "offspring capacity")]
    fn a_batch_beyond_the_offspring_capacity_fails_loudly() {
        let params = MutationParams { chain_count: 4, lambda: 64, ..MutationParams::default() };
        let _ = Workload::for_batch(&params, [384, 384], 4, 128, 1);
    }

    #[test]
    #[should_panic(expected = "chain capacity")]
    fn a_batch_beyond_the_chain_capacity_fails_loudly() {
        let params = MutationParams { chain_count: 8, lambda: 1, ..MutationParams::default() };
        let _ = Workload::for_batch(&params, [384, 384], 4, 256, 1);
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
            let work = Workload::for_batch(&params, [65, 51], 1024, 8192, group);
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
        ];
        for short in undersized {
            assert!(required.validate(short, 128 * 1024 * 1024).is_err());
        }
        assert!(required.validate(required, required.tile_data - 1).is_err());
    }
}
