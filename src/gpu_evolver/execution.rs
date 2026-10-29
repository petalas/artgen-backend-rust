//! Capacity, storage and dispatch rules shared by allocation and execution.
//! Logical raster tiles and logical mutation groups are independent of lanes.
use crate::{mutation_params::MutationParams, settings::TILE_MAX_POLYS};

use super::buffers::GPU_DRAWING_STATE_SIZE;

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
        let active = params.chain_count.min(chain_capacity);
        let offspring = active
            .checked_mul(params.lambda)
            .expect("offspring count overflow");
        assert!(offspring <= offspring_capacity,
            "active({active}) * lambda({}) = {offspring} exceeds offspring_capacity({offspring_capacity})",
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

/// raster_worklist.wgsl's header: the entry count.
pub(super) const RASTER_WORK_HEADER_BYTES: u64 = 4;

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
            fitness: chains * 4,
        }
    }

    pub fn entries(self) -> [(&'static str, u64); 10] {
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
    pub fn new(requested_chains: u32, image: [u32; 2], max_storage: u64) -> Self {
        let states = max_storage / GPU_DRAWING_STATE_SIZE as u64;
        let chains = u64::from(requested_chains).min(states);
        let offspring = states.min(chains * u64::from(crate::settings::GPU_MAX_LAMBDA));
        let mut resources =
            ResourceSizes::new(chains, offspring, image, RasterGrid::new(image, [8, 8]));
        // Preserve the existing requested device limit and independent clamps.
        // Usable capacities are checked against actual storage before qualification.
        let max_buffer_size =
            (chains * u64::from(crate::settings::GPU_MAX_LAMBDA) * GPU_DRAWING_STATE_SIZE as u64)
                .min(max_storage)
                .max(resources.tile_data.min(max_storage))
                .max((resources.tile_counts - 4).min(max_storage));
        resources.tile_data = resources.tile_data.min(max_buffer_size);
        resources.tile_counts = (resources.tile_counts - 4).min(max_buffer_size);
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
    fn allocation_preserves_independent_storage_clamps() {
        let plan = AllocationPlan::new(1024, [384, 384], 128 * 1024 * 1024);
        assert_eq!(plan.chain_count, 1024);
        assert_eq!(plan.offspring_capacity, 8371);
        assert_eq!(plan.max_buffer_size, 134_217_728);
        assert_eq!(plan.resources.framebuffers, 134_217_728);
        assert_eq!(plan.resources.tile_data, 134_217_728);
        assert_eq!(plan.resources.tile_counts, 77_147_136);
        assert_eq!(plan.resources.raster_work, 77_147_140);
        let smaller = AllocationPlan::new(1, [17, 19], 128 * 1024 * 1024);
        assert_eq!(smaller.chain_count, 1);
        assert_eq!(smaller.offspring_capacity, 64);
        assert_eq!(smaller.resources.framebuffers, 1292);
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
