//! What each stage binds, declared once. A stage's list builds both its bind
//! group layout and its bind group; a binding's number is its position in the
//! list. Group 0 holds a stage's own resources; group 1 is the params uniform,
//! shared by every stage. The minimum binding size is one element of the
//! resource, so it follows from the resource. A CPU test compares every
//! bundled shader's declarations with these lists. Shader variants are
//! installed against the same layouts and may declare a subset.

use std::num::NonZeroU64;

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType, Device, ShaderStages,
    TextureSampleType, TextureViewDimension,
};

use super::buffers::{ControlFlags, GpuParams, GPU_DRAWING_STATE_SIZE};
use super::changes::GPU_CHAIN_CHANGE_SIZE;

/// A resource of `GpuPipeline` that a stage binds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Resource {
    ChainStates,
    WorkingStates,
    ReferenceImage,
    ErrorAccumulators,
    Control,
    FitnessPacked,
    TileData,
    TileCounts,
    RasterWork,
    ChainFramebuffers,
    ChainTotalErrors,
    ChainUpdateFlags,
    ChainChanges,
    Params,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Access {
    Read,
    /// Read-write storage. The reference texture and params are read only.
    Write,
}

pub(super) struct Stage {
    /// Labels are `{name}_bgl` and `{name}_bg`.
    pub name: &'static str,
    pub bindings: &'static [(Resource, Access)],
}

use Access::{Read, Write};
use Resource::{
    ChainChanges, ChainFramebuffers, ChainStates, ChainTotalErrors, ChainUpdateFlags, Control,
    ErrorAccumulators, FitnessPacked, Params, RasterWork, ReferenceImage, TileCounts, TileData,
    WorkingStates,
};

pub(super) const PARAMS: Stage = Stage {
    name: "params",
    bindings: &[(Params, Read)],
};

pub(super) const MUTATE: Stage = Stage {
    name: "mutate",
    bindings: &[(ChainStates, Read), (WorkingStates, Write)],
};

pub(super) const RASTERIZE_ERROR: Stage = Stage {
    name: "rasterize_error",
    bindings: &[
        (WorkingStates, Read),
        (ReferenceImage, Read),
        (ErrorAccumulators, Write),
        (TileData, Read),
        (TileCounts, Read),
        (ChainStates, Read),
        (ChainFramebuffers, Read),
        (RasterWork, Read),
    ],
};

pub(super) const BIN_POLYGONS: Stage = Stage {
    name: "bin_polygons",
    bindings: &[
        (WorkingStates, Read),
        (TileData, Write),
        (TileCounts, Write),
        (ChainStates, Read),
        (RasterWork, Write),
    ],
};

/// The worklist pass that shader-variant binners without `raster_work` need.
pub(super) const RASTER_WORKLIST: Stage = Stage {
    name: "raster_worklist",
    bindings: &[
        (WorkingStates, Read),
        (ChainStates, Read),
        (TileCounts, Read),
        (RasterWork, Write),
    ],
};

pub(super) const SELECT: Stage = Stage {
    name: "select",
    bindings: &[
        (ChainStates, Write),
        (WorkingStates, Read),
        (ErrorAccumulators, Write),
        (Control, Write),
        (FitnessPacked, Write),
        (ChainUpdateFlags, Write),
        (ChainTotalErrors, Write),
        (ChainChanges, Write),
    ],
};

pub(super) const INIT_FRAMEBUFFERS: Stage = Stage {
    name: "init_framebuffers",
    bindings: &[
        (ChainStates, Read),
        (ReferenceImage, Read),
        (ChainFramebuffers, Write),
        (ChainTotalErrors, Write),
    ],
};

pub(super) const UPDATE_FRAMEBUFFERS: Stage = Stage {
    name: "update_framebuffers",
    bindings: &[
        (ChainStates, Read),
        (ChainFramebuffers, Write),
        (ChainUpdateFlags, Read),
        // The winning offspring's tile polygon lists.
        (TileData, Read),
        (TileCounts, Read),
        (RasterWork, Write),
        (ChainChanges, Read),
    ],
};

const STAGES: [Stage; 7] = [
    MUTATE,
    RASTERIZE_ERROR,
    BIN_POLYGONS,
    RASTER_WORKLIST,
    SELECT,
    INIT_FRAMEBUFFERS,
    UPDATE_FRAMEBUFFERS,
];

/// The most storage buffers any stage binds, the device limit to request.
pub(super) const MAX_STORAGE_BUFFERS: u32 = {
    let mut max = 0;
    let mut s = 0;
    while s < STAGES.len() {
        let bindings = STAGES[s].bindings;
        let mut count = 0;
        let mut b = 0;
        while b < bindings.len() {
            if !matches!(bindings[b].0, ReferenceImage | Params) {
                count += 1;
            }
            b += 1;
        }
        if count > max {
            max = count;
        }
        s += 1;
    }
    max
};

/// The binding number of `resource` in `stage`.
pub(super) const fn binding(stage: &Stage, resource: Resource) -> u32 {
    let mut b: u32 = 0;
    while (b as usize) < stage.bindings.len() {
        if stage.bindings[b as usize].0 as u32 == resource as u32 {
            return b;
        }
        b += 1;
    }
    panic!("the stage does not bind the resource");
}

impl Resource {
    /// Bytes of one element: the minimum binding size.
    fn element_size(self) -> usize {
        match self {
            ChainStates | WorkingStates => GPU_DRAWING_STATE_SIZE,
            Control => std::mem::size_of::<ControlFlags>(),
            ChainChanges => GPU_CHAIN_CHANGE_SIZE,
            Params => std::mem::size_of::<GpuParams>(),
            ReferenceImage => unreachable!("the reference image is a texture"),
            ErrorAccumulators | FitnessPacked | TileData | TileCounts | RasterWork
            | ChainFramebuffers | ChainTotalErrors | ChainUpdateFlags => 4,
        }
    }

    fn binding_type(self, access: Access) -> BindingType {
        assert!(
            access == Read || !matches!(self, ReferenceImage | Params),
            "{self:?} is bound read only"
        );
        let ty = match self {
            ReferenceImage => {
                return BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                };
            }
            Params => BufferBindingType::Uniform,
            _ => BufferBindingType::Storage {
                read_only: access == Read,
            },
        };
        BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(self.element_size() as u64),
        }
    }
}

fn layout_entries(stage: &Stage) -> Vec<BindGroupLayoutEntry> {
    (0u32..)
        .zip(stage.bindings)
        .map(|(binding, &(resource, access))| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: resource.binding_type(access),
            count: None,
        })
        .collect()
}

pub(super) fn layout(device: &Device, stage: &Stage) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some(&format!("{}_bgl", stage.name)),
        entries: &layout_entries(stage),
    })
}

/// The stage's bind group over `layout`, binding each resource that `bind` returns.
pub(super) fn bind_group<'a>(
    device: &Device,
    stage: &Stage,
    layout: &BindGroupLayout,
    bind: impl Fn(Resource) -> BindingResource<'a>,
) -> BindGroup {
    let entries: Vec<_> = (0u32..)
        .zip(stage.bindings)
        .map(|(binding, &(resource, _))| BindGroupEntry {
            binding,
            resource: bind(resource),
        })
        .collect();
    device.create_bind_group(&BindGroupDescriptor {
        label: Some(&format!("{}_bg", stage.name)),
        layout,
        entries: &entries,
    })
}

#[cfg(test)]
mod tests {
    use wgpu::naga::{
        self, AddressSpace, ImageClass, ImageDimension, ScalarKind, StorageAccess, TypeInner,
    };
    use wgpu::{BindingType, BufferBindingType, TextureSampleType, TextureViewDimension};

    use super::{
        layout_entries, Stage, BIN_POLYGONS, INIT_FRAMEBUFFERS, MUTATE, PARAMS, RASTERIZE_ERROR,
        RASTER_WORKLIST, SELECT, UPDATE_FRAMEBUFFERS,
    };
    use crate::gpu_evolver::programs::{bundled_shader_source, BundledShader, TileStage};

    /// A binding as both sides can describe it: (group, binding, kind, minimum size).
    type Declared = (u32, u32, &'static str, u64);

    fn from_layouts(stage: &Stage) -> Vec<Declared> {
        let groups = [(0, stage), (1, &PARAMS)];
        let mut declared = Vec::new();
        for (group, stage) in groups {
            for entry in layout_entries(stage) {
                let BindingType::Buffer {
                    ty,
                    has_dynamic_offset: false,
                    min_binding_size,
                } = entry.ty
                else {
                    assert_eq!(
                        entry.ty,
                        BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        }
                    );
                    declared.push((group, entry.binding, "texture_2d<f32>", 0));
                    continue;
                };
                let kind = match ty {
                    BufferBindingType::Uniform => "uniform",
                    BufferBindingType::Storage { read_only: true } => "storage, read",
                    BufferBindingType::Storage { read_only: false } => "storage, read_write",
                };
                declared.push((group, entry.binding, kind, min_binding_size.unwrap().get()));
            }
        }
        declared
    }

    fn from_shader(shader: BundledShader) -> Vec<Declared> {
        let source = bundled_shader_source(shader);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("{shader:?}: {}", error.emit_to_string(&source)));
        let mut declared: Vec<Declared> = module
            .global_variables
            .iter()
            .filter_map(|(_, global)| {
                let slot = global.binding.as_ref()?;
                let inner = &module.types[global.ty].inner;
                // One element of a runtime-sized array, as wgpu checks it.
                let size = u64::from(inner.size(module.to_ctx()));
                let kind = match global.space {
                    AddressSpace::Uniform => "uniform",
                    AddressSpace::Storage { access } if access.contains(StorageAccess::STORE) => {
                        "storage, read_write"
                    }
                    AddressSpace::Storage { .. } => "storage, read",
                    AddressSpace::Handle => {
                        let TypeInner::Image {
                            dim: ImageDimension::D2,
                            arrayed: false,
                            class:
                                ImageClass::Sampled {
                                    kind: ScalarKind::Float,
                                    multi: false,
                                },
                        } = inner
                        else {
                            panic!("{shader:?}: unexpected handle {inner:?}");
                        };
                        return Some((slot.group, slot.binding, "texture_2d<f32>", 0));
                    }
                    other => panic!("{shader:?}: unexpected address space {other:?}"),
                };
                Some((slot.group, slot.binding, kind, size))
            })
            .collect();
        declared.sort_unstable();
        declared
    }

    /// Every bundled program declares exactly its stage's bindings: group,
    /// number, kind, access and element size.
    #[test]
    fn bundled_shaders_declare_their_stage_bindings() {
        let tile = |stage| BundledShader::Tile(stage, [16, 16]);
        let stages = [
            (BundledShader::Mutate, MUTATE),
            (tile(TileStage::RasterizeError), RASTERIZE_ERROR),
            (tile(TileStage::BinPolygons), BIN_POLYGONS),
            (tile(TileStage::RasterWorklist), RASTER_WORKLIST),
            (BundledShader::Select, SELECT),
            (tile(TileStage::InitFramebuffers), INIT_FRAMEBUFFERS),
            (tile(TileStage::UpdateFramebuffers), UPDATE_FRAMEBUFFERS),
        ];
        for (shader, stage) in &stages {
            assert_eq!(from_shader(*shader), from_layouts(stage), "{shader:?}");
        }
    }
}
