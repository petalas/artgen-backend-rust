//! Default-off benchmark prototype. Only dispatch mapping changes; the accepted
//! C raster arithmetic is copied byte-for-byte after its metadata prologue.
use std::num::NonZeroU64;

use sha2::{Digest, Sha256};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType,
    BufferDescriptor, BufferUsages, ComputePipeline, ComputePipelineDescriptor, Device,
    ErrorFilter, ErrorScopeGuard, PipelineCache, PipelineCompilationOptions, PipelineLayout,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
    TextureSampleType, TextureViewDimension,
};

use super::pipeline::{specialize_raster_source, GpuPipeline};

pub(super) const ACCEPTED_C_SHA256: &str = "a292287ae6d9c212c7970dd3e712001a1b5a827975e6216355384fc0106eb3af";
const MAX_VARIANTS: usize = 5;
const DISPATCH_LIMIT: u64 = 65535;
pub(super) const DEFAULT_ENABLED: bool = false;

/// Actual compiled engine resources for a caller-drained, compatible-buffer checker.
/// Group 1 is the existing Params layout. Prefix group 0 is working/prefix/args
/// at 0/1/2; raster group 0 is the original 0..6 plus read-only prefix at 7.
pub struct CompactRasterResources {
    pub prefix_buf: Buffer,
    pub indirect_args_buf: Buffer,
    pub prefix_layout: BindGroupLayout,
    pub raster_layout: BindGroupLayout,
    pub prefix_bind_group: BindGroup,
    pub raster_bind_group: BindGroup,
    pub prefix_pipeline: ComputePipeline,
    pub raster_pipeline: ComputePipeline,
    pub prefix_source: String,
    pub raster_source: String,
    wg: [u32; 2],
    variants: Vec<CompactVariant>,
}

struct CompactVariant {
    wg: [u32; 2],
    prefix: ComputePipeline,
    raster: ComputePipeline,
    prefix_source: String,
    raster_source: String,
}

pub(super) fn validate_source(source: &str) -> Result<(), String> {
    if source.len() > 256 * 1024
        || format!("{:x}", Sha256::digest(source.as_bytes())) != ACCEPTED_C_SHA256
    {
        return Err("compact raster requires the exact accepted a292 C source".into());
    }
    Ok(())
}

/// Checks allocated prefix capacity and the worst possible active tile total.
/// The existing benchmark resource checks separately validate active tile strides.
pub(super) fn prefix_bytes(
    capacity: u32,
    image: [u32; 2],
    wg: [u32; 2],
    byte_limit: u64,
    dispatch_limit: u32,
) -> Result<u64, String> {
    if capacity == 0
        || image.contains(&0)
        || ![[32, 16], [16, 16], [32, 8], [16, 8], [8, 8]].contains(&wg)
        || u64::from(dispatch_limit) < DISPATCH_LIMIT
    {
        return Err(
            "unsupported compact raster capacity, image, workgroup or dispatch limit".into(),
        );
    }
    let bytes = (u64::from(capacity) + 1) * 4;
    let tiles = u64::from(image[0].div_ceil(wg[0]))
        .checked_mul(u64::from(image[1].div_ceil(wg[1])))
        .ok_or("compact tile count overflow")?;
    let total = u64::from(capacity)
        .checked_mul(tiles)
        .ok_or("compact prefix sum overflow")?;
    let padding_bound = total.div_ceil(DISPATCH_LIMIT) - 1;
    if byte_limit < 12
        || bytes > byte_limit
        || bytes / 4 > u64::from(u32::MAX)
        || total
            .checked_add(padding_bound)
            .is_none_or(|n| n > u64::from(u32::MAX))
        || image[0].checked_add(wg[0] - 1).is_none()
        || image[1].checked_add(wg[1] - 1).is_none()
        || total.div_ceil(DISPATCH_LIMIT) > DISPATCH_LIMIT
    {
        return Err("compact raster exceeds buffer, u32 indexing or 2D dispatch capacity".into());
    }
    Ok(bytes)
}

pub(super) fn sources(source: &str, wg: [u32; 2]) -> Result<(String, String), String> {
    validate_source(source)?;
    let declarations_end = source
        .find("@group(0)")
        .ok_or("missing raster declarations")?;
    let prefix = format!(
        "{}{}",
        &source[..declarations_end],
        include_str!("compact_prefix.wgsl")
    );
    let start = source
        .find("    // Derive 2D pixel coordinates")
        .ok_or("unsupported raster prologue")?;
    let end = source
        .find("    var pixel_error = 0u;")
        .ok_or("unsupported raster metadata")?;
    let raster = format!(
        "{}{}{}",
        &source[..start],
        include_str!("compact_raster_prologue.wgsl"),
        &source[end..]
    );
    let raster = replace_once(
        &raster,
        "    @builtin(workgroup_id) wid: vec3<u32>,",
        "    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(num_workgroups) grid: vec3<u32>,",
    )?;
    let raster = replace_once(&raster, "@group(1) @binding(0) var<uniform>",
        "@group(0) @binding(7) var<storage, read> compact_prefix: array<u32>;\n@group(1) @binding(0) var<uniform>")?;
    let raster = replace_once(&raster, "var<workgroup> shared_skip_tile:",
        "var<workgroup> shared_compact_chain: u32;\nvar<workgroup> shared_compact_x: u32;\nvar<workgroup> shared_compact_y: u32;\nvar<workgroup> shared_skip_tile:")?;
    let raster = replace_once(
        &raster,
        "    let tile_id = wid.y * num_tiles_x + wid.x;",
        "    let tile_id = shared_compact_y * num_tiles_x + shared_compact_x;",
    )?;
    Ok((
        specialize_raster_source(&prefix, wg),
        specialize_raster_source(&raster, wg),
    ))
}

fn replace_once(source: &str, from: &str, to: &str) -> Result<String, String> {
    if source.matches(from).count() != 1 {
        return Err(format!("unsupported compact raster marker {from:?}"));
    }
    Ok(source.replacen(from, to, 1))
}

fn storage_entry(binding: u32, read_only: bool, size: u64) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(size),
        },
        count: None,
    }
}

fn layouts(device: &Device) -> (BindGroupLayout, BindGroupLayout) {
    let prefix = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("compact_prefix_layout"),
        entries: &[
            storage_entry(0, true, super::buffers::GPU_DRAWING_STATE_SIZE as u64),
            storage_entry(1, false, 4),
            storage_entry(2, false, 12),
        ],
    });
    let raster = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("compact_raster_layout"),
        entries: &[
            storage_entry(0, true, super::buffers::GPU_DRAWING_STATE_SIZE as u64),
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            storage_entry(2, false, 4),
            storage_entry(3, true, 4),
            storage_entry(4, true, 4),
            storage_entry(5, true, super::buffers::GPU_DRAWING_STATE_SIZE as u64),
            storage_entry(6, true, 4),
            storage_entry(7, true, 4),
        ],
    });
    (prefix, raster)
}

// Power of Ten rule 4: keep this 65-line pair of concrete descriptors together
// so the original raster bindings and added prefix remain directly auditable.
// The scope is two fixed groups; the compatible-buffer checker verifies both.
fn bind_groups(
    p: &GpuPipeline,
    prefix: &Buffer,
    args: &Buffer,
    prefix_layout: &BindGroupLayout,
    raster_layout: &BindGroupLayout,
) -> (BindGroup, BindGroup) {
    let prefix_bg = p.device.create_bind_group(&BindGroupDescriptor {
        label: Some("compact_prefix_bg"),
        layout: prefix_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: p.working_states_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: prefix.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: args.as_entire_binding(),
            },
        ],
    });
    let raster_bg = p.device.create_bind_group(&BindGroupDescriptor {
        label: Some("compact_raster_bg"),
        layout: raster_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: p.working_states_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&p.reference_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: p.error_accumulators_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: p.tile_data_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: p.tile_counts_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: p.chain_states_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: p.chain_framebuffers_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 7,
                resource: prefix.as_entire_binding(),
            },
        ],
    });
    (prefix_bg, raster_bg)
}

fn error_scopes(device: &Device) -> [ErrorScopeGuard; 3] {
    [
        device.push_error_scope(ErrorFilter::Validation),
        device.push_error_scope(ErrorFilter::OutOfMemory),
        device.push_error_scope(ErrorFilter::Internal),
    ]
}

fn finish_scopes(scopes: [ErrorScopeGuard; 3]) -> Result<(), String> {
    let mut errors = Vec::with_capacity(3);
    for scope in scopes.into_iter().rev() {
        if let Some(error) = futures_lite::future::block_on(scope.pop()) {
            errors.push(error.to_string());
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

impl CompactRasterResources {
    pub(super) fn new(
        p: &GpuPipeline,
        source: &str,
        cache: &PipelineCache,
        wg: [u32; 2],
        bytes: u64,
    ) -> Result<Self, String> {
        let generated = sources(source, wg)?;
        let scopes = error_scopes(&p.device);
        let prefix_buf = p.device.create_buffer(&BufferDescriptor {
            label: Some("compact_prefix"),
            size: bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let indirect_args_buf = p.device.create_buffer(&BufferDescriptor {
            label: Some("compact_indirect_args"),
            size: 12,
            usage: BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let (prefix_layout, raster_layout) = layouts(&p.device);
        let (prefix_bind_group, raster_bind_group) = bind_groups(
            p,
            &prefix_buf,
            &indirect_args_buf,
            &prefix_layout,
            &raster_layout,
        );
        let params_layout = p.rasterize_error_pipeline.get_bind_group_layout(1);
        let variant = compile_variant(
            &p.device,
            &prefix_layout,
            &raster_layout,
            &params_layout,
            cache,
            wg,
            generated,
        );
        finish_scopes(scopes)?;
        Ok(Self {
            prefix_buf,
            indirect_args_buf,
            prefix_layout,
            raster_layout,
            prefix_bind_group,
            raster_bind_group,
            prefix_pipeline: variant.prefix.clone(),
            raster_pipeline: variant.raster.clone(),
            prefix_source: variant.prefix_source.clone(),
            raster_source: variant.raster_source.clone(),
            wg,
            variants: vec![variant],
        })
    }

    pub(super) fn set_wg(
        &mut self,
        p: &Device,
        params: &BindGroupLayout,
        cache: &PipelineCache,
        source: &str,
        wg: [u32; 2],
    ) -> Result<(), String> {
        if wg == self.wg {
            return Ok(());
        }
        let index = if let Some(index) = self.variants.iter().position(|v| v.wg == wg) {
            index
        } else {
            if self.variants.len() == MAX_VARIANTS {
                return Err("compact variant cache is full".into());
            }
            let generated = sources(source, wg)?;
            let scopes = error_scopes(p);
            let variant = compile_variant(
                p,
                &self.prefix_layout,
                &self.raster_layout,
                params,
                cache,
                wg,
                generated,
            );
            finish_scopes(scopes)?;
            self.variants.push(variant);
            self.variants.len() - 1
        };
        let variant = &self.variants[index];
        self.prefix_pipeline = variant.prefix.clone();
        self.raster_pipeline = variant.raster.clone();
        self.prefix_source.clone_from(&variant.prefix_source);
        self.raster_source.clone_from(&variant.raster_source);
        self.wg = wg;
        Ok(())
    }
}

fn compile_variant(
    device: &Device,
    prefix: &BindGroupLayout,
    raster: &BindGroupLayout,
    params: &BindGroupLayout,
    cache: &PipelineCache,
    wg: [u32; 2],
    (prefix_source, raster_source): (String, String),
) -> CompactVariant {
    let prefix_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("compact_prefix_pipeline_layout"),
        bind_group_layouts: &[Some(prefix), Some(params)],
        immediate_size: 0,
    });
    let raster_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("compact_raster_pipeline_layout"),
        bind_group_layouts: &[Some(raster), Some(params)],
        immediate_size: 0,
    });
    let prefix = compile_shader(
        device,
        "compact_prefix",
        &prefix_layout,
        &prefix_source,
        cache,
    );
    let raster = compile_shader(
        device,
        "compact_raster",
        &raster_layout,
        &raster_source,
        cache,
    );
    CompactVariant {
        wg,
        prefix,
        raster,
        prefix_source,
        raster_source,
    }
}

fn compile_shader(
    device: &Device,
    label: &str,
    layout: &PipelineLayout,
    source: &str,
    cache: &PipelineCache,
) -> ComputePipeline {
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some(label),
        source: ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: PipelineCompilationOptions::default(),
        cache: Some(cache),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ORDINARY: &str = include_str!("../shaders/rasterize_error.wgsl");
    const WGS: [[u32; 2]; 5] = [[32, 16], [16, 16], [32, 8], [16, 8], [8, 8]];

    #[test]
    fn capacity_checks_bound_prefix_sum_padding_bytes_and_dispatch() {
        assert_eq!(
            prefix_bytes(256, [384, 384], [32, 16], 1028, 65535),
            Ok(1028)
        );
        for (n, image, wg, bytes, dims) in [
            (0, [384, 384], [32, 16], 1028, 65535),
            (256, [0, 384], [32, 16], 1028, 65535),
            (256, [384, 384], [4, 4], 1028, 65535),
            (256, [384, 384], [32, 16], 1027, 65535),
            (256, [384, 384], [32, 16], 1028, 65534),
            (u32::MAX, [1, 1], [8, 8], u64::MAX, 65535),
            (1, [u32::MAX, 1], [8, 8], u64::MAX, 65535),
            (65536, [65535, 65535], [8, 8], u64::MAX, 65535),
        ] {
            assert!(prefix_bytes(n, image, wg, bytes, dims).is_err());
        }
    }

    #[test]
    fn generated_sources_preserve_pixel_math_and_exact_specialization() {
        for wg in WGS {
            let (prefix, raster) = sources(ORDINARY, wg).unwrap();
            let ordinary = specialize_raster_source(ORDINARY, wg);
            let marker = "    var pixel_error = 0u;";
            let expected = ordinary[ordinary.find(marker).unwrap()..].replace(
                "    let tile_id = wid.y * num_tiles_x + wid.x;",
                "    let tile_id = shared_compact_y * num_tiles_x + shared_compact_x;",
            );
            assert_eq!(&raster[raster.find(marker).unwrap()..], expected);
            assert_eq!(
                raster.matches("workgroupBarrier();").count(),
                ordinary.matches("workgroupBarrier();").count()
            );
            assert!(prefix.contains(&format!("const WG_X: u32 = {};", wg[0])));
            assert!(prefix.contains(&format!("const WG_Y: u32 = {};", wg[1])));
            assert!(!prefix.contains("arrayLength"));
            for source in [&prefix, &raster] {
                let module = wgpu::naga::front::wgsl::parse_str(source)
                    .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
                wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap();
            }
        }
        for changed in [
            String::new(),
            format!("{ORDINARY}\n"),
            ORDINARY.replace(
                "    // Derive 2D pixel coordinates",
                "    // changed prologue",
            ),
            ORDINARY.replace("const WG_X: u32 = 16;", "const WG_X: u32 = 32;"),
        ] {
            assert!(sources(&changed, [32, 16]).is_err());
        }
    }

    // CPU oracles validate the proposed integer mapping against the original
    // strict overlap predicate. Actual shader coverage remains a separate GPU gate.
    fn axis(lo: u32, hi: u32, extent: u32, tile: u32) -> std::ops::Range<u32> {
        let start = lo / tile;
        let end = hi.div_ceil(tile).min(extent.div_ceil(tile));
        if lo >= extent || end <= start {
            0..0
        } else {
            start..end
        }
    }

    #[test]
    fn bbox_axes_equal_strict_overlap_including_zero_and_inverted_interiors() {
        for tile in [8, 16, 32] {
            for extent in [1u32, 17, 35, 384] {
                for lo in [0, 1, 7, 8, 15, 16, 17, 31, 32, 34, 35, 383, 384, 65535] {
                    for hi in [0, 1, 7, 8, 15, 16, 17, 31, 32, 34, 35, 383, 384, 65535] {
                        let expected: Vec<_> = (0..extent.div_ceil(tile))
                            .filter(|&i| (i * tile + tile).min(extent) > lo && i * tile < hi)
                            .collect();
                        assert_eq!(axis(lo, hi, extent, tile).collect::<Vec<_>>(), expected);
                    }
                }
            }
        }
        assert_eq!(axis(17, 17, 35, 16), 1..2);
        assert_eq!(axis(19, 17, 35, 16), 1..2);
        assert!(axis(35, 40, 35, 16).is_empty());
    }

    fn owner(prefix: &[u32], q: u32) -> usize {
        let mut lower = 0;
        let mut upper = prefix.len() - 1;
        for _ in 0..32 {
            if lower >= upper {
                break;
            }
            let middle = lower + (upper - lower) / 2;
            if prefix[middle + 1] > q {
                upper = middle;
            } else {
                lower = middle + 1;
            }
        }
        assert_eq!(lower, upper);
        lower
    }

    #[test]
    fn duplicate_prefixes_map_each_consumed_tile_once_and_skip_inactive_tail() {
        let prefix = [0, 0, 2, 2, 2, 5, 5];
        let mapped: Vec<_> = (0..5)
            .map(|q| {
                let i = owner(&prefix, q);
                (i, q - prefix[i])
            })
            .collect();
        assert_eq!(mapped, [(1, 0), (1, 1), (4, 0), (4, 1), (4, 2)]);
        assert_eq!(owner(&[0, 0, 0, 1, 1], 0), 2);
    }

    #[test]
    fn flat_grid_covers_65k_boundaries_with_only_uniform_padding() {
        for (total, expected) in [
            (0u32, [0, 1, 1]),
            (1, [1, 1, 1]),
            (65534, [65534, 1, 1]),
            (65535, [65535, 1, 1]),
            (65536, [32768, 2, 1]),
            (65537, [32769, 2, 1]),
            (73728, [36864, 2, 1]),
        ] {
            let y = total.div_ceil(65535).max(1);
            let x = total.div_ceil(y);
            assert_eq!([x, y, 1], expected);
            let consumed: Vec<_> = (0..y)
                .flat_map(|row| (0..x).map(move |col| col + x * row))
                .filter(|&q| q < total)
                .collect();
            assert_eq!(consumed, (0..total).collect::<Vec<_>>());
        }
    }
}
