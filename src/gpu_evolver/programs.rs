//! Own shader identity, stage-specific compilation and compatible installed programs.
//! Storage, binding resources and command scheduling stay in `GpuPipeline`.
//! Callers drain submitted work before installing a different program variant.
//! Production installs the bundled sources; benchmarks install shader variants
//! through the `variants` seam, which is empty in production.

mod variants;

use std::path::PathBuf;

use wgpu::{
    AdapterInfo, Backend, BindGroup, BindGroupLayout, ComputePipeline,
    ComputePipelineDescriptor, Device, ErrorFilter, ErrorScopeGuard, PipelineCache,
    PipelineCacheDescriptor, PipelineCompilationOptions, PipelineLayout,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource,
};

use variants::{BenchmarkMutateKey, ShaderVariants};
pub(super) use variants::WorklistStorage;
pub use variants::{
    benchmark_mutate_compilation_options, benchmark_mutate_group_size_from_source,
    BenchmarkMutateOptions, ShaderVariantInstaller,
};

const MUTATE_SOURCE: &str = include_str!("../shaders/mutate.wgsl");
const RASTER_SOURCE: &str = include_str!("../shaders/rasterize_error.wgsl");
const BIN_SOURCE: &str = include_str!("../shaders/bin_polygons.wgsl");

// Bundled policy is separate from the historical-source default.
const BUNDLED_MUTATE_OPTIONS: BenchmarkMutateOptions = BenchmarkMutateOptions {
    zero_initialize_workgroup_memory: true,
};

#[derive(Clone)]
pub(super) struct MutationProgram {
    key: BenchmarkMutateKey,
    pub(super) pipeline: ComputePipeline,
}

impl MutationProgram {
    pub(super) fn group_size(&self) -> u32 {
        self.key.group_size
    }

    pub(super) fn options(&self) -> BenchmarkMutateOptions {
        self.key.options
    }

    pub(super) fn matches(&self, source: &str, options: BenchmarkMutateOptions) -> bool {
        self.key
            .matches_installed(source, self.key.group_size, options)
    }
}

pub(super) struct ProgramBindings<'a> {
    pub params: &'a BindGroupLayout,
    pub mutate: &'a BindGroupLayout,
    pub raster: &'a BindGroupLayout,
    pub bin: &'a BindGroupLayout,
    pub select: &'a BindGroupLayout,
    pub init: &'a BindGroupLayout,
    pub update: &'a BindGroupLayout,
}

struct ProgramLayouts {
    mutate: PipelineLayout,
    raster: PipelineLayout,
    bin: PipelineLayout,
    select: PipelineLayout,
    init: PipelineLayout,
    update: PipelineLayout,
}

impl ProgramLayouts {
    fn new(device: &Device, bindings: &ProgramBindings<'_>) -> Self {
        Self {
            mutate: stage_layout(device, "mutate_layout", bindings.mutate, bindings.params),
            raster: stage_layout(device, "rasterize_error_layout", bindings.raster, bindings.params),
            bin: stage_layout(device, "bin_polygons_layout", bindings.bin, bindings.params),
            select: stage_layout(device, "select_layout", bindings.select, bindings.params),
            init: stage_layout(device, "init_framebuffers_layout", bindings.init, bindings.params),
            update: stage_layout(device, "update_framebuffers_layout", bindings.update, bindings.params),
        }
    }
}

fn stage_layout(
    device: &Device,
    label: &str,
    stage: &BindGroupLayout,
    params: &BindGroupLayout,
) -> PipelineLayout {
    device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(stage), Some(params)],
        immediate_size: 0,
    })
}

struct ProgramCompiler {
    device: Device,
    layouts: ProgramLayouts,
    cache: PipelineCache,
    cache_path: Option<PathBuf>,
}

impl ProgramCompiler {
    fn new(device: &Device, adapter: &AdapterInfo, bindings: &ProgramBindings<'_>) -> Self {
        let cache_path = pipeline_cache_path(adapter);
        let cache_data = cache_path
            .as_ref()
            .and_then(|path| std::fs::read(path).ok());
        // SAFETY: data comes from this adapter's previous run, or is absent.
        // Fallback creates a fresh cache if the persisted bytes are invalid.
        let cache = unsafe {
            device.create_pipeline_cache(&PipelineCacheDescriptor {
                label: Some("gpu_evolver_cache"),
                data: cache_data.as_deref(),
                fallback: true,
            })
        };
        if cache_data.is_some() {
            println!("Loaded pipeline cache from disk");
        }
        Self {
            device: device.clone(),
            layouts: ProgramLayouts::new(device, bindings),
            cache,
            cache_path,
        }
    }

    fn mutate(&self, source: &str, options: BenchmarkMutateOptions) -> ComputePipeline {
        let shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("mutate_shader"),
            source: ShaderSource::Wgsl(source.into()),
        });
        self.device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("mutate_pipeline"),
                layout: Some(&self.layouts.mutate),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: benchmark_mutate_compilation_options(options),
                cache: Some(&self.cache),
            })
    }

    fn select(&self) -> ComputePipeline {
        let shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("select_shader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/select.wgsl").into()),
        });
        self.device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("select_pipeline"),
                layout: Some(&self.layouts.select),
                module: &shader,
                entry_point: Some("select_main"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: Some(&self.cache),
            })
    }

    /// Compile an already specialized raster/error source.
    fn raster(&self, source: &str) -> ComputePipeline {
        self.tile_stage("rasterize_error", &self.layouts.raster, source)
    }

    /// Compile an already specialized binning source.
    fn bin(&self, source: &str) -> ComputePipeline {
        self.tile_stage("bin_polygons", &self.layouts.bin, source)
    }

    fn bundled_stages(&self, wg: [u32; 2]) -> TileStages {
        TileStages {
            bin: self.bin(&specialize_bin_source(BIN_SOURCE, wg)),
            raster: self.raster(&specialize_raster_source(RASTER_SOURCE, wg)),
            schedule: RasterSchedule::ListedByBinner,
        }
    }

    /// Complete a tile group with the bundled framebuffer stages for `wg`.
    fn tiles(&self, wg: [u32; 2], stages: TileStages) -> TilePrograms {
        TilePrograms {
            wg,
            raster: stages.raster,
            bin: stages.bin,
            schedule: stages.schedule,
            init_framebuffers_pipeline: self.tile_stage(
                "init_framebuffers",
                &self.layouts.init,
                &specialize_raster_source(include_str!("../shaders/init_framebuffers.wgsl"), wg),
            ),
            update_framebuffers_pipeline: self.tile_stage(
                "update_framebuffers",
                &self.layouts.update,
                &specialize_raster_source(include_str!("../shaders/update_framebuffers.wgsl"), wg),
            ),
        }
    }

    fn tile_stage(&self, label: &str, layout: &PipelineLayout, source: &str) -> ComputePipeline {
        create_tile_pipeline(&self.device, layout, label, source, &self.cache)
    }
}

/// How `rasterize_error` finds the (offspring, tile) pairs it must run. Decided
/// when a program is installed; command recording only matches on it.
#[derive(Clone)]
pub(super) enum RasterSchedule {
    /// The binner appends the list and raster loops over it (bundled programs).
    ListedByBinner,
    /// A shader-variant binner lists nothing, so a separate pass lists the pairs
    /// with the same test before raster loops over them.
    ListedByPass {
        pipeline: ComputePipeline,
        bind_group: BindGroup,
    },
    /// A shader-variant raster indexes tiles by workgroup id over the full grid.
    FullGrid,
}

/// The raster and binning programs of one tile group, before framebuffer stages.
struct TileStages {
    raster: ComputePipeline,
    bin: ComputePipeline,
    schedule: RasterSchedule,
}

// A tile transition replaces this whole value only after every stage succeeds.
// Shader variants and diagnostic mode always describe these installed handles.
pub(super) struct TilePrograms {
    pub wg: [u32; 2],
    pub raster: ComputePipeline,
    pub bin: ComputePipeline,
    pub schedule: RasterSchedule,
    pub init_framebuffers_pipeline: ComputePipeline,
    pub update_framebuffers_pipeline: ComputePipeline,
}

pub(super) struct GpuPrograms {
    compiler: ProgramCompiler,
    mutation: MutationProgram,
    tiles: TilePrograms,
    select: ComputePipeline,
    variants: ShaderVariants,
}

impl GpuPrograms {
    pub(super) fn new(
        device: &Device,
        adapter: &AdapterInfo,
        bindings: &ProgramBindings<'_>,
        wg: [u32; 2],
    ) -> Self {
        let compiler = ProgramCompiler::new(device, adapter, bindings);
        let key = BenchmarkMutateKey::new(MUTATE_SOURCE, BUNDLED_MUTATE_OPTIONS)
            .expect("bundled mutation source must have valid group metadata");
        let mutation = MutationProgram {
            pipeline: compiler.mutate(&key.source, key.options),
            key,
        };
        let tiles = compiler.tiles(wg, compiler.bundled_stages(wg));
        Self {
            mutation,
            tiles,
            select: compiler.select(),
            variants: ShaderVariants::new(device, bindings.params),
            compiler,
        }
    }

    pub(super) fn mutation(&self) -> &MutationProgram {
        &self.mutation
    }

    pub(super) fn tiles(&self) -> &TilePrograms {
        &self.tiles
    }

    pub(super) fn select(&self) -> &ComputePipeline {
        &self.select
    }

    pub(super) fn try_set_rasterize_wg(&mut self, wg: [u32; 2]) -> Result<(), String> {
        if wg == self.tiles.wg {
            return Ok(());
        }
        // The outer scopes also check bundled raster/init/update compilation.
        // Pop every scope even when a candidate cache or compilation rejects.
        let scopes = compilation_scopes(&self.compiler.device);
        let prepared = self.prepare_tiles(wg);
        let compilation = finish_compilation(scopes);
        let tiles = prepared?;
        compilation?;
        println!(
            "Recreating rasterize_error + bin_polygons pipelines: {}x{} -> {}x{}",
            self.tiles.wg[0], self.tiles.wg[1], wg[0], wg[1]
        );
        self.tiles = tiles;
        Ok(())
    }

    fn prepare_tiles(&mut self, wg: [u32; 2]) -> Result<TilePrograms, String> {
        let stages = match self.variants.tile_stages(&self.compiler, wg) {
            Some(stages) => stages?,
            None => self.compiler.bundled_stages(wg),
        };
        Ok(self.compiler.tiles(wg, stages))
    }

    /// Save pipeline cache data to disk for faster startup next time.
    pub(super) fn save_pipeline_cache(&self) {
        let Some(ref path) = self.compiler.cache_path else {
            return;
        };
        let Some(data) = self.compiler.cache.get_data() else {
            return;
        };
        // Atomic write: write to temp file, then rename
        let tmp = path.with_extension("bin.tmp");
        if let Err(e) = std::fs::write(&tmp, &data) {
            eprintln!("Failed to write pipeline cache: {e}");
            return;
        }
        if let Err(e) = std::fs::rename(&tmp, path) {
            eprintln!("Failed to rename pipeline cache: {e}");
            // Best-effort cleanup of an uninstalled file; the live cache is unchanged.
            let _ = std::fs::remove_file(&tmp);
            return;
        }
        println!("Saved pipeline cache ({} bytes)", data.len());
    }
}

fn compilation_scopes(device: &Device) -> [ErrorScopeGuard; 3] {
    [
        device.push_error_scope(ErrorFilter::Validation),
        device.push_error_scope(ErrorFilter::OutOfMemory),
        device.push_error_scope(ErrorFilter::Internal),
    ]
}

// Pipeline compilation and scope completion use wgpu's blocking driver path.
// The lab's parent process timeout bounds a stalled driver, as for GPU drain.
fn finish_compilation(scopes: [ErrorScopeGuard; 3]) -> Result<(), String> {
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

/// Create a tile-geometry compute pipeline (`main` entry point) from a source
/// already specialized for its workgroup. The labels name the stage.
fn create_tile_pipeline(
    device: &Device,
    layout: &PipelineLayout,
    label: &str,
    source: &str,
    cache: &PipelineCache,
) -> ComputePipeline {
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some(&format!("{label}_shader")),
        source: ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(&format!("{label}_pipeline")),
        layout: Some(layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: rasterize_compilation_options(device),
        cache: Some(cache),
    })
}

/// Specialize the raster tile size of raster, worklist and framebuffer sources.
/// Uses string replacement on the shader source since naga does not support
/// override constants in `@workgroup_size` or const expressions.
fn specialize_raster_source(source: &str, wg: [u32; 2]) -> String {
    source
        .replace(
            "const WG_X: u32 = 16;",
            &format!("const WG_X: u32 = {};", wg[0]),
        )
        .replace(
            "const WG_Y: u32 = 16;",
            &format!("const WG_Y: u32 = {};", wg[1]),
        )
}

/// Specialize binning tiles to match the rasterize workgroup.
fn specialize_bin_source(source: &str, wg: [u32; 2]) -> String {
    source
        .replace(
            "const TILE_W: u32 = 16;",
            &format!("const TILE_W: u32 = {};", wg[0]),
        )
        .replace(
            "const TILE_H: u32 = 16;",
            &format!("const TILE_H: u32 = {};", wg[1]),
        )
}

// Only the rasterize/error, bin, init-framebuffers, and update-framebuffers
// shaders use these options. Every shared value they read is explicitly written
// before a workgroup barrier; unused polygon/reduction/mask slots are never read.
// Most bin workgroups return before touching shared memory, so zeroing would be
// their main cost. Avoid the
// redundant zeroing code, which is expensive to compile on the WSL Dozen driver.
// Keep default initialization for mutation: empty-parent crossover reads slot 0.
// Vulkan provides native subgroup IDs. Naga's DX12 translation infers them from
// local indices, so retain default initialization on that backend.
fn rasterize_compilation_options(device: &Device) -> PipelineCompilationOptions<'static> {
    PipelineCompilationOptions {
        zero_initialize_workgroup_memory: device.adapter_info().backend != Backend::Vulkan,
        ..Default::default()
    }
}

/// Get the disk path for storing this adapter's pipeline cache.
/// Returns `None` if the backend doesn't support pipeline caching.
fn pipeline_cache_path(adapter_info: &AdapterInfo) -> Option<PathBuf> {
    let key = wgpu::util::pipeline_cache_key(adapter_info)?;
    Some(PathBuf::from(format!("pipeline_cache_{key}.bin")))
}
