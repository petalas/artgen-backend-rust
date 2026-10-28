//! Own shader identity, stage-specific compilation and compatible installed programs.
//! Storage, binding resources and command scheduling stay in `GpuPipeline`.
//! Callers drain submitted work before installing a different program variant.

use std::path::PathBuf;
use std::sync::Arc;

use wgpu::{
    AdapterInfo, Backend, BindGroupLayout, ComputePipeline, ComputePipelineDescriptor, Device,
    ErrorFilter, ErrorScopeGuard, PipelineCache, PipelineCacheDescriptor,
    PipelineCompilationOptions, PipelineLayout, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource,
};

// Capacity for 17 lab sources, five supported raster workgroups, and two modes.
const MAX_BENCHMARK_BIN_PIPELINES: usize = 17 * 5 * 2;
const MAX_BENCHMARK_SOURCE_BYTES: usize = 256 * 1024;
const MAX_BENCHMARK_RASTER_PIPELINES: usize = 17 * 5;
const MAX_BENCHMARK_MUTATE_PIPELINES: usize = 17;

/// Mutation-only compiler policy. Historical benchmark sources require zeroing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct BenchmarkMutateOptions {
    pub zero_initialize_workgroup_memory: bool,
}

impl Default for BenchmarkMutateOptions {
    fn default() -> Self {
        Self {
            zero_initialize_workgroup_memory: true,
        }
    }
}

// Bundled policy is separate from the historical-source default.
const BUNDLED_MUTATE_OPTIONS: BenchmarkMutateOptions = BenchmarkMutateOptions {
    zero_initialize_workgroup_memory: true,
};

/// Shared by the mutation factory and the direct numerical checker.
#[must_use]
pub fn benchmark_mutate_compilation_options(
    options: BenchmarkMutateOptions,
) -> PipelineCompilationOptions<'static> {
    PipelineCompilationOptions {
        zero_initialize_workgroup_memory: options.zero_initialize_workgroup_memory,
        ..PipelineCompilationOptions::default()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BenchmarkMutateKey {
    source: Arc<str>,
    group_size: u32,
    options: BenchmarkMutateOptions,
}

impl BenchmarkMutateKey {
    fn new(source: &str, options: BenchmarkMutateOptions) -> Result<Self, String> {
        let group_size = benchmark_mutate_group_size_from_source(source)?;
        Ok(Self {
            source: Arc::from(source),
            group_size,
            options,
        })
    }

    fn matches_installed(
        &self,
        source: &str,
        group_size: u32,
        options: BenchmarkMutateOptions,
    ) -> bool {
        self.source.as_ref() == source && self.group_size == group_size && self.options == options
    }
}

#[derive(Clone)]
pub(super) struct MutationProgram {
    key: BenchmarkMutateKey,
    pub(super) pipeline: ComputePipeline,
}

#[derive(Debug, PartialEq, Eq)]
struct BenchmarkRasterKey {
    source: Arc<str>,
    wg: [u32; 2],
}

struct BenchmarkRasterPipeline {
    key: BenchmarkRasterKey,
    pipeline: ComputePipeline,
    shader: ShaderModule,
}

struct BenchmarkBinPipeline {
    source: Arc<str>,
    wg: [u32; 2],
    validation: bool,
    pipeline: ComputePipeline,
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
        let layout = |label, stage| {
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(stage), Some(bindings.params)],
                immediate_size: 0,
            })
        };
        Self {
            mutate: layout("mutate_layout", bindings.mutate),
            raster: layout("rasterize_error_layout", bindings.raster),
            bin: layout("bin_polygons_layout", bindings.bin),
            select: layout("select_layout", bindings.select),
            init: layout("init_framebuffers_layout", bindings.init),
            update: layout("update_framebuffers_layout", bindings.update),
        }
    }
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

    fn bundled_tiles(&self, wg: [u32; 2]) -> TilePrograms {
        let (pipeline, shader) =
            create_rasterize_pipeline(&self.device, &self.layouts.raster, wg, &self.cache);
        let bin = create_bin_polygons_pipeline(&self.device, &self.layouts.bin, wg, &self.cache);
        TilePrograms {
            wg,
            raster: RasterProgram {
                pipeline,
                shader,
                source: None,
            },
            bin: BinProgram {
                pipeline: bin,
                source: None,
                validation: false,
            },
            init_framebuffers_pipeline: create_init_framebuffers_pipeline(
                &self.device,
                &self.layouts.init,
                wg,
                &self.cache,
            ),
            update_framebuffers_pipeline: create_update_framebuffers_pipeline(
                &self.device,
                &self.layouts.update,
                wg,
                &self.cache,
            ),
        }
    }
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

pub(super) struct RasterProgram {
    pub pipeline: ComputePipeline,
    shader: ShaderModule,
    source: Option<Arc<str>>,
}

pub(super) struct BinProgram {
    pub pipeline: ComputePipeline,
    source: Option<Arc<str>>,
    pub validation: bool,
}

// A tile transition replaces this whole value only after every stage succeeds.
// Source overrides and diagnostic mode always describe these installed handles.
pub(super) struct TilePrograms {
    pub wg: [u32; 2],
    pub raster: RasterProgram,
    pub bin: BinProgram,
    pub init_framebuffers_pipeline: ComputePipeline,
    pub update_framebuffers_pipeline: ComputePipeline,
}

pub(super) struct GpuPrograms {
    compiler: ProgramCompiler,
    mutation: MutationProgram,
    tiles: TilePrograms,
    select: ComputePipeline,
    // Empty in production. Preserve exact source/tile/policy identities and limits.
    benchmark_mutate_pipelines: Vec<MutationProgram>,
    benchmark_raster_pipelines: Vec<BenchmarkRasterPipeline>,
    benchmark_bin_pipelines: Vec<BenchmarkBinPipeline>,
}

impl GpuPrograms {
    pub(super) fn new(
        device: &Device,
        adapter: &AdapterInfo,
        bindings: &ProgramBindings<'_>,
        wg: [u32; 2],
    ) -> Self {
        let compiler = ProgramCompiler::new(device, adapter, bindings);
        let key = BenchmarkMutateKey::new(
            include_str!("../shaders/mutate.wgsl"),
            BUNDLED_MUTATE_OPTIONS,
        )
        .expect("bundled mutation source must have valid group metadata");
        let mutation = MutationProgram {
            pipeline: create_mutate_pipeline_from_source(
                device,
                &compiler.layouts.mutate,
                &key.source,
                key.options,
                &compiler.cache,
            ),
            key,
        };
        let tiles = compiler.bundled_tiles(wg);
        let select = create_select_pipeline(device, &compiler.layouts.select, &compiler.cache);
        Self {
            compiler,
            mutation,
            tiles,
            select,
            benchmark_mutate_pipelines: Vec::new(),
            benchmark_raster_pipelines: Vec::new(),
            benchmark_bin_pipelines: Vec::new(),
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
        let bin = self.prepare_bin(wg)?;
        let raster = self.prepare_raster(wg)?;
        Ok(TilePrograms {
            wg,
            raster,
            bin,
            init_framebuffers_pipeline: create_init_framebuffers_pipeline(
                &self.compiler.device,
                &self.compiler.layouts.init,
                wg,
                &self.compiler.cache,
            ),
            update_framebuffers_pipeline: create_update_framebuffers_pipeline(
                &self.compiler.device,
                &self.compiler.layouts.update,
                wg,
                &self.compiler.cache,
            ),
        })
    }

    fn prepare_bin(&mut self, wg: [u32; 2]) -> Result<BinProgram, String> {
        let bin_source = self.tiles.bin.source.clone();
        let validation = self.tiles.bin.validation;
        let pipeline = if let Some(source) = bin_source.clone() {
            self.cached_benchmark_bin(source, wg, validation)?
        } else if validation {
            self.cached_benchmark_bin(
                Arc::from(include_str!("../shaders/bin_polygons.wgsl")),
                wg,
                true,
            )?
        } else {
            create_bin_polygons_pipeline(
                &self.compiler.device,
                &self.compiler.layouts.bin,
                wg,
                &self.compiler.cache,
            )
        };
        Ok(BinProgram {
            pipeline,
            source: bin_source,
            validation,
        })
    }

    fn prepare_raster(&mut self, wg: [u32; 2]) -> Result<RasterProgram, String> {
        let source = self.tiles.raster.source.clone();
        let (pipeline, shader) = if let Some(source) = source.clone() {
            self.cached_benchmark_raster(source, wg)?
        } else {
            create_rasterize_pipeline(
                &self.compiler.device,
                &self.compiler.layouts.raster,
                wg,
                &self.compiler.cache,
            )
        };
        Ok(RasterProgram {
            pipeline,
            shader,
            source,
        })
    }

    pub(super) fn set_benchmark_mutate_shader(
        &mut self,
        source: &str,
        options: BenchmarkMutateOptions,
    ) -> Result<(), String> {
        let key = BenchmarkMutateKey::new(source, options)?;
        if let Some(entry) = self
            .benchmark_mutate_pipelines
            .iter()
            .find(|entry| entry.key == key)
        {
            self.mutation = entry.clone();
            return Ok(());
        }
        if self.benchmark_mutate_pipelines.len() == MAX_BENCHMARK_MUTATE_PIPELINES {
            return Err("benchmark mutate pipeline cache is full".into());
        }
        let pipeline = if self.benchmark_mutate_pipelines.is_empty()
            && self.mutation.matches(source, options)
        {
            self.mutation.pipeline.clone()
        } else {
            self.compile_benchmark_mutate(source, options)?
        };
        let selected = MutationProgram { key, pipeline };
        self.benchmark_mutate_pipelines.push(selected.clone());
        self.mutation = selected;
        Ok(())
    }
    fn compile_benchmark_mutate(
        &self,
        source: &str,
        options: BenchmarkMutateOptions,
    ) -> Result<ComputePipeline, String> {
        let scopes = compilation_scopes(&self.compiler.device);
        let pipeline = create_mutate_pipeline_from_source(
            &self.compiler.device,
            &self.compiler.layouts.mutate,
            source,
            options,
            &self.compiler.cache,
        );
        finish_compilation(scopes)?;
        Ok(pipeline)
    }

    pub(super) fn set_benchmark_raster_shader(&mut self, source: &str) -> Result<(), String> {
        validate_benchmark_raster_source(source)?;
        let source = self
            .benchmark_raster_pipelines
            .iter()
            .find(|entry| entry.key.source.as_ref() == source)
            .map_or_else(|| Arc::from(source), |entry| Arc::clone(&entry.key.source));
        let (pipeline, shader) =
            self.cached_benchmark_raster(Arc::clone(&source), self.tiles.wg)?;
        self.tiles.raster = RasterProgram {
            pipeline,
            shader,
            source: Some(source),
        };
        Ok(())
    }

    fn cached_benchmark_raster(
        &mut self,
        source: Arc<str>,
        wg: [u32; 2],
    ) -> Result<(ComputePipeline, ShaderModule), String> {
        let key = BenchmarkRasterKey { source, wg };
        if let Some(entry) = self
            .benchmark_raster_pipelines
            .iter()
            .find(|entry| entry.key == key)
        {
            return Ok((entry.pipeline.clone(), entry.shader.clone()));
        }
        if self.benchmark_raster_pipelines.len() == MAX_BENCHMARK_RASTER_PIPELINES {
            return Err("benchmark raster pipeline cache is full".into());
        }
        let (pipeline, shader) = if self.benchmark_raster_pipelines.is_empty()
            && self.tiles.raster.source.is_none()
            && wg == self.tiles.wg
            && key.source.as_ref() == include_str!("../shaders/rasterize_error.wgsl")
        {
            (
                self.tiles.raster.pipeline.clone(),
                self.tiles.raster.shader.clone(),
            )
        } else {
            self.compile_benchmark_raster(&key.source, wg)?
        };
        self.benchmark_raster_pipelines
            .push(BenchmarkRasterPipeline {
                key,
                pipeline: pipeline.clone(),
                shader: shader.clone(),
            });
        Ok((pipeline, shader))
    }

    fn compile_benchmark_raster(
        &self,
        source: &str,
        wg: [u32; 2],
    ) -> Result<(ComputePipeline, ShaderModule), String> {
        let specialized = specialize_raster_source(source, wg);
        let scopes = compilation_scopes(&self.compiler.device);
        let compiled = create_rasterize_pipeline_from_source(
            &self.compiler.device,
            &self.compiler.layouts.raster,
            &specialized,
            &self.compiler.cache,
        );
        finish_compilation(scopes)?;
        Ok(compiled)
    }

    /// Install a binning source through the engine's specialization and pipeline cache.
    ///
    /// The caller must own this pipeline exclusively and drain submitted GPU work
    /// before calling. The source is retained across raster workgroup changes;
    /// compilation uses the current workgroup and benchmark validation setting.
    ///
    /// # Errors
    /// Returns an error for invalid source bounds or specialization markers, a
    /// full benchmark cache, or GPU pipeline compilation failure. The installed
    /// source and active pipeline remain unchanged on error.
    pub(super) fn set_benchmark_bin_shader(&mut self, source: &str) -> Result<(), String> {
        validate_benchmark_bin_source(source)?;
        let source = self
            .benchmark_bin_pipelines
            .iter()
            .find(|entry| entry.source.as_ref() == source)
            .map_or_else(|| Arc::from(source), |entry| Arc::clone(&entry.source));
        let pipeline = self.cached_benchmark_bin(
            Arc::clone(&source),
            self.tiles.wg,
            self.tiles.bin.validation,
        )?;
        self.tiles.bin = BinProgram {
            pipeline,
            source: Some(source),
            validation: self.tiles.bin.validation,
        };
        Ok(())
    }

    /// Instrument the actual installed source, including after workgroup changes.
    pub(super) fn set_benchmark_validation(&mut self, enabled: bool) -> Result<(), String> {
        if enabled == self.tiles.bin.validation {
            return Ok(());
        }
        let source = self
            .tiles
            .bin
            .source
            .clone()
            .unwrap_or_else(|| Arc::from(include_str!("../shaders/bin_polygons.wgsl")));
        let pipeline = self.cached_benchmark_bin(source, self.tiles.wg, enabled)?;
        self.tiles.bin = BinProgram {
            pipeline,
            source: self.tiles.bin.source.clone(),
            validation: enabled,
        };
        Ok(())
    }

    fn cached_benchmark_bin(
        &mut self,
        source: Arc<str>,
        wg: [u32; 2],
        validation: bool,
    ) -> Result<ComputePipeline, String> {
        if let Some(entry) = self.benchmark_bin_pipelines.iter().find(|entry| {
            entry.source == source && entry.wg == wg && entry.validation == validation
        }) {
            return Ok(entry.pipeline.clone());
        }
        if self.benchmark_bin_pipelines.len() == MAX_BENCHMARK_BIN_PIPELINES {
            return Err("benchmark bin pipeline cache is full".into());
        }
        let pipeline = if self.benchmark_bin_pipelines.is_empty()
            && self.tiles.bin.source.is_none()
            && !self.tiles.bin.validation
            && !validation
            && wg == self.tiles.wg
            && source.as_ref() == include_str!("../shaders/bin_polygons.wgsl")
        {
            self.tiles.bin.pipeline.clone()
        } else {
            self.compile_benchmark_bin(&source, wg, validation)?
        };
        self.benchmark_bin_pipelines.push(BenchmarkBinPipeline {
            source,
            wg,
            validation,
            pipeline: pipeline.clone(),
        });
        Ok(pipeline)
    }

    // Pipeline compilation and scope completion use wgpu's blocking driver path.
    // The lab's parent process timeout bounds a stalled driver, as for GPU drain.
    fn compile_benchmark_bin(
        &self,
        source: &str,
        wg: [u32; 2],
        validation: bool,
    ) -> Result<ComputePipeline, String> {
        let specialized = specialize_benchmark_bin_source(source, wg, validation)?;
        let scopes = compilation_scopes(&self.compiler.device);
        let pipeline = create_bin_pipeline_from_source(
            &self.compiler.device,
            &self.compiler.layouts.bin,
            &specialized,
            &self.compiler.cache,
        );
        finish_compilation(scopes)?;
        Ok(pipeline)
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

fn create_select_pipeline(
    device: &Device,
    layout: &PipelineLayout,
    cache: &PipelineCache,
) -> ComputePipeline {
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("select_shader"),
        source: ShaderSource::Wgsl(include_str!("../shaders/select.wgsl").into()),
    });
    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("select_pipeline"),
        layout: Some(layout),
        module: &shader,
        entry_point: Some("select_main"),
        compilation_options: PipelineCompilationOptions::default(),
        cache: Some(cache),
    })
}

fn create_mutate_pipeline_from_source(
    device: &Device,
    layout: &PipelineLayout,
    source: &str,
    options: BenchmarkMutateOptions,
    cache: &PipelineCache,
) -> ComputePipeline {
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("mutate_shader"),
        source: ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("mutate_pipeline"),
        layout: Some(layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: benchmark_mutate_compilation_options(options),
        cache: Some(cache),
    })
}

fn validate_benchmark_mutate_source(source: &str) -> Result<(), String> {
    if source.is_empty() || source.len() > MAX_BENCHMARK_SOURCE_BYTES {
        return Err("mutate shader must contain 1..262144 bytes".into());
    }
    Ok(())
}

/// Read the exact logical offspring width without compiling or rewriting WGSL.
/// Marker-free historical sources use 64, independently of physical lane count.
///
/// # Errors
/// Rejects oversized/empty sources and missing, duplicate or nonliteral markers.
pub fn benchmark_mutate_group_size_from_source(source: &str) -> Result<u32, String> {
    const NAME: &str = "OFFSPRING_GROUP_SIZE";
    const PREFIX: &str = "const OFFSPRING_GROUP_SIZE: u32 = ";
    validate_benchmark_mutate_source(source)?;
    if !source.contains(NAME) {
        return Ok(64);
    }
    let mut group_size = None;
    let mut offset = 0;
    let mut previous = (0, "");
    while let Some((start, token)) = next_mutate_source_token(source, &mut offset)? {
        if token == NAME && previous.1 == "const" {
            if group_size.is_some() {
                return Err("duplicate OFFSPRING_GROUP_SIZE declaration".into());
            }
            let declaration = source[previous.0..]
                .split_once(';')
                .map(|(declaration, _)| declaration)
                .ok_or("missing OFFSPRING_GROUP_SIZE declaration terminator")?;
            let literal = declaration
                .strip_prefix(PREFIX)
                .and_then(|s| s.strip_suffix('u'));
            group_size = Some(match literal {
                Some("1") => 1,
                Some("2") => 2,
                Some("4") => 4,
                Some("8") => 8,
                Some("16") => 16,
                Some("32") => 32,
                Some("64") => 64,
                _ => {
                    return Err(
                        "OFFSPRING_GROUP_SIZE must be an exact literal power of two in 1..64"
                            .into(),
                    )
                }
            });
        }
        previous = (start, token);
    }
    group_size
        .ok_or_else(|| "OFFSPRING_GROUP_SIZE is present without its exact const declaration".into())
}

// Each cursor advance consumes at least one source byte, bounded by the 256 KiB
// source cap. Comments cannot supply a declaration; nested comments use no stack.
fn next_mutate_source_token<'a>(
    source: &'a str,
    offset: &mut usize,
) -> Result<Option<(usize, &'a str)>, String> {
    let bytes = source.as_bytes();
    while *offset < bytes.len() {
        let tail = &bytes[*offset..];
        if tail[0].is_ascii_whitespace() {
            *offset += 1;
        } else if tail.starts_with(b"//") {
            *offset += tail.iter().position(|&b| b == b'\n').unwrap_or(tail.len());
        } else if tail.starts_with(b"/*") {
            *offset += 2;
            let mut depth = 1;
            while *offset < bytes.len() && depth > 0 {
                let tail = &bytes[*offset..];
                if tail.starts_with(b"/*") {
                    depth += 1;
                    *offset += 2;
                } else if tail.starts_with(b"*/") {
                    depth -= 1;
                    *offset += 2;
                } else {
                    *offset += 1;
                }
            }
            if depth != 0 {
                return Err("unterminated comment in mutation group metadata".into());
            }
        } else {
            let start = *offset;
            let ch = source[start..]
                .chars()
                .next()
                .expect("cursor is within source");
            *offset += ch.len_utf8();
            if ch.is_ascii_alphanumeric() || ch == '_' {
                while *offset < bytes.len()
                    && (bytes[*offset].is_ascii_alphanumeric() || bytes[*offset] == b'_')
                {
                    *offset += 1;
                }
            }
            return Ok(Some((start, &source[start..*offset])));
        }
    }
    Ok(None)
}

/// Create a `rasterize_error` compute pipeline with the given workgroup size.
/// Uses string replacement on the shader source since naga does not support
/// override constants in `@workgroup_size` or const expressions.
fn create_rasterize_pipeline(
    device: &Device,
    layout: &PipelineLayout,
    wg: [u32; 2],
    cache: &PipelineCache,
) -> (ComputePipeline, ShaderModule) {
    let source = specialize_raster_source(include_str!("../shaders/rasterize_error.wgsl"), wg);
    create_rasterize_pipeline_from_source(device, layout, &source, cache)
}

fn create_rasterize_pipeline_from_source(
    device: &Device,
    layout: &PipelineLayout,
    source: &str,
    cache: &PipelineCache,
) -> (ComputePipeline, ShaderModule) {
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("rasterize_error_shader"),
        source: ShaderSource::Wgsl(source.into()),
    });

    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("rasterize_error_pipeline"),
        layout: Some(layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: rasterize_compilation_options(device),
        cache: Some(cache),
    });

    (pipeline, shader)
}

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

fn validate_benchmark_raster_source(source: &str) -> Result<(), String> {
    if source.is_empty() || source.len() > MAX_BENCHMARK_SOURCE_BYTES {
        return Err("raster shader must contain 1..262144 bytes".into());
    }
    for marker in ["const WG_X: u32 = 16;", "const WG_Y: u32 = 16;"] {
        if source.matches(marker).count() != 1 {
            return Err(format!(
                "raster specialization must match {marker:?} exactly once"
            ));
        }
    }
    Ok(())
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

/// Create a `bin_polygons` compute pipeline with tile size matching the rasterize workgroup.
fn create_bin_polygons_pipeline(
    device: &Device,
    layout: &PipelineLayout,
    wg: [u32; 2],
    cache: &PipelineCache,
) -> ComputePipeline {
    let source = bin_polygons_source(wg);
    create_bin_pipeline_from_source(device, layout, &source, cache)
}

fn create_bin_pipeline_from_source(
    device: &Device,
    layout: &PipelineLayout,
    source: &str,
    cache: &PipelineCache,
) -> ComputePipeline {
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("bin_polygons_shader"),
        source: ShaderSource::Wgsl(source.into()),
    });

    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("bin_polygons_pipeline"),
        layout: Some(layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: rasterize_compilation_options(device),
        cache: Some(cache),
    })
}

fn bin_polygons_source(wg: [u32; 2]) -> String {
    specialize_bin_source(include_str!("../shaders/bin_polygons.wgsl"), wg)
}

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

fn validate_benchmark_bin_source(source: &str) -> Result<(), String> {
    if source.is_empty() || source.len() > MAX_BENCHMARK_SOURCE_BYTES {
        return Err("bin shader must contain 1..262144 bytes".into());
    }
    for marker in ["const TILE_W: u32 = 16;", "const TILE_H: u32 = 16;"] {
        if source.matches(marker).count() != 1 {
            return Err(format!(
                "bin specialization must match {marker:?} exactly once"
            ));
        }
    }
    Ok(())
}

fn specialize_benchmark_bin_source(
    source: &str,
    wg: [u32; 2],
    validation: bool,
) -> Result<String, String> {
    let specialized = specialize_bin_source(source, wg);
    if validation {
        benchmark_bin_source(&specialized)
    } else {
        Ok(specialized)
    }
}

fn benchmark_bin_source(source: &str) -> Result<String, String> {
    let add = "let slot = atomicAdd(&tile_counts[tile_global], 1u);";
    let store = "atomicStore(&tile_counts[tile_global], raw_count);";
    let plain_store = "tile_counts[tile_global] = raw_count;";
    // The reserved last word records uncapped occupancy over ALL iterations.
    // Recognize exactly one known counting mechanism in the installed source.
    match (
        source.matches(add).count(),
        source.matches(store).count(),
        source.matches(plain_store).count(),
    ) {
        (1, 0, 0) => Ok(source.replace(add, &format!("{add}\n                atomicMax(&tile_counts[arrayLength(&tile_counts) - 1u], slot + 1u);"))),
        (0, 1, 0) => Ok(source.replace(store, &format!("atomicMax(&tile_counts[arrayLength(&tile_counts) - 1u], raw_count);\n        {store}"))),
        (0, 0, 1) => {
            let plain_binding = "@group(0) @binding(2) var<storage, read_write> tile_counts:    array<u32>;";
            if source.matches(plain_binding).count() != 1
                || source.matches("tile_counts:").count() != 1
            {
                return Err("plain-count instrumentation requires exactly one binding-2 array<u32> declaration".into());
            }
            Ok(source.replace(plain_binding,
                "@group(0) @binding(2) var<storage, read_write> tile_counts:    array<atomic<u32>>;"
            ).replace(plain_store, &format!("atomicMax(&tile_counts[arrayLength(&tile_counts) - 1u], raw_count);\n        {store}")))
        }
        _ => Err("benchmark bin instrumentation requires exactly one recognized count marker".into()),
    }
}

/// Create an `init_framebuffers` compute pipeline with the given workgroup size.
fn create_init_framebuffers_pipeline(
    device: &Device,
    layout: &PipelineLayout,
    wg: [u32; 2],
    cache: &PipelineCache,
) -> ComputePipeline {
    let source = include_str!("../shaders/init_framebuffers.wgsl")
        .replace(
            "const WG_X: u32 = 16;",
            &format!("const WG_X: u32 = {};", wg[0]),
        )
        .replace(
            "const WG_Y: u32 = 16;",
            &format!("const WG_Y: u32 = {};", wg[1]),
        );

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("init_framebuffers_shader"),
        source: ShaderSource::Wgsl(source.into()),
    });

    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("init_framebuffers_pipeline"),
        layout: Some(layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: rasterize_compilation_options(device),
        cache: Some(cache),
    })
}

/// Create an `update_framebuffers` compute pipeline with the given workgroup size.
fn create_update_framebuffers_pipeline(
    device: &Device,
    layout: &PipelineLayout,
    wg: [u32; 2],
    cache: &PipelineCache,
) -> ComputePipeline {
    let source = include_str!("../shaders/update_framebuffers.wgsl")
        .replace(
            "const WG_X: u32 = 16;",
            &format!("const WG_X: u32 = {};", wg[0]),
        )
        .replace(
            "const WG_Y: u32 = 16;",
            &format!("const WG_Y: u32 = {};", wg[1]),
        );

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("update_framebuffers_shader"),
        source: ShaderSource::Wgsl(source.into()),
    });

    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("update_framebuffers_pipeline"),
        layout: Some(layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: rasterize_compilation_options(device),
        cache: Some(cache),
    })
}

/// Get the disk path for storing this adapter's pipeline cache.
/// Returns `None` if the backend doesn't support pipeline caching.
fn pipeline_cache_path(adapter_info: &AdapterInfo) -> Option<PathBuf> {
    let key = wgpu::util::pipeline_cache_key(adapter_info)?;
    Some(PathBuf::from(format!("pipeline_cache_{key}.bin")))
}

#[cfg(test)]
mod benchmark_tests {
    use sha2::{Digest, Sha256};

    use super::{
        benchmark_bin_source, benchmark_mutate_compilation_options,
        benchmark_mutate_group_size_from_source, bin_polygons_source,
        specialize_benchmark_bin_source, specialize_bin_source, specialize_raster_source,
        validate_benchmark_bin_source, validate_benchmark_mutate_source,
        validate_benchmark_raster_source, BenchmarkMutateKey, BenchmarkMutateOptions,
        BenchmarkRasterKey, BUNDLED_MUTATE_OPTIONS, MAX_BENCHMARK_SOURCE_BYTES,
    };

    const HISTORICAL_MUTATE: &str = include_str!(
        "../../benchmarks/evolution/overnight-20261024/fixtures/mutate-e133-reference.wgsl"
    );
    const GROUPED_MUTATE_G1: &str = include_str!(
        "../../benchmarks/evolution/overnight-20261024/fixtures/mutate-g1-wg64-copy64.wgsl"
    );
    // Metadata/cache fixture only; this empty kernel does not test mutation equivalence.
    const GROUPED_MUTATE_G64: &str = "const OFFSPRING_GROUP_SIZE: u32 = 64u;\n\
        @compute @workgroup_size(64) fn main() {}";

    #[test]
    fn mutate_fixtures_pin_historical_and_grouped_valid_wgsl() {
        for (source, expected_sha256) in [
            (
                HISTORICAL_MUTATE,
                "e13308a7cc59071043a3f5750a7f25214933ae4645cd016875f2dfa12c3dfe73",
            ),
            (
                GROUPED_MUTATE_G1,
                "3a36417746ab19b0c800c718e8c87795c17d9bff78114fd65bc8369b016578d8",
            ),
        ] {
            assert_eq!(
                format!("{:x}", Sha256::digest(source.as_bytes())),
                expected_sha256
            );
        }
        assert!(!HISTORICAL_MUTATE.contains("OFFSPRING_GROUP_SIZE"));
        for (source, group) in [
            (HISTORICAL_MUTATE, 64),
            (GROUPED_MUTATE_G1, 1),
            (GROUPED_MUTATE_G64, 64),
        ] {
            assert_eq!(
                benchmark_mutate_group_size_from_source(source).unwrap(),
                group
            );
            let module = wgpu::naga::front::wgsl::parse_str(source).unwrap();
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
        }
    }

    #[test]
    fn mutate_source_boundaries_need_no_specialization_markers() {
        let original = include_str!("../shaders/mutate.wgsl");
        assert!(!original.contains("const TILE_"));
        assert!(!original.contains("const WG_X"));
        assert!(!original.contains("const WG_Y"));
        validate_benchmark_mutate_source(original).unwrap();
        assert!(validate_benchmark_mutate_source("").is_err());
        let maximum = format!("//{}", " ".repeat(MAX_BENCHMARK_SOURCE_BYTES - 2));
        validate_benchmark_mutate_source(&maximum).unwrap();
        assert!(validate_benchmark_mutate_source(&format!("{maximum} ")).is_err());
        let multibyte = "λ".repeat(MAX_BENCHMARK_SOURCE_BYTES / 2 + 1);
        assert!(validate_benchmark_mutate_source(&multibyte).is_err());
    }

    #[test]
    fn mutate_cache_keys_couple_exact_source_and_checked_group_width() {
        let bundled = include_str!("../shaders/mutate.wgsl");
        let bundled_group = benchmark_mutate_group_size_from_source(bundled).unwrap();
        for (original, group) in [
            (HISTORICAL_MUTATE, 64),
            (GROUPED_MUTATE_G1, 1),
            (GROUPED_MUTATE_G64, 64),
            (bundled, bundled_group),
        ] {
            let key = BenchmarkMutateKey::new(original, BenchmarkMutateOptions::default()).unwrap();
            assert_eq!(key.group_size, group);
            assert_eq!(key.source.as_ref(), original);
            assert_eq!(
                key,
                BenchmarkMutateKey::new(original, BenchmarkMutateOptions::default()).unwrap()
            );
            for suffix in ["\n", "\n// candidate λ"] {
                let changed = BenchmarkMutateKey::new(
                    &format!("{original}{suffix}"),
                    BenchmarkMutateOptions::default(),
                )
                .unwrap();
                assert_ne!(key, changed);
                assert_eq!(changed.group_size, group);
            }
        }
        let grouped = BenchmarkMutateKey::new(
            "const OFFSPRING_GROUP_SIZE: u32 = 4u;",
            BenchmarkMutateOptions::default(),
        )
        .unwrap();
        assert_eq!(grouped.group_size, 4);
        assert_ne!(
            grouped,
            BenchmarkMutateKey::new(HISTORICAL_MUTATE, BenchmarkMutateOptions::default()).unwrap()
        );
        assert_ne!(
            BenchmarkMutateKey::new(GROUPED_MUTATE_G1, BenchmarkMutateOptions::default()).unwrap(),
            BenchmarkMutateKey::new(GROUPED_MUTATE_G64, BenchmarkMutateOptions::default()).unwrap()
        );
        assert_eq!(
            grouped,
            BenchmarkMutateKey::new(&grouped.source, BenchmarkMutateOptions::default()).unwrap()
        );
        assert!(BenchmarkMutateKey::new(
            "const OFFSPRING_GROUP_SIZE: u32 = 3u;",
            BenchmarkMutateOptions::default()
        )
        .is_err());
        assert!(wgpu::PipelineCompilationOptions::default().zero_initialize_workgroup_memory);
    }

    #[test]
    fn mutate_compilation_policy_changes_only_zeroing() {
        let defaults = wgpu::PipelineCompilationOptions::default();
        assert!(BenchmarkMutateOptions::default().zero_initialize_workgroup_memory);
        for zero_initialize_workgroup_memory in [true, false] {
            let actual = benchmark_mutate_compilation_options(BenchmarkMutateOptions {
                zero_initialize_workgroup_memory,
            });
            assert_eq!(
                actual.zero_initialize_workgroup_memory,
                zero_initialize_workgroup_memory
            );
            assert_eq!(actual.constants, defaults.constants);
            assert_eq!(actual.constants, []);
        }
    }

    #[test]
    fn mutate_cache_invalidates_on_policy_including_bundled_source() {
        let bundled = include_str!("../shaders/mutate.wgsl");
        let installed = BenchmarkMutateKey::new(bundled, BUNDLED_MUTATE_OPTIONS).unwrap();
        assert_eq!(
            installed.group_size,
            benchmark_mutate_group_size_from_source(bundled).unwrap()
        );
        for source in [
            HISTORICAL_MUTATE,
            GROUPED_MUTATE_G1,
            GROUPED_MUTATE_G64,
            bundled,
        ] {
            for zero_initialize_workgroup_memory in [true, false] {
                let options = BenchmarkMutateOptions {
                    zero_initialize_workgroup_memory,
                };
                let key = BenchmarkMutateKey::new(source, options).unwrap();
                assert_eq!(key.options, options);
                assert_eq!(key, BenchmarkMutateKey::new(source, options).unwrap());
                let other = BenchmarkMutateKey::new(
                    source,
                    BenchmarkMutateOptions {
                        zero_initialize_workgroup_memory: !zero_initialize_workgroup_memory,
                    },
                )
                .unwrap();
                assert_ne!(key, other);
                if source == bundled {
                    assert_eq!(key == installed, options == BUNDLED_MUTATE_OPTIONS);
                }
            }
        }
    }

    #[test]
    fn future_bundled_false_policy_cannot_reuse_initial_pipeline_for_historical_true() {
        let bundled = include_str!("../shaders/mutate.wgsl");
        let future_options = BenchmarkMutateOptions {
            zero_initialize_workgroup_memory: false,
        };
        let requested =
            BenchmarkMutateKey::new(bundled, BenchmarkMutateOptions::default()).unwrap();
        assert!(requested.options.zero_initialize_workgroup_memory);
        assert!(!requested.matches_installed(bundled, requested.group_size, future_options));
        assert!(requested.matches_installed(bundled, requested.group_size, requested.options));
        let future_request = BenchmarkMutateKey::new(bundled, future_options).unwrap();
        assert!(future_request.matches_installed(
            bundled,
            future_request.group_size,
            future_options
        ));
    }

    #[test]
    fn mutate_group_source_accepts_only_supported_exact_literals_and_historical_sources() {
        assert_eq!(
            benchmark_mutate_group_size_from_source(HISTORICAL_MUTATE).unwrap(),
            64
        );
        for group in [1, 2, 4, 8, 16, 32, 64] {
            let source = format!(
                "// λ\n/* outer /* nested */ comment */\n\
                const OFFSPRING_GROUP_SIZE: u32 = {group}u;\n\
                let logical = OFFSPRING_GROUP_SIZE * 2u; // OFFSPRING_GROUP_SIZE"
            );
            assert_eq!(
                benchmark_mutate_group_size_from_source(&source).unwrap(),
                group
            );
        }
        let maximum = format!("//{}", " ".repeat(MAX_BENCHMARK_SOURCE_BYTES - 2));
        assert_eq!(
            benchmark_mutate_group_size_from_source(&maximum).unwrap(),
            64
        );
        assert!(benchmark_mutate_group_size_from_source(&format!("{maximum} ")).is_err());
        assert!(benchmark_mutate_group_size_from_source("").is_err());
    }

    #[test]
    fn mutate_group_source_rejects_missing_nonliteral_malformed_and_duplicate_markers() {
        for literal in [
            "0u",
            "3u",
            "63u",
            "128u",
            "4294967297u",
            "0x4u",
            "04u",
            "4",
            "4U",
            "-4u",
            "4.0u",
            "4u + 0u",
            "u32(4)",
            "OTHER",
            "4 u",
        ] {
            let source = format!("const OFFSPRING_GROUP_SIZE: u32 = {literal};");
            assert!(
                benchmark_mutate_group_size_from_source(&source).is_err(),
                "{source}"
            );
        }
        for source in [
            "OFFSPRING_GROUP_SIZE",
            "// const OFFSPRING_GROUP_SIZE: u32 = 4u;",
            "/* const OFFSPRING_GROUP_SIZE: u32 = 4u; */",
            "/* /* nested */ const OFFSPRING_GROUP_SIZE: u32 = 4u; */",
            "const OFFSPRING_GROUP_SIZE: u32 = 4u",
            "const\nOFFSPRING_GROUP_SIZE: u32 = 4u;",
            "const OFFSPRING_GROUP_SIZE:u32 = 4u;",
            "const OFFSPRING_GROUP_SIZE: u32 =  4u;",
            "const /* comment */ OFFSPRING_GROUP_SIZE: u32 = 4u;",
            "override OFFSPRING_GROUP_SIZE: u32 = 4u;",
            "const OFFSPRING_GROUP_SIZE: u32 = 4u; /* unterminated",
            "const OFFSPRING_GROUP_SIZE: u32 = 4u; const OFFSPRING_GROUP_SIZE: u32 = 4u;",
            "const OFFSPRING_GROUP_SIZE: u32 = 4u; const\nOFFSPRING_GROUP_SIZE: u32 = 4u;",
            "// const OFFSPRING_GROUP_SIZE: u32 = 4u;\nconst OFFSPRING_GROUP_SIZE = 16u;",
        ] {
            assert!(
                benchmark_mutate_group_size_from_source(source).is_err(),
                "{source}"
            );
        }
    }

    #[test]
    fn raster_source_specialization_preserves_candidate_and_cache_identity() {
        let original = include_str!("../shaders/rasterize_error.wgsl");
        let candidate = format!("{original}\n// actual candidate");
        validate_benchmark_raster_source(&candidate).unwrap();
        for wg in [[32, 16], [16, 16], [32, 8], [16, 8], [8, 8]] {
            let source = specialize_raster_source(&candidate, wg);
            assert!(source.ends_with("// actual candidate"));
            assert!(source.contains(&format!("const WG_X: u32 = {};", wg[0])));
            assert!(source.contains(&format!("const WG_Y: u32 = {};", wg[1])));
            let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            let key = BenchmarkRasterKey {
                source: candidate.as_str().into(),
                wg,
            };
            assert_eq!(
                key,
                BenchmarkRasterKey {
                    source: candidate.as_str().into(),
                    wg
                }
            );
            assert_ne!(
                key,
                BenchmarkRasterKey {
                    source: original.into(),
                    wg
                }
            );
            assert_ne!(
                key,
                BenchmarkRasterKey {
                    source: candidate.as_str().into(),
                    wg: [1, 1]
                }
            );
        }
    }

    #[test]
    fn raster_source_boundaries_and_marker_cardinality_fail_closed() {
        let original = include_str!("../shaders/rasterize_error.wgsl");
        assert!(validate_benchmark_raster_source("").is_err());
        assert!(
            validate_benchmark_raster_source(&" ".repeat(MAX_BENCHMARK_SOURCE_BYTES + 1)).is_err()
        );
        for marker in ["const WG_X: u32 = 16;", "const WG_Y: u32 = 16;"] {
            assert!(validate_benchmark_raster_source(&original.replace(marker, "")).is_err());
            assert!(validate_benchmark_raster_source(&format!("{original}\n{marker}")).is_err());
        }
    }

    #[test]
    fn benchmark_bin_variants_validate_without_a_gpu() {
        for wg in [[32, 16], [16, 16], [32, 8], [16, 8], [8, 8]] {
            let normal = bin_polygons_source(wg);
            assert!(!normal.contains("atomicMax"));
            let source = benchmark_bin_source(&normal).unwrap();
            let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
        }
    }

    #[test]
    fn candidate_sources_keep_workgroup_and_validation_specialization() {
        // Keep the legacy atomic-add fixture independent of the production winner.
        // The variants test validates the currently installed source separately.
        let original = include_str!(
            "../../benchmarks/evolution/overnight-20261024/fixtures/bin-tile32-reference.wgsl"
        );
        let annotation = original
            .lines()
            .find(|line| line.starts_with("@compute @workgroup_size("))
            .unwrap();
        for internal_wg in [32, 64, 128, 256] {
            let candidate = original
                .replace(
                    annotation,
                    &format!("@compute @workgroup_size({internal_wg}, 1, 1)"),
                )
                .replace("+= 32u", &format!("+= {internal_wg}u"));
            validate_benchmark_bin_source(&candidate).unwrap();
            for wg in [[32, 16], [16, 16], [32, 8], [16, 8], [8, 8]] {
                let specialized = specialize_bin_source(&candidate, wg);
                assert!(specialized.contains(&format!("@workgroup_size({internal_wg}, 1, 1)")));
                assert!(specialized.contains(&format!("const TILE_W: u32 = {};", wg[0])));
                assert!(specialized.contains(&format!("const TILE_H: u32 = {};", wg[1])));
                let validation = benchmark_bin_source(&specialized).unwrap();
                assert!(validation.contains(
                    "atomicMax(&tile_counts[arrayLength(&tile_counts) - 1u], slot + 1u)"
                ));
                let module = wgpu::naga::front::wgsl::parse_str(&validation).unwrap();
                wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap();
            }
        }
    }

    #[test]
    fn candidate_source_boundaries_and_instrumentation_fail_closed() {
        let original = include_str!("../shaders/bin_polygons.wgsl");
        assert!(validate_benchmark_bin_source("").is_err());
        assert!(
            validate_benchmark_bin_source(&" ".repeat(MAX_BENCHMARK_SOURCE_BYTES + 1)).is_err()
        );
        assert!(
            validate_benchmark_bin_source(&original.replace("const TILE_W: u32 = 16;", ""))
                .is_err()
        );
        assert!(
            validate_benchmark_bin_source(&format!("{original}\nconst TILE_H: u32 = 16;")).is_err()
        );
        assert!(benchmark_bin_source("no atomicAdd marker").is_err());
        assert!(benchmark_bin_source(&format!("{original}\n{original}")).is_err());
    }

    #[test]
    fn local_count_validation_uses_uncapped_actual_store() {
        let source = r"
const TILE_W: u32 = 16;
const TILE_H: u32 = 16;
@group(0) @binding(0) var<storage, read_write> tile_counts: array<atomic<u32>>;
@compute @workgroup_size(32, 1, 1)
fn main() {
    let tile_global = 0u;
    let raw_count = 257u;
    atomicStore(&tile_counts[tile_global], raw_count);
}";
        validate_benchmark_bin_source(source).unwrap();
        let instrumented = benchmark_bin_source(source).unwrap();
        assert!(instrumented
            .contains("atomicMax(&tile_counts[arrayLength(&tile_counts) - 1u], raw_count);"));
        assert!(instrumented.contains("atomicStore(&tile_counts[tile_global], raw_count);"));
        let module = wgpu::naga::front::wgsl::parse_str(&instrumented).unwrap();
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let mixed = format!("{source}\nlet slot = atomicAdd(&tile_counts[tile_global], 1u);");
        assert!(benchmark_bin_source(&mixed).is_err());
        assert!(benchmark_bin_source(&format!("{source}\n{source}")).is_err());
    }

    const PLAIN_COUNT_BINDING: &str =
        "@group(0) @binding(2) var<storage, read_write> tile_counts:    array<u32>;";
    const ATOMIC_COUNT_BINDING: &str =
        "@group(0) @binding(2) var<storage, read_write> tile_counts:    array<atomic<u32>>;";
    const PLAIN_COUNT_STORE: &str = "tile_counts[tile_global] = raw_count;";
    const DIAGNOSTIC_COUNT_STORE: &str = "atomicMax(&tile_counts[arrayLength(&tile_counts) - 1u], raw_count);\n        atomicStore(&tile_counts[tile_global], raw_count);";

    fn plain_count_candidate() -> String {
        use sha2::Digest;

        // Keep historical instrumentation input independent of future shader tuning.
        let source = include_str!(
            "../../benchmarks/evolution/overnight-20261024/fixtures/bin-plain-count-reference.wgsl"
        )
        .to_owned();
        assert_eq!(
            format!("{:x}", sha2::Sha256::digest(source.as_bytes())),
            "dc8ac3971a96e0ad90bc4ce113675c11ac3352323e586bc806a5a0f3f3a3e105",
            "must match bin-micros/manifest.json candidate C"
        );
        source
    }

    fn validated_bin_module(source: &str) -> wgpu::naga::Module {
        let module = wgpu::naga::front::wgsl::parse_str(source).unwrap();
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        module
    }

    #[test]
    fn plain_count_candidate_preserves_normal_source_and_validates_all_tile_shapes() {
        let candidate = plain_count_candidate();
        validate_benchmark_bin_source(&candidate).unwrap();
        for wg in [[32, 16], [16, 16], [32, 8], [16, 8], [8, 8]] {
            let normal = specialize_benchmark_bin_source(&candidate, wg, false).unwrap();
            for (axis, size) in [("W", wg[0]), ("H", wg[1])] {
                assert_eq!(
                    normal
                        .matches(&format!("const TILE_{axis}: u32 = {size};"))
                        .count(),
                    1
                );
            }
            // Only the requested tile dimensions may differ in normal mode.
            assert_eq!(
                normal
                    .replace(
                        &format!("const TILE_W: u32 = {};", wg[0]),
                        "const TILE_W: u32 = 16;"
                    )
                    .replace(
                        &format!("const TILE_H: u32 = {};", wg[1]),
                        "const TILE_H: u32 = 16;"
                    ),
                candidate
            );
            if wg == [16, 16] {
                assert_eq!(normal.as_bytes(), candidate.as_bytes());
            }
            validated_bin_module(&normal);
            let diagnostic = specialize_benchmark_bin_source(&candidate, wg, true).unwrap();
            assert_eq!(
                diagnostic,
                normal
                    .replace(PLAIN_COUNT_BINDING, ATOMIC_COUNT_BINDING)
                    .replace(PLAIN_COUNT_STORE, DIAGNOSTIC_COUNT_STORE)
            );
            assert_eq!(diagnostic.matches(ATOMIC_COUNT_BINDING).count(), 1);
            assert_eq!(diagnostic.matches("atomicMax(").count(), 1);
            validated_bin_module(&diagnostic);
        }
    }

    #[test]
    fn plain_count_diagnostic_tail_is_uncapped_and_never_reset() {
        use wgpu::naga::{AtomicFunction, Expression, Literal, Statement};

        for raw_count in [257u32, 1000, 0] {
            let source = format!(
                "{PLAIN_COUNT_BINDING}
@compute @workgroup_size(1)
fn main() {{
    let tile_global = 0u;
    let raw_count = {raw_count}u;
    {PLAIN_COUNT_STORE}
}}"
            );
            let diagnostic = benchmark_bin_source(&source).unwrap();
            assert_eq!(diagnostic.matches(DIAGNOSTIC_COUNT_STORE).count(), 1);
            let module = validated_bin_module(&diagnostic);
            let function = &module.entry_points[0].function;
            let mut maxima = 0;
            let mut stores = 0;
            // Inspect parsed effects too: no cap and no assignment can clear the tail.
            for statement in &function.body {
                match *statement {
                    Statement::Atomic { fun, value, .. } => {
                        assert_eq!(fun, AtomicFunction::Max);
                        assert!(matches!(function.expressions[value],
                            Expression::Literal(Literal::U32(value)) if value == raw_count));
                        maxima += 1;
                    }
                    Statement::Store { value, .. } => {
                        assert!(matches!(function.expressions[value],
                            Expression::Literal(Literal::U32(value)) if value == raw_count));
                        stores += 1;
                    }
                    Statement::Emit(_) | Statement::Return { .. } => {}
                    _ => panic!("unexpected diagnostic effect: {statement:?}"),
                }
            }
            assert_eq!((maxima, stores), (1, 1));
        }
    }

    #[test]
    fn count_instrumentation_rejects_missing_duplicate_and_mixed_markers() {
        let candidate = plain_count_candidate();
        let markers = [
            "let slot = atomicAdd(&tile_counts[tile_global], 1u);",
            "atomicStore(&tile_counts[tile_global], raw_count);",
            PLAIN_COUNT_STORE,
        ];
        assert!(benchmark_bin_source(&candidate.replace(PLAIN_COUNT_STORE, "")).is_err());
        for marker in markers {
            let source = candidate.replace(PLAIN_COUNT_STORE, marker);
            for extra in markers {
                assert!(
                    benchmark_bin_source(&format!("{source}\n{extra}")).is_err(),
                    "accepted duplicate or mixed markers: {marker}, {extra}"
                );
            }
        }
    }

    #[test]
    fn plain_count_instrumentation_rejects_binding_mismatches() {
        let candidate = plain_count_candidate();
        for binding in [
            String::new(),
            ATOMIC_COUNT_BINDING.to_owned(),
            PLAIN_COUNT_BINDING.replace("@group(0)", "@group(1)"),
            PLAIN_COUNT_BINDING.replace("@binding(2)", "@binding(3)"),
            PLAIN_COUNT_BINDING.replace("read_write", "read"),
            format!("{PLAIN_COUNT_BINDING}\n{PLAIN_COUNT_BINDING}"),
            format!("{PLAIN_COUNT_BINDING}\n{ATOMIC_COUNT_BINDING}"),
            format!(
                "{PLAIN_COUNT_BINDING}\n{}",
                PLAIN_COUNT_BINDING.replace("@binding(2)", "@binding(3)")
            ),
        ] {
            assert!(
                benchmark_bin_source(&candidate.replace(PLAIN_COUNT_BINDING, &binding)).is_err(),
                "accepted count binding mismatch: {binding}"
            );
        }
    }
}
