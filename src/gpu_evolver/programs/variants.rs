//! Shader variants: historical and candidate programs that benchmarks install in
//! place of the bundled ones. Production never installs one; the state is empty.
//!
//! Variants keep their exact source, tile and policy identities in bounded caches
//! and survive tile changes. They are complete sources without the bundled
//! prelude; raster and binning variants mark their tile size for specialization. Binning variants can be instrumented for diagnostic
//! occupancy. A binning variant that does not list raster work gets a separate
//! worklist pass; a raster variant that does not read the list uses the full grid.

use std::num::NonZeroU64;
use std::sync::Arc;

use wgpu::naga::{front::wgsl, ResourceBinding};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, ComputePipeline, Device,
    PipelineCompilationOptions, PipelineLayout, ShaderStages,
};

use super::{
    compilation_scopes, create_tile_pipeline, finish_compilation, stage_layout, tile_source,
    GpuPrograms, MutationProgram, ProgramCompiler, RasterSchedule, TileStage, TileStages,
};
use crate::gpu_evolver::buffers::GPU_DRAWING_STATE_SIZE;

// Capacity for 17 lab sources, five supported raster workgroups, and two modes.
const MAX_BENCHMARK_BIN_PIPELINES: usize = 17 * 5 * 2;
const MAX_BENCHMARK_SOURCE_BYTES: usize = 256 * 1024;
const MAX_BENCHMARK_RASTER_PIPELINES: usize = 17 * 5;
const MAX_BENCHMARK_MUTATE_PIPELINES: usize = 17;

// Slots of the raster_work buffer in the binning and raster/error bind groups.
const BIN_RASTER_WORK_BINDING: u32 = 4;
const RASTER_RASTER_WORK_BINDING: u32 = 7;

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
pub(super) struct BenchmarkMutateKey {
    pub(super) source: Arc<str>,
    pub(super) group_size: u32,
    pub(super) options: BenchmarkMutateOptions,
}

impl BenchmarkMutateKey {
    pub(super) fn new(source: &str, options: BenchmarkMutateOptions) -> Result<Self, String> {
        let group_size = benchmark_mutate_group_size_from_source(source)?;
        Ok(Self {
            source: Arc::from(source),
            group_size,
            options,
        })
    }

    pub(super) fn matches_installed(
        &self,
        source: &str,
        group_size: u32,
        options: BenchmarkMutateOptions,
    ) -> bool {
        self.source.as_ref() == source && self.group_size == group_size && self.options == options
    }
}

#[derive(Debug, PartialEq, Eq)]
struct BenchmarkRasterKey {
    source: Arc<str>,
    wg: [u32; 2],
}

struct BenchmarkRasterPipeline {
    key: BenchmarkRasterKey,
    pipeline: ComputePipeline,
}

/// The source of a binning program: the bundled stage, which diagnostic
/// validation can instrument, or an installed variant.
#[derive(Clone, PartialEq, Eq)]
enum BinSource {
    Bundled,
    Variant(Arc<str>),
}

struct BenchmarkBinPipeline {
    source: BinSource,
    wg: [u32; 2],
    validation: bool,
    pipeline: ComputePipeline,
}

/// An installed raster or binning variant. Whether it binds its stage's
/// `raster_work` slot is decided once, when the variant is installed.
struct TileVariant {
    source: Arc<str>,
    raster_work: bool,
}

impl TileVariant {
    /// Older binners declare no `raster_work` binding and list nothing; older
    /// rasters declare none and index tiles by workgroup id.
    fn new(source: Arc<str>, binding: u32) -> Result<Self, String> {
        let module = wgsl::parse_str(&source).map_err(|error| error.emit_to_string(&source))?;
        let slot = Some(ResourceBinding { group: 0, binding });
        let raster_work = module
            .global_variables
            .iter()
            .any(|(_, global)| global.binding == slot);
        Ok(Self {
            source,
            raster_work,
        })
    }
}

/// Buffers read and written by the separate worklist pass.
pub(in crate::gpu_evolver) struct WorklistStorage<'a> {
    pub working_states: &'a Buffer,
    pub chain_states: &'a Buffer,
    pub tile_counts: &'a Buffer,
    pub raster_work: &'a Buffer,
}

/// The `raster_worklist.wgsl` pass for binning variants that do not list raster
/// work. A sixth dispatch per iteration made 512-batch runs host-bound, so the
/// bundled binner lists the pairs itself and production never runs this pass.
struct WorklistAdapter {
    bind_group_layout: BindGroupLayout,
    layout: PipelineLayout,
    bind_group: Option<BindGroup>,
    compiled: Option<([u32; 2], ComputePipeline)>,
}

impl WorklistAdapter {
    fn new(device: &Device, params: &BindGroupLayout) -> Self {
        // working_states(read), chain_states(read), tile_counts(read), raster_work(rw)
        let state_size = GPU_DRAWING_STATE_SIZE as u64;
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("raster_worklist_bgl"),
            entries: &[
                storage_entry(0, true, state_size),
                storage_entry(1, true, state_size),
                storage_entry(2, true, 4),
                storage_entry(3, false, 4),
            ],
        });
        let layout = stage_layout(device, "raster_worklist_layout", &bind_group_layout, params);
        Self {
            bind_group_layout,
            layout,
            bind_group: None,
            compiled: None,
        }
    }

    fn bind(&mut self, device: &Device, storage: &WorklistStorage<'_>) {
        if self.bind_group.is_some() {
            return;
        }
        let entries = [
            storage.working_states,
            storage.chain_states,
            storage.tile_counts,
            storage.raster_work,
        ];
        let entries = [0, 1, 2, 3].map(|binding| BindGroupEntry {
            binding,
            resource: entries[binding as usize].as_entire_binding(),
        });
        self.bind_group = Some(device.create_bind_group(&BindGroupDescriptor {
            label: Some("raster_worklist_bg"),
            layout: &self.bind_group_layout,
            entries: &entries,
        }));
    }

    /// Schedule for the selected raster and binning programs (`None` is bundled).
    fn schedule(
        &mut self,
        compiler: &ProgramCompiler,
        raster: Option<&TileVariant>,
        bin: Option<&TileVariant>,
        wg: [u32; 2],
    ) -> Result<RasterSchedule, String> {
        if !raster.is_none_or(|variant| variant.raster_work) {
            return Ok(RasterSchedule::FullGrid);
        }
        if bin.is_none_or(|variant| variant.raster_work) {
            return Ok(RasterSchedule::ListedByBinner);
        }
        let bind_group = self
            .bind_group
            .clone()
            .ok_or("worklist storage must be bound before a variant is installed")?;
        let pipeline = match &self.compiled {
            Some((compiled_wg, pipeline)) if *compiled_wg == wg => pipeline.clone(),
            _ => {
                let source = tile_source(TileStage::RasterWorklist, wg);
                checked(&compiler.device, || {
                    create_tile_pipeline(
                        &compiler.device,
                        &self.layout,
                        "raster_worklist",
                        &source,
                        &compiler.cache,
                    )
                })?
            }
        };
        self.compiled = Some((wg, pipeline.clone()));
        Ok(RasterSchedule::ListedByPass {
            pipeline,
            bind_group,
        })
    }
}

fn storage_entry(binding: u32, read_only: bool, min_size: u64) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(min_size),
        },
        count: None,
    }
}

/// Benchmark-installed programs and their compiled caches. `None` selects the
/// bundled raster or binning source; diagnostic validation instruments either.
pub(super) struct ShaderVariants {
    mutate: Vec<MutationProgram>,
    raster: Vec<BenchmarkRasterPipeline>,
    bin: Vec<BenchmarkBinPipeline>,
    raster_variant: Option<TileVariant>,
    bin_variant: Option<TileVariant>,
    validation: bool,
    worklist: WorklistAdapter,
}

impl ShaderVariants {
    pub(super) fn new(device: &Device, params: &BindGroupLayout) -> Self {
        Self {
            mutate: Vec::new(),
            raster: Vec::new(),
            bin: Vec::new(),
            raster_variant: None,
            bin_variant: None,
            validation: false,
            worklist: WorklistAdapter::new(device, params),
        }
    }

    /// Raster and binning programs for a tile change, or `None` when the bundled
    /// programs apply. Installed sources and validation carry over to `wg`.
    pub(super) fn tile_stages(
        &mut self,
        compiler: &ProgramCompiler,
        wg: [u32; 2],
    ) -> Option<Result<TileStages, String>> {
        if self.raster_variant.is_none() && self.bin_variant.is_none() && !self.validation {
            return None;
        }
        Some(self.prepare_stages(compiler, wg))
    }

    fn prepare_stages(
        &mut self,
        compiler: &ProgramCompiler,
        wg: [u32; 2],
    ) -> Result<TileStages, String> {
        let bin_source = self.bin_variant.as_ref().map(|variant| Arc::clone(&variant.source));
        let bin = match bin_source {
            Some(source) => {
                self.cached_bin(compiler, BinSource::Variant(source), wg, self.validation)?
            }
            None if self.validation => self.cached_bin(compiler, BinSource::Bundled, wg, true)?,
            None => compiler.bundled_bin(wg),
        };
        let raster_source = self
            .raster_variant
            .as_ref()
            .map(|variant| Arc::clone(&variant.source));
        let raster = match raster_source {
            Some(source) => self.cached_raster(compiler, source, wg)?,
            None => compiler.bundled_raster(wg),
        };
        let schedule = self.worklist.schedule(
            compiler,
            self.raster_variant.as_ref(),
            self.bin_variant.as_ref(),
            wg,
        )?;
        Ok(TileStages {
            raster,
            bin,
            schedule,
        })
    }

    fn cached_raster(
        &mut self,
        compiler: &ProgramCompiler,
        source: Arc<str>,
        wg: [u32; 2],
    ) -> Result<ComputePipeline, String> {
        let key = BenchmarkRasterKey { source, wg };
        if let Some(entry) = self.raster.iter().find(|entry| entry.key == key) {
            return Ok(entry.pipeline.clone());
        }
        if self.raster.len() == MAX_BENCHMARK_RASTER_PIPELINES {
            return Err("benchmark raster pipeline cache is full".into());
        }
        let specialized = specialize_raster_source(&key.source, wg);
        let pipeline = checked(&compiler.device, || compiler.raster(&specialized))?;
        self.raster.push(BenchmarkRasterPipeline {
            key,
            pipeline: pipeline.clone(),
        });
        Ok(pipeline)
    }

    fn cached_bin(
        &mut self,
        compiler: &ProgramCompiler,
        source: BinSource,
        wg: [u32; 2],
        validation: bool,
    ) -> Result<ComputePipeline, String> {
        if let Some(entry) = self.bin.iter().find(|entry| {
            entry.source == source && entry.wg == wg && entry.validation == validation
        }) {
            return Ok(entry.pipeline.clone());
        }
        if self.bin.len() == MAX_BENCHMARK_BIN_PIPELINES {
            return Err("benchmark bin pipeline cache is full".into());
        }
        let specialized = specialize_benchmark_bin_source(&source, wg, validation)?;
        let pipeline = checked(&compiler.device, || compiler.bin(&specialized))?;
        self.bin.push(BenchmarkBinPipeline {
            source,
            wg,
            validation,
            pipeline: pipeline.clone(),
        });
        Ok(pipeline)
    }
}

/// Compile under error scopes so a rejected variant leaves the installed one.
fn checked<T>(device: &Device, compile: impl FnOnce() -> T) -> Result<T, String> {
    let scopes = compilation_scopes(device);
    let compiled = compile();
    finish_compilation(scopes)?;
    Ok(compiled)
}

impl GpuPrograms {
    /// Open the variant seam. Binds the worklist pass's storage on first use.
    pub(in crate::gpu_evolver) fn shader_variants(
        &mut self,
        storage: &WorklistStorage<'_>,
    ) -> ShaderVariantInstaller<'_> {
        self.variants.worklist.bind(&self.compiler.device, storage);
        ShaderVariantInstaller { programs: self }
    }

    /// Whether the installed binner records diagnostic tile occupancy.
    pub(in crate::gpu_evolver) fn benchmark_validation(&self) -> bool {
        self.variants.validation
    }
}

/// Installs shader variants through the engine's specialization and pipeline
/// caches. The caller must own the pipeline exclusively and drain submitted GPU
/// work first. Installed sources and validation survive tile changes. Every
/// method leaves the installed programs unchanged on error.
pub struct ShaderVariantInstaller<'a> {
    programs: &'a mut GpuPrograms,
}

impl ShaderVariantInstaller<'_> {
    /// Install a mutation source with its compiler policy. Logical group width
    /// comes from the source; physical lanes do not change lambda.
    ///
    /// # Errors
    /// Rejects invalid group metadata, an exhausted cache or GPU errors.
    pub fn set_mutate_shader(
        &mut self,
        source: &str,
        options: BenchmarkMutateOptions,
    ) -> Result<(), String> {
        let key = BenchmarkMutateKey::new(source, options)?;
        let p = &mut *self.programs;
        let cache = &mut p.variants.mutate;
        if let Some(entry) = cache.iter().find(|entry| entry.key == key) {
            p.mutation = entry.clone();
            return Ok(());
        }
        if cache.len() == MAX_BENCHMARK_MUTATE_PIPELINES {
            return Err("benchmark mutate pipeline cache is full".into());
        }
        let pipeline = if cache.is_empty() && p.mutation.matches(source, options) {
            p.mutation.pipeline.clone()
        } else {
            checked(&p.compiler.device, || p.compiler.mutate(source, options))?
        };
        let selected = MutationProgram { key, pipeline };
        cache.push(selected.clone());
        p.mutation = selected;
        Ok(())
    }

    /// Install a raster/error source for the current tile size.
    ///
    /// # Errors
    /// Rejects invalid source bounds or specialization markers, an exhausted
    /// cache or GPU errors.
    pub fn set_raster_shader(&mut self, source: &str) -> Result<(), String> {
        validate_benchmark_raster_source(source)?;
        let p = &mut *self.programs;
        let variants = &mut p.variants;
        let source = variants
            .raster
            .iter()
            .find(|entry| entry.key.source.as_ref() == source)
            .map_or_else(|| Arc::from(source), |entry| Arc::clone(&entry.key.source));
        let pipeline = variants.cached_raster(&p.compiler, Arc::clone(&source), p.tiles.wg)?;
        let variant = TileVariant::new(source, RASTER_RASTER_WORK_BINDING)?;
        let schedule = variants.worklist.schedule(
            &p.compiler,
            Some(&variant),
            variants.bin_variant.as_ref(),
            p.tiles.wg,
        )?;
        p.tiles.raster = pipeline;
        p.tiles.schedule = schedule;
        variants.raster_variant = Some(variant);
        Ok(())
    }

    /// Install a binning source for the current tile size and validation mode.
    ///
    /// # Errors
    /// Rejects invalid source bounds or specialization markers, a full cache,
    /// failed instrumentation or GPU errors.
    pub fn set_bin_shader(&mut self, source: &str) -> Result<(), String> {
        validate_benchmark_bin_source(source)?;
        let p = &mut *self.programs;
        let variants = &mut p.variants;
        let source = variants
            .bin
            .iter()
            .find_map(|entry| match &entry.source {
                BinSource::Variant(cached) if cached.as_ref() == source => Some(Arc::clone(cached)),
                _ => None,
            })
            .unwrap_or_else(|| Arc::from(source));
        let pipeline = variants.cached_bin(
            &p.compiler,
            BinSource::Variant(Arc::clone(&source)),
            p.tiles.wg,
            variants.validation,
        )?;
        let variant = TileVariant::new(source, BIN_RASTER_WORK_BINDING)?;
        let schedule = variants.worklist.schedule(
            &p.compiler,
            variants.raster_variant.as_ref(),
            Some(&variant),
            p.tiles.wg,
        )?;
        p.tiles.bin = pipeline;
        p.tiles.schedule = schedule;
        variants.bin_variant = Some(variant);
        Ok(())
    }

    /// Instrument the actual installed binning source, including after tile changes.
    ///
    /// # Errors
    /// Rejects a full cache, failed instrumentation or GPU errors.
    pub fn set_validation(&mut self, enabled: bool) -> Result<(), String> {
        let p = &mut *self.programs;
        let variants = &mut p.variants;
        if enabled == variants.validation {
            return Ok(());
        }
        let source = variants.bin_variant.as_ref().map_or(BinSource::Bundled, |variant| {
            BinSource::Variant(Arc::clone(&variant.source))
        });
        p.tiles.bin = variants.cached_bin(&p.compiler, source, p.tiles.wg, enabled)?;
        variants.validation = enabled;
        Ok(())
    }
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

// Tile-size markers. Historical rasters use the first pair; historical binners
// and bundled sources emitted at 16x16 (programs::bundled) use the second.
const RASTER_TILE_MARKERS: [[&str; 2]; 2] = [
    ["const WG_X: u32 = 16;", "const WG_Y: u32 = 16;"],
    BIN_TILE_MARKERS,
];
const BIN_TILE_MARKERS: [&str; 2] = ["const TILE_W: u32 = 16;", "const TILE_H: u32 = 16;"];

fn validate_benchmark_raster_source(source: &str) -> Result<(), String> {
    if source.is_empty() || source.len() > MAX_BENCHMARK_SOURCE_BYTES {
        return Err("raster shader must contain 1..262144 bytes".into());
    }
    let counts = RASTER_TILE_MARKERS.map(|pair| pair.map(|marker| source.matches(marker).count()));
    match counts {
        [[1, 1], [0, 0]] | [[0, 0], [1, 1]] => Ok(()),
        _ => Err(format!(
            "raster specialization must match one marker pair of {RASTER_TILE_MARKERS:?} exactly once"
        )),
    }
}

fn validate_benchmark_bin_source(source: &str) -> Result<(), String> {
    if source.is_empty() || source.len() > MAX_BENCHMARK_SOURCE_BYTES {
        return Err("bin shader must contain 1..262144 bytes".into());
    }
    for marker in BIN_TILE_MARKERS {
        if source.matches(marker).count() != 1 {
            return Err(format!(
                "bin specialization must match {marker:?} exactly once"
            ));
        }
    }
    Ok(())
}

/// Specialize a validated raster variant's tile-size markers for `wg`. Naga
/// supports no override constants in `@workgroup_size` or const expressions.
fn specialize_raster_source(source: &str, wg: [u32; 2]) -> String {
    let [wg_markers, tile_markers] = RASTER_TILE_MARKERS;
    replace_tile_markers(&replace_tile_markers(source, wg_markers, wg), tile_markers, wg)
}

/// Specialize a validated binning variant's tile-size markers for `wg`.
fn specialize_bin_source(source: &str, wg: [u32; 2]) -> String {
    replace_tile_markers(source, BIN_TILE_MARKERS, wg)
}

/// Replace `const NAME: u32 = 16;` markers with the tile's width and height.
fn replace_tile_markers(source: &str, markers: [&str; 2], wg: [u32; 2]) -> String {
    let [width, height] = markers.map(|marker| {
        marker
            .strip_suffix("16;")
            .expect("tile markers end with their 16 default")
    });
    source
        .replace(markers[0], &format!("{width}{};", wg[0]))
        .replace(markers[1], &format!("{height}{};", wg[1]))
}

/// The binning program for `wg`, instrumented when `validation` is set.
fn specialize_benchmark_bin_source(
    source: &BinSource,
    wg: [u32; 2],
    validation: bool,
) -> Result<String, String> {
    let specialized = match source {
        BinSource::Bundled => tile_source(TileStage::BinPolygons, wg),
        BinSource::Variant(source) => specialize_bin_source(source, wg),
    };
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

#[cfg(test)]
mod benchmark_tests {
    use sha2::{Digest, Sha256};

    use super::super::{
        bundled_shader_source, tile_source, BundledShader, TileStage, BUNDLED_MUTATE_OPTIONS,
    };
    use super::{
        benchmark_bin_source, benchmark_mutate_compilation_options,
        benchmark_mutate_group_size_from_source, specialize_benchmark_bin_source,
        specialize_bin_source, specialize_raster_source, validate_benchmark_bin_source,
        validate_benchmark_mutate_source, validate_benchmark_raster_source, BenchmarkMutateKey,
        BenchmarkMutateOptions, BenchmarkRasterKey, BinSource, TileVariant,
        BIN_RASTER_WORK_BINDING, MAX_BENCHMARK_SOURCE_BYTES, RASTER_RASTER_WORK_BINDING,
    };

    const TILES: [[u32; 2]; 5] = [[32, 16], [16, 16], [32, 8], [16, 8], [8, 8]];
    const HISTORICAL_RASTER: &str = include_str!(
        "../../../benchmarks/evolution/overnight-20261024/fixtures/raster-ae1-reference.wgsl"
    );

    const HISTORICAL_MUTATE: &str = include_str!(
        "../../../benchmarks/evolution/overnight-20261024/fixtures/mutate-e133-reference.wgsl"
    );
    const GROUPED_MUTATE_G1: &str = include_str!(
        "../../../benchmarks/evolution/overnight-20261024/fixtures/mutate-g1-wg64-copy64.wgsl"
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
        let original = &bundled_shader_source(BundledShader::Mutate);
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
        let bundled = &bundled_shader_source(BundledShader::Mutate);
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
        let bundled = &bundled_shader_source(BundledShader::Mutate);
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
        let bundled = &bundled_shader_source(BundledShader::Mutate);
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
        let original = HISTORICAL_RASTER;
        let candidate = format!("{original}\n// actual candidate");
        validate_benchmark_raster_source(&candidate).unwrap();
        for wg in TILES {
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
        let emitted = tile_source(TileStage::RasterizeError, [16, 16]);
        assert!(validate_benchmark_raster_source("").is_err());
        assert!(
            validate_benchmark_raster_source(&" ".repeat(MAX_BENCHMARK_SOURCE_BYTES + 1)).is_err()
        );
        for (original, markers) in [
            (HISTORICAL_RASTER, ["const WG_X: u32 = 16;", "const WG_Y: u32 = 16;"]),
            (&emitted, ["const TILE_W: u32 = 16;", "const TILE_H: u32 = 16;"]),
        ] {
            validate_benchmark_raster_source(original).unwrap();
            for marker in markers {
                assert!(validate_benchmark_raster_source(&original.replace(marker, "")).is_err());
                assert!(
                    validate_benchmark_raster_source(&format!("{original}\n{marker}")).is_err()
                );
            }
        }
        // A source may not mark its tile with both pairs.
        let both = format!("{HISTORICAL_RASTER}\nconst TILE_W: u32 = 16;\nconst TILE_H: u32 = 16;");
        assert!(validate_benchmark_raster_source(&both).is_err());
        // The installed bundled source is not a variant.
        let installed = tile_source(TileStage::RasterizeError, [32, 16]);
        assert!(validate_benchmark_raster_source(&installed).is_err());
    }

    /// Bundled raster and binning sources emitted at 16x16 install as variants
    /// and specialize to exactly the installed bundled source at every tile.
    #[test]
    fn emitted_bundled_sources_specialize_like_installed_ones() {
        let raster = tile_source(TileStage::RasterizeError, [16, 16]);
        let bin = tile_source(TileStage::BinPolygons, [16, 16]);
        validate_benchmark_raster_source(&raster).unwrap();
        validate_benchmark_bin_source(&bin).unwrap();
        for wg in TILES {
            assert_eq!(
                specialize_raster_source(&raster, wg),
                tile_source(TileStage::RasterizeError, wg)
            );
            assert_eq!(specialize_bin_source(&bin, wg), tile_source(TileStage::BinPolygons, wg));
        }
    }

    #[test]
    fn benchmark_bin_variants_validate_without_a_gpu() {
        for wg in TILES {
            let normal = specialize_benchmark_bin_source(&BinSource::Bundled, wg, false).unwrap();
            assert_eq!(normal, tile_source(TileStage::BinPolygons, wg));
            assert!(!normal.contains("atomicMax"));
            let source = specialize_benchmark_bin_source(&BinSource::Bundled, wg, true).unwrap();
            assert_eq!(source, benchmark_bin_source(&normal).unwrap());
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
            "../../../benchmarks/evolution/overnight-20261024/fixtures/bin-tile32-reference.wgsl"
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
            for wg in TILES {
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
        let original = &tile_source(TileStage::BinPolygons, [16, 16]);
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
            "../../../benchmarks/evolution/overnight-20261024/fixtures/bin-plain-count-reference.wgsl"
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
        let variant = BinSource::Variant(candidate.as_str().into());
        for wg in TILES {
            let normal = specialize_benchmark_bin_source(&variant, wg, false).unwrap();
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
            let diagnostic = specialize_benchmark_bin_source(&variant, wg, true).unwrap();
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

    #[test]
    fn raster_work_role_follows_declared_binding_not_text() {
        let fixture = |name: &str| {
            std::fs::read_to_string(format!(
                "{}/benchmarks/evolution/overnight-20261024/fixtures/{name}",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap()
        };
        let role = |source: &str, binding| {
            TileVariant::new(source.into(), binding).unwrap().raster_work
        };
        let bin = tile_source(TileStage::BinPolygons, [16, 16]);
        let raster = tile_source(TileStage::RasterizeError, [16, 16]);
        assert!(role(&bin, BIN_RASTER_WORK_BINDING));
        assert!(role(&raster, RASTER_RASTER_WORK_BINDING));
        for name in ["bin-tile32-reference.wgsl", "bin-plain-count-reference.wgsl"] {
            assert!(!role(&fixture(name), BIN_RASTER_WORK_BINDING), "{name}");
        }
        for name in [
            "raster-ae1-reference.wgsl",
            "raster-direct-t2.wgsl",
            "raster-direct-t4.wgsl",
        ] {
            assert!(!role(&fixture(name), RASTER_RASTER_WORK_BINDING), "{name}");
        }
        // A mention outside a binding declaration does not make a source list work.
        let mention = format!("// writes raster_work\n{}", fixture("bin-tile32-reference.wgsl"));
        assert!(!role(&mention, BIN_RASTER_WORK_BINDING));
        assert!(TileVariant::new("fn".into(), BIN_RASTER_WORK_BINDING).is_err());
    }
}
