//! Shader variants: candidate programs that benchmarks install in place of the
//! bundled ones. Production never installs one; the state is empty.
//!
//! Variants keep their exact source, tile and policy identities in bounded caches
//! and survive tile changes. They are complete current-tree sources, as
//! `examples/emit-bundled-shaders` writes them: raster and binning variants mark
//! their 16x16 tile size for specialization and bind their stage's `raster_work`
//! slot, since the binner lists raster work and raster loops over the list. The
//! binning program can be instrumented for diagnostic occupancy.

use std::sync::Arc;

use wgpu::naga::{front::wgsl, ResourceBinding};
use wgpu::{ComputePipeline, Device, PipelineCompilationOptions};

use super::{
    compilation_scopes, finish_compilation, tile_source, GpuPrograms, MutationProgram,
    ProgramCompiler, TileStage, TileStages,
};
use crate::gpu_evolver::bindings::{self, Resource, BIN_POLYGONS, RASTERIZE_ERROR};
use crate::settings::POLYGON_CAPACITY;

const STATE_MEMBER: &str = "polygons: array<Polygon, POLYGON_CAPACITY>,";
const CAPACITY_DECLARATION: &str = "const POLYGON_CAPACITY: u32 = ";

/// Check that a variant's drawing state has the engine's stride. Emitted bundled
/// sources declare `polygons: array<Polygon, POLYGON_CAPACITY>,` with
/// `const POLYGON_CAPACITY: u32 = Nu;`, where N must be this build's capacity. A
/// source with no `Polygon` array binds no drawing state.
///
/// # Errors
/// Rejects any other polygon storage declaration.
pub fn validate_polygon_storage(source: &str) -> Result<(), String> {
    let declaration = format!("{CAPACITY_DECLARATION}{POLYGON_CAPACITY}u;");
    let counts = [STATE_MEMBER, CAPACITY_DECLARATION, declaration.as_str()]
        .map(|marker| source.matches(marker).count());
    match counts {
        [1, 1, 1] => Ok(()),
        [0, 0, 0] if !source.contains("array<Polygon") => Ok(()),
        _ => Err(format!(
            "a variant's drawing state must declare `{STATE_MEMBER}` with `{declaration}`"
        )),
    }
}

// Capacity for 17 lab sources, five supported raster workgroups, and two modes.
const MAX_BENCHMARK_BIN_PIPELINES: usize = 17 * 5 * 2;
const MAX_BENCHMARK_SOURCE_BYTES: usize = 256 * 1024;
const MAX_BENCHMARK_RASTER_PIPELINES: usize = 17 * 5;
const MAX_BENCHMARK_MUTATE_PIPELINES: usize = 17;

// Slots of the raster_work buffer in the binning and raster/error bind groups.
const BIN_RASTER_WORK_BINDING: u32 = bindings::binding(&BIN_POLYGONS, Resource::RasterWork);
const RASTER_RASTER_WORK_BINDING: u32 =
    bindings::binding(&RASTERIZE_ERROR, Resource::RasterWork);

/// Mutation-only compiler policy. Workgroup-memory zeroing is the default.
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

/// Benchmark-installed programs and their compiled caches. `None` selects the
/// bundled raster or binning source; diagnostic validation instruments either.
pub(super) struct ShaderVariants {
    mutate: Vec<MutationProgram>,
    raster: Vec<BenchmarkRasterPipeline>,
    bin: Vec<BenchmarkBinPipeline>,
    raster_variant: Option<Arc<str>>,
    bin_variant: Option<Arc<str>>,
    validation: bool,
}

impl ShaderVariants {
    pub(super) fn new() -> Self {
        Self {
            mutate: Vec::new(),
            raster: Vec::new(),
            bin: Vec::new(),
            raster_variant: None,
            bin_variant: None,
            validation: false,
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
        let bin = match self.bin_variant.clone() {
            Some(source) => {
                self.cached_bin(compiler, BinSource::Variant(source), wg, self.validation)?
            }
            None if self.validation => self.cached_bin(compiler, BinSource::Bundled, wg, true)?,
            None => compiler.bundled_bin(wg),
        };
        let raster = match self.raster_variant.clone() {
            Some(source) => self.cached_raster(compiler, source, wg)?,
            None => compiler.bundled_raster(wg),
        };
        Ok(TileStages { raster, bin })
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
        let specialized = specialize_tile_source(&key.source, wg);
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
    /// Open the variant seam.
    pub(in crate::gpu_evolver) fn shader_variants(&mut self) -> ShaderVariantInstaller<'_> {
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
    /// Rejects invalid group metadata or polygon storage, an exhausted cache or
    /// GPU errors.
    pub fn set_mutate_shader(
        &mut self,
        source: &str,
        options: BenchmarkMutateOptions,
    ) -> Result<(), String> {
        let key = BenchmarkMutateKey::new(source, options)?;
        validate_polygon_storage(source)?;
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
    /// Rejects invalid source bounds, specialization markers or bindings, an
    /// exhausted cache or GPU errors.
    pub fn set_raster_shader(&mut self, source: &str) -> Result<(), String> {
        validate_tile_source("raster", source, RASTER_RASTER_WORK_BINDING)?;
        let p = &mut *self.programs;
        let variants = &mut p.variants;
        let source = variants
            .raster
            .iter()
            .find(|entry| entry.key.source.as_ref() == source)
            .map_or_else(|| Arc::from(source), |entry| Arc::clone(&entry.key.source));
        p.tiles.raster = variants.cached_raster(&p.compiler, Arc::clone(&source), p.tiles.wg)?;
        variants.raster_variant = Some(source);
        Ok(())
    }

    /// Install a binning source for the current tile size and validation mode.
    ///
    /// # Errors
    /// Rejects invalid source bounds, specialization markers or bindings, a full
    /// cache, failed instrumentation or GPU errors.
    pub fn set_bin_shader(&mut self, source: &str) -> Result<(), String> {
        validate_tile_source("bin", source, BIN_RASTER_WORK_BINDING)?;
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
        p.tiles.bin = variants.cached_bin(
            &p.compiler,
            BinSource::Variant(Arc::clone(&source)),
            p.tiles.wg,
            variants.validation,
        )?;
        variants.bin_variant = Some(source);
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
        let source = variants
            .bin_variant
            .clone()
            .map_or(BinSource::Bundled, BinSource::Variant);
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
/// Every mutation source declares it, independently of physical lane count.
///
/// # Errors
/// Rejects oversized/empty sources and missing, duplicate or nonliteral markers.
pub fn benchmark_mutate_group_size_from_source(source: &str) -> Result<u32, String> {
    const NAME: &str = "OFFSPRING_GROUP_SIZE";
    const PREFIX: &str = "const OFFSPRING_GROUP_SIZE: u32 = ";
    validate_benchmark_mutate_source(source)?;
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
        .ok_or_else(|| "missing exact `const OFFSPRING_GROUP_SIZE: u32 = Nu;` declaration".into())
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

// Tile-size markers of raster and binning sources emitted at 16x16
// (programs::bundled).
const TILE_MARKERS: [&str; 2] = ["const TILE_W: u32 = 16;", "const TILE_H: u32 = 16;"];

/// Check a raster or binning variant: its size, polygon storage, one of each
/// tile marker, and its stage's `raster_work` binding.
fn validate_tile_source(stage: &str, source: &str, raster_work: u32) -> Result<(), String> {
    if source.is_empty() || source.len() > MAX_BENCHMARK_SOURCE_BYTES {
        return Err(format!("{stage} shader must contain 1..262144 bytes"));
    }
    validate_polygon_storage(source)?;
    for marker in TILE_MARKERS {
        if source.matches(marker).count() != 1 {
            return Err(format!(
                "{stage} specialization must match {marker:?} exactly once"
            ));
        }
    }
    if !declares_binding(source, raster_work)? {
        return Err(format!(
            "{stage} shader must bind raster_work at @group(0) @binding({raster_work})"
        ));
    }
    Ok(())
}

/// Whether `source` declares a resource at group 0, `binding`. Only a
/// declaration counts, not a mention in text.
fn declares_binding(source: &str, binding: u32) -> Result<bool, String> {
    let module = wgsl::parse_str(source).map_err(|error| error.emit_to_string(source))?;
    let slot = Some(ResourceBinding { group: 0, binding });
    let declared = module
        .global_variables
        .iter()
        .any(|(_, global)| global.binding == slot);
    Ok(declared)
}

/// Replace a validated variant's `const NAME: u32 = 16;` tile markers with the
/// tile's width and height. Naga supports no override constants in
/// `@workgroup_size` or const expressions.
fn specialize_tile_source(source: &str, wg: [u32; 2]) -> String {
    let [width, height] = TILE_MARKERS.map(|marker| {
        marker
            .strip_suffix("16;")
            .expect("tile markers end with their 16 default")
    });
    source
        .replace(TILE_MARKERS[0], &format!("{width}{};", wg[0]))
        .replace(TILE_MARKERS[1], &format!("{height}{};", wg[1]))
}

/// The binning program for `wg`, instrumented when `validation` is set.
fn specialize_benchmark_bin_source(
    source: &BinSource,
    wg: [u32; 2],
    validation: bool,
) -> Result<String, String> {
    let specialized = match source {
        BinSource::Bundled => tile_source(TileStage::BinPolygons, wg),
        BinSource::Variant(source) => specialize_tile_source(source, wg),
    };
    if validation {
        benchmark_bin_source(&specialized)
    } else {
        Ok(specialized)
    }
}

fn benchmark_bin_source(source: &str) -> Result<String, String> {
    let store = "atomicStore(&tile_counts[tile_global], raw_count);";
    // The reserved last word records uncapped occupancy over ALL iterations.
    if source.matches(store).count() != 1 {
        return Err(format!(
            "benchmark bin instrumentation requires exactly one `{store}`"
        ));
    }
    Ok(source.replace(store, &format!("atomicMax(&tile_counts[arrayLength(&tile_counts) - 1u], raw_count);\n        {store}")))
}

#[cfg(test)]
mod benchmark_tests {
    use super::super::{
        bundled_shader_source, tile_source, BundledShader, TileStage, BUNDLED_MUTATE_OPTIONS,
    };
    use super::{
        benchmark_bin_source, benchmark_mutate_compilation_options,
        benchmark_mutate_group_size_from_source, declares_binding,
        specialize_benchmark_bin_source, specialize_tile_source, validate_benchmark_mutate_source,
        validate_polygon_storage, validate_tile_source, BenchmarkMutateKey,
        BenchmarkMutateOptions, BenchmarkRasterKey, BinSource, BIN_RASTER_WORK_BINDING,
        MAX_BENCHMARK_SOURCE_BYTES, POLYGON_CAPACITY, RASTER_RASTER_WORK_BINDING,
    };

    const TILES: [[u32; 2]; 5] = [[32, 16], [16, 16], [32, 8], [16, 8], [8, 8]];
    // Metadata/cache fixtures only; these empty kernels do not test mutation equivalence.
    const GROUPED_MUTATE_G1: &str = "const OFFSPRING_GROUP_SIZE: u32 = 1u;\n\
        @compute @workgroup_size(64) fn main() {}";
    const GROUPED_MUTATE_G64: &str = "const OFFSPRING_GROUP_SIZE: u32 = 64u;\n\
        @compute @workgroup_size(64) fn main() {}";

    fn validate_raster(source: &str) -> Result<(), String> {
        validate_tile_source("raster", source, RASTER_RASTER_WORK_BINDING)
    }

    fn validate_bin(source: &str) -> Result<(), String> {
        validate_tile_source("bin", source, BIN_RASTER_WORK_BINDING)
    }

    fn validated_module(source: &str) -> wgpu::naga::Module {
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
    fn polygon_storage_accepts_this_builds_emitted_sources_and_rejects_others() {
        let capacity = POLYGON_CAPACITY;
        for shader in [BundledShader::Mutate, BundledShader::Tile(TileStage::BinPolygons, [16, 16])] {
            let source = bundled_shader_source(shader);
            validate_polygon_storage(&source).unwrap();
            // Another build's capacity would address states at another stride.
            let declaration = format!("const POLYGON_CAPACITY: u32 = {capacity}u;");
            for other in ["1000u", "4096u"] {
                let changed =
                    source.replace(&declaration, &format!("const POLYGON_CAPACITY: u32 = {other};"));
                assert!(validate_polygon_storage(&changed).is_err(), "{other}");
            }
            let literal = source.replace(
                "polygons: array<Polygon, POLYGON_CAPACITY>,",
                "polygons: array<Polygon, 1000>,",
            );
            assert!(validate_polygon_storage(&literal).is_err());
            let twice = format!("{source}\nstruct Again {{ polygons: array<Polygon, POLYGON_CAPACITY>, }}");
            assert!(validate_polygon_storage(&twice).is_err());
        }
        // A program that binds no drawing state needs no specialization.
        validate_polygon_storage(GROUPED_MUTATE_G64).unwrap();
    }

    #[test]
    fn mutate_source_boundaries_need_no_specialization_markers() {
        let original = &bundled_shader_source(BundledShader::Mutate);
        assert!(!original.contains("const TILE_W"));
        assert!(!original.contains("const TILE_H"));
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
            BenchmarkMutateKey::new(GROUPED_MUTATE_G64, BenchmarkMutateOptions::default()).unwrap()
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
        for source in [GROUPED_MUTATE_G1, GROUPED_MUTATE_G64, bundled] {
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
    fn future_bundled_false_policy_cannot_reuse_initial_pipeline_for_default_true() {
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
    fn mutate_group_source_accepts_only_supported_exact_literals() {
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
        let declaration = "const OFFSPRING_GROUP_SIZE: u32 = 64u;";
        let maximum = format!(
            "{declaration}//{}",
            " ".repeat(MAX_BENCHMARK_SOURCE_BYTES - declaration.len() - 2)
        );
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
            "@compute @workgroup_size(64) fn main() {}",
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
        let original = tile_source(TileStage::RasterizeError, [16, 16]);
        let candidate = format!("{original}\n// actual candidate");
        validate_raster(&candidate).unwrap();
        for wg in TILES {
            let source = specialize_tile_source(&candidate, wg);
            assert!(source.ends_with("// actual candidate"));
            assert!(source.contains(&format!("const TILE_W: u32 = {};", wg[0])));
            assert!(source.contains(&format!("const TILE_H: u32 = {};", wg[1])));
            validated_module(&source);
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
                    source: original.as_str().into(),
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
        assert!(validate_raster("").is_err());
        assert!(validate_raster(&" ".repeat(MAX_BENCHMARK_SOURCE_BYTES + 1)).is_err());
        validate_raster(&emitted).unwrap();
        for marker in ["const TILE_W: u32 = 16;", "const TILE_H: u32 = 16;"] {
            assert!(validate_raster(&emitted.replace(marker, "")).is_err());
            assert!(validate_raster(&format!("{emitted}\n{marker}")).is_err());
        }
        // The installed bundled source is not a variant.
        let installed = tile_source(TileStage::RasterizeError, [32, 16]);
        assert!(validate_raster(&installed).is_err());
    }

    /// Bundled raster and binning sources emitted at 16x16 install as variants
    /// and specialize to exactly the installed bundled source at every tile.
    #[test]
    fn emitted_bundled_sources_specialize_like_installed_ones() {
        let raster = tile_source(TileStage::RasterizeError, [16, 16]);
        let bin = tile_source(TileStage::BinPolygons, [16, 16]);
        validate_raster(&raster).unwrap();
        validate_bin(&bin).unwrap();
        for wg in TILES {
            assert_eq!(
                specialize_tile_source(&raster, wg),
                tile_source(TileStage::RasterizeError, wg)
            );
            assert_eq!(specialize_tile_source(&bin, wg), tile_source(TileStage::BinPolygons, wg));
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
            validated_module(&source);
        }
    }

    #[test]
    fn candidate_sources_keep_tile_and_validation_specialization() {
        let original = tile_source(TileStage::BinPolygons, [16, 16]);
        let candidate = format!("{original}\n// actual candidate");
        validate_bin(&candidate).unwrap();
        let variant = BinSource::Variant(candidate.as_str().into());
        for wg in TILES {
            let normal = specialize_benchmark_bin_source(&variant, wg, false).unwrap();
            assert_eq!(normal, specialize_tile_source(&candidate, wg));
            assert!(normal.ends_with("// actual candidate"));
            assert!(normal.contains(&format!("const TILE_W: u32 = {};", wg[0])));
            assert!(normal.contains(&format!("const TILE_H: u32 = {};", wg[1])));
            let validation = specialize_benchmark_bin_source(&variant, wg, true).unwrap();
            assert_eq!(validation, benchmark_bin_source(&normal).unwrap());
            assert_eq!(validation.matches("atomicMax(").count(), 1);
            validated_module(&validation);
        }
    }

    #[test]
    fn candidate_source_boundaries_and_instrumentation_fail_closed() {
        let original = &tile_source(TileStage::BinPolygons, [16, 16]);
        assert!(validate_bin("").is_err());
        assert!(validate_bin(&" ".repeat(MAX_BENCHMARK_SOURCE_BYTES + 1)).is_err());
        assert!(validate_bin(&original.replace("const TILE_W: u32 = 16;", "")).is_err());
        assert!(validate_bin(&format!("{original}\nconst TILE_H: u32 = 16;")).is_err());
        assert!(benchmark_bin_source("no count marker").is_err());
        assert!(benchmark_bin_source(&format!("{original}\n{original}")).is_err());
    }

    #[test]
    fn local_count_validation_uses_uncapped_actual_store() {
        let source = r"
@group(0) @binding(0) var<storage, read_write> tile_counts: array<atomic<u32>>;
@compute @workgroup_size(32, 1, 1)
fn main() {
    let tile_global = 0u;
    let raw_count = 257u;
    atomicStore(&tile_counts[tile_global], raw_count);
}";
        let instrumented = benchmark_bin_source(source).unwrap();
        assert!(instrumented
            .contains("atomicMax(&tile_counts[arrayLength(&tile_counts) - 1u], raw_count);"));
        assert!(instrumented.contains("atomicStore(&tile_counts[tile_global], raw_count);"));
        validated_module(&instrumented);
        assert!(benchmark_bin_source(&format!("{source}\n{source}")).is_err());
    }

    #[test]
    fn tile_variants_must_declare_raster_work() {
        let bin = tile_source(TileStage::BinPolygons, [16, 16]);
        let raster = tile_source(TileStage::RasterizeError, [16, 16]);
        assert!(declares_binding(&bin, BIN_RASTER_WORK_BINDING).unwrap());
        assert!(declares_binding(&raster, RASTER_RASTER_WORK_BINDING).unwrap());
        // A mention outside a binding declaration does not make a source list work.
        let unbound = "// writes raster_work\nconst TILE_W: u32 = 16;\nconst TILE_H: u32 = 16;\n\
            @compute @workgroup_size(1) fn main() {}";
        assert!(!declares_binding(unbound, BIN_RASTER_WORK_BINDING).unwrap());
        assert!(validate_bin(unbound).is_err());
        assert!(validate_raster(unbound).is_err());
        let bound = format!(
            "@group(0) @binding({BIN_RASTER_WORK_BINDING}) var<storage, read_write> raster_work: array<u32>;\n{unbound}"
        );
        validate_bin(&bound).unwrap();
        assert!(declares_binding("fn", BIN_RASTER_WORK_BINDING).is_err());
    }
}
