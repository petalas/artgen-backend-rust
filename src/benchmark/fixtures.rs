//! Immutable, local copies of mature project inputs. No GPU work occurs here.
use std::{
    fs,
    io::{Cursor, Read},
    path::{Component, Path},
};

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::reference::{self, Complexity};
use crate::{
    gpu_evolver::buffers::drawing_to_gpu, models::drawing::Drawing,
    mutation_params::MutationParams, projects, settings::MAX_POLYGONS_PER_IMAGE,
};

pub const PROTOCOL_VERSION: u32 = 1;
pub const PROJECTS: [&str; 3] = ["ff", "mona", "darwin"];
const MAX_DRAWING_BYTES: u64 = 2 * 1024 * 1024;
const MAX_IMAGE_BYTES: u64 = 32 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub protocol_version: u32,
    pub start_mode: String,
    pub resolution: u32,
    pub minimum_similarity_percent: f32,
    pub params: MutationParams,
    pub cases: Vec<Fixture>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fixture {
    pub id: String,
    pub width: u32,
    pub height: u32,
    pub drawing_file: String,
    pub reference_file: String,
    pub drawing_sha256: String,
    pub reference_sha256: String,
    pub source_image_sha256: String,
    pub stored_fitness_percent: f32,
    pub scalar_fitness_percent: f32,
    pub complexity: Complexity,
}

pub struct LoadedFixture {
    pub manifest: Manifest,
    pub fixture: Fixture,
    pub drawing: Drawing,
    pub rgba: Vec<u8>,
    pub manifest_sha256: String,
}

#[must_use]
pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// # Errors
/// Rejects unreadable files and inputs larger than the declared limit.
pub fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let file = fs::File::open(path).with_context(|| format!("read {}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= limit,
        "{} exceeds {limit} bytes",
        path.display()
    );
    Ok(bytes)
}

/// # Errors
/// Rejects invalid sources, unsupported dimensions and an existing destination.
pub fn freeze(projects_dir: &Path, output: &Path, resolution: u32) -> Result<Manifest> {
    ensure!(
        (64..=1024).contains(&resolution),
        "resolution must be 64..1024"
    );
    ensure!(
        !output.exists(),
        "fixture destination already exists; inputs are immutable"
    );
    let mut cases = Vec::with_capacity(PROJECTS.len());
    // Validate all source files before creating the destination.
    let inputs: Vec<_> = PROJECTS
        .iter()
        .map(|id| import(projects_dir, id, resolution))
        .collect::<Result<_>>()?;
    fs::create_dir_all(output)?;
    for (fixture, drawing_bytes, rgba) in inputs {
        let case_dir = output.join(&fixture.id);
        fs::create_dir(&case_dir)?;
        fs::write(output.join(&fixture.drawing_file), drawing_bytes)?;
        fs::write(output.join(&fixture.reference_file), rgba)?;
        cases.push(fixture);
    }
    let manifest = Manifest {
        protocol_version: PROTOCOL_VERSION,
        start_mode: "mature_drawing_restart".into(),
        resolution,
        minimum_similarity_percent: 97.0,
        params: MutationParams::default(),
        cases,
    };
    fs::write(
        output.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(manifest)
}

fn import(root: &Path, id: &str, resolution: u32) -> Result<(Fixture, Vec<u8>, Vec<u8>)> {
    let dir = root.join(id);
    let drawing_bytes = read_bounded(&dir.join("best.json"), MAX_DRAWING_BYTES)?;
    let drawing: Drawing = serde_json::from_slice(&drawing_bytes)?;
    validate_drawing(&drawing)?;
    let original = dir.join("reference.original");
    let source = if original.exists() {
        original
    } else {
        dir.join("reference.png")
    };
    let image_bytes = read_bounded(&source, MAX_IMAGE_BYTES)?;
    let reader = image::ImageReader::new(Cursor::new(&image_bytes)).with_guessed_format()?;
    let (source_w, source_h) = reader.into_dimensions()?;
    ensure!(
        source_w <= 8192 && source_h <= 8192,
        "source dimensions exceed 8192"
    );
    let (rgba, width, height) = projects::load_and_normalize_image(&image_bytes, Some(resolution))
        .map_err(anyhow::Error::msg)?;
    let (width, height) = (u32::try_from(width)?, u32::try_from(height)?);
    let packed = drawing_to_gpu(&drawing, 0);
    let polygons: Vec<_> = packed.polygons[..packed.polygon_count as usize]
        .iter()
        .map(|p| p.data)
        .collect();
    let complexity = reference::complexity(
        &polygons,
        width,
        height,
        MutationParams::default().rasterize_wg,
    )?;
    let evaluation = reference::evaluate(&polygons, &rgba, width, height)?;
    let fixture = Fixture {
        id: id.into(),
        width,
        height,
        drawing_file: format!("{id}/drawing.json"),
        reference_file: format!("{id}/reference.rgba"),
        drawing_sha256: sha256(&drawing_bytes),
        reference_sha256: sha256(&rgba),
        source_image_sha256: sha256(&image_bytes),
        stored_fitness_percent: drawing.fitness,
        scalar_fitness_percent: evaluation.fitness,
        complexity,
    };
    Ok((fixture, drawing_bytes, rgba))
}

/// # Errors
/// Rejects unsupported manifests, malformed drawings and changed fixture bytes.
pub fn load(manifest_path: &Path, id: &str) -> Result<LoadedFixture> {
    let bytes = read_bounded(manifest_path, MAX_DRAWING_BYTES)?;
    let manifest: Manifest = serde_json::from_slice(&bytes)?;
    validate_manifest(&manifest)?;
    let fixture = manifest
        .cases
        .iter()
        .find(|case| case.id == id)
        .with_context(|| format!("unknown case {id}"))?
        .clone();
    let root = manifest_path.parent().context("manifest has no parent")?;
    let drawing_bytes = read_fixture(root, &fixture.drawing_file, MAX_DRAWING_BYTES)?;
    let rgba = read_fixture(root, &fixture.reference_file, 4 * 1024 * 1024)?;
    ensure!(
        sha256(&drawing_bytes) == fixture.drawing_sha256,
        "drawing hash mismatch"
    );
    ensure!(
        sha256(&rgba) == fixture.reference_sha256,
        "reference hash mismatch"
    );
    ensure!(
        rgba.len() == fixture.width as usize * fixture.height as usize * 4,
        "reference dimensions/length mismatch"
    );
    let drawing = serde_json::from_slice(&drawing_bytes)?;
    validate_drawing(&drawing)?;
    Ok(LoadedFixture {
        manifest,
        fixture,
        drawing,
        rgba,
        manifest_sha256: sha256(&bytes),
    })
}

fn read_fixture(root: &Path, relative: &str, limit: u64) -> Result<Vec<u8>> {
    let path = Path::new(relative);
    ensure!(
        path.components().all(|c| matches!(c, Component::Normal(_))),
        "fixture paths must be relative without traversal"
    );
    read_bounded(&root.join(path), limit)
}

fn validate_manifest(manifest: &Manifest) -> Result<()> {
    ensure!(
        manifest.protocol_version == PROTOCOL_VERSION,
        "unsupported protocol"
    );
    ensure!(
        manifest.start_mode == "mature_drawing_restart",
        "unsupported start mode"
    );
    ensure!(
        manifest.minimum_similarity_percent.to_bits() == 97.0f32.to_bits(),
        "similarity requirement changed"
    );
    ensure!(
        (64..=1024).contains(&manifest.resolution),
        "invalid resolution"
    );
    ensure!(
        manifest.cases.len() == PROJECTS.len(),
        "all three projects are required"
    );
    let mut sanitized = manifest.params.clone();
    sanitized.sanitize();
    ensure!(
        sanitized == manifest.params,
        "parameters would be silently sanitized"
    );
    for id in PROJECTS {
        ensure!(
            manifest.cases.iter().filter(|case| case.id == id).count() == 1,
            "missing or duplicate project {id}"
        );
    }
    for case in &manifest.cases {
        ensure!(
            (2..=1024).contains(&case.width) && (2..=1024).contains(&case.height),
            "invalid fixture dimensions"
        );
        ensure!(
            case.width.max(case.height) == manifest.resolution,
            "fixture resolution differs from the manifest"
        );
    }
    Ok(())
}

fn validate_drawing(drawing: &Drawing) -> Result<()> {
    ensure!(
        drawing.polygons.len() <= MAX_POLYGONS_PER_IMAGE,
        "too many polygons"
    );
    let mut triangles = 0;
    for polygon in &drawing.polygons {
        ensure!(
            (3..=MAX_POLYGONS_PER_IMAGE + 2).contains(&polygon.points.len()),
            "invalid polygon vertex count"
        );
        triangles += polygon.points.len() - 2;
        ensure!(
            triangles <= MAX_POLYGONS_PER_IMAGE,
            "packing would truncate triangles"
        );
        for point in &polygon.points {
            if !point.x.is_finite()
                || !point.y.is_finite()
                || !(0.0..=1.0).contains(&point.x)
                || !(0.0..=1.0).contains(&point.y)
            {
                bail!("invalid normalized vertex");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_has_standard_known_answer() {
        assert_eq!(
            sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn fixture_paths_cannot_escape_root() {
        assert!(read_fixture(Path::new("."), "../Cargo.toml", 100).is_err());
        assert!(read_fixture(Path::new("."), "/etc/passwd", 100).is_err());
    }

    #[test]
    fn frozen_f64_metadata_round_trips_without_changing_bits() {
        // This observed bbox p95 shifted by one ULP with Serde's fast parser,
        // causing the Python supervisor to reject otherwise valid GPU trials.
        let bytes = b"0.19420557263631666";
        let parsed: f64 = serde_json::from_slice(bytes).unwrap();
        assert_eq!(parsed.to_bits(), 0.19420557263631666f64.to_bits());
        assert_eq!(serde_json::to_vec(&parsed).unwrap(), bytes);
    }

    #[test]
    fn frozen_inputs_are_immutable_and_corruption_is_rejected() {
        let root = std::env::temp_dir().join(format!("artgen-fixtures-{}", uuid::Uuid::new_v4()));
        let source = root.join("projects");
        let destination = root.join("frozen");
        let result = fixture_round_trip(&source, &destination);
        fs::remove_dir_all(&root).expect("remove fixture test inputs");
        result.expect("freeze and hash verification");
    }

    fn fixture_round_trip(source: &Path, destination: &Path) -> Result<()> {
        let drawing = serde_json::json!({"polygons": [], "isDirty": false, "fitness": 100.0});
        for id in PROJECTS {
            let dir = source.join(id);
            fs::create_dir_all(&dir)?;
            fs::write(dir.join("best.json"), serde_json::to_vec(&drawing)?)?;
            image::RgbaImage::from_pixel(4, 4, image::Rgba([255, 255, 255, 255]))
                .save(dir.join("reference.png"))?;
        }
        freeze(source, destination, 384)?;
        let manifest = destination.join("manifest.json");
        ensure!(load(&manifest, "ff").is_ok(), "valid frozen fixture failed");
        ensure!(
            freeze(source, destination, 384).is_err(),
            "immutable output overwritten"
        );
        fs::write(destination.join("ff/reference.rgba"), b"corrupt")?;
        let error = load(&manifest, "ff")
            .err()
            .context("corruption was accepted")?;
        ensure!(
            error.to_string().contains("hash mismatch"),
            "wrong corruption failure"
        );
        Ok(())
    }
}
