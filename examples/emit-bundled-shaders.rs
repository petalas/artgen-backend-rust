//! Write the complete bundled shader sources, prelude included, to a new
//! directory. Stage files in `src/shaders/` are not complete WGSL on their own;
//! benchmark tools that take a source file (`engine-lab`, `check-gpu-mutation`,
//! `check-gpu-binning --shader`) need these instead. Tile-group stages are
//! emitted at 16x16, the shader variants' marker form, so the raster and binning
//! sources install as variants and specialize to the bundled source at any
//! tile. No GPU work. Run `cargo run --release --example emit-bundled-shaders -- NEWDIR`.

use std::fs;
use std::path::PathBuf;

use anyhow::{ensure, Context, Result};
use artgen_backend_rust::gpu_evolver::{bundled_shader_source, BundledShader, TileStage};

const USAGE: &str = "emit-bundled-shaders NEWDIR\nNEWDIR must not exist; its parent must.";
// The variants' tile-size markers name 16x16 (programs/variants.rs).
const MARKER_TILE: [u32; 2] = [16, 16];

fn main() -> Result<()> {
    // Two arguments suffice to reject every unsupported invocation.
    let args: Vec<_> = std::env::args_os().skip(1).take(2).collect();
    if args == ["--help"] {
        println!("{USAGE}");
        return Ok(());
    }
    ensure!(args.len() == 1, "{USAGE}");
    let output = PathBuf::from(&args[0]);
    // create_dir refuses an existing destination, so no file is overwritten.
    fs::create_dir(&output).with_context(|| format!("create {}", output.display()))?;
    let shaders = [BundledShader::Mutate, BundledShader::Select]
        .into_iter()
        .chain(TileStage::ALL.map(|stage| BundledShader::Tile(stage, MARKER_TILE)));
    for shader in shaders {
        let (name, _) = shader.file();
        let path = output.join(name);
        fs::write(&path, bundled_shader_source(shader))
            .with_context(|| format!("write {}", path.display()))?;
        println!("{}", path.display());
    }
    Ok(())
}
