use std::path::{Path, PathBuf};

use image::imageops::FilterType::Lanczos3;
use image::{EncodableLayout, ImageEncoder};
use serde::{Deserialize, Serialize};

use crate::settings::{MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH, MIN_IMAGE_HEIGHT, MIN_IMAGE_WIDTH};

const PROJECTS_DIR: &str = "projects";

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ProjectInfo {
    pub name: String,
    pub has_best: bool,
    pub fitness: Option<f32>,
    pub polygons: Option<usize>,
}

fn projects_root() -> PathBuf {
    PathBuf::from(PROJECTS_DIR)
}

fn project_dir(name: &str) -> PathBuf {
    projects_root().join(name)
}

pub fn project_reference_path(name: &str) -> PathBuf {
    project_dir(name).join("reference.png")
}

pub fn project_best_json_path(name: &str) -> PathBuf {
    project_dir(name).join("best.json")
}

pub fn project_best_png_path(name: &str) -> PathBuf {
    project_dir(name).join("best.png")
}

fn validate_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("Project name cannot be empty".into());
    }
    if name.len() > 64 {
        return Err("Project name cannot exceed 64 characters".into());
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err("Project name can only contain letters, digits, hyphens, and underscores".into());
    }
    Ok(())
}

pub fn list_projects() -> Vec<ProjectInfo> {
    let root = projects_root();
    if !root.exists() {
        return vec![];
    }

    let mut projects = Vec::new();
    let Ok(entries) = std::fs::read_dir(&root) else {
        return vec![];
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()).map(String::from) else {
            continue;
        };
        // Must have reference.png to be a valid project
        if !project_reference_path(&name).exists() {
            continue;
        }

        let best_path = project_best_json_path(&name);
        let has_best = best_path.exists();
        let (fitness, polygons) = if has_best {
            read_best_summary(&best_path)
        } else {
            (None, None)
        };

        projects.push(ProjectInfo {
            name,
            has_best,
            fitness,
            polygons,
        });
    }

    projects.sort_by(|a, b| a.name.cmp(&b.name));
    projects
}

fn read_best_summary(path: &Path) -> (Option<f32>, Option<usize>) {
    let Ok(data) = std::fs::read_to_string(path) else {
        return (None, None);
    };
    let Ok(val) = serde_json::from_str::<serde_json::Value>(&data) else {
        return (None, None);
    };
    let fitness = val["fitness"].as_f64().map(|f| f as f32);
    let polygons = val["polygons"].as_array().map(|a| a.len());
    (fitness, polygons)
}

/// Decode raw image bytes, resize to engine bounds, return RGBA pixels + dimensions.
pub fn load_and_normalize_image(raw_bytes: &[u8]) -> Result<(Vec<u8>, usize, usize), String> {
    let img = image::load_from_memory(raw_bytes).map_err(|e| format!("Failed to decode image: {}", e))?;

    let mut w = img.width() as usize;
    let mut h = img.height() as usize;

    let resized = if w < MIN_IMAGE_WIDTH || h < MIN_IMAGE_HEIGHT || w > MAX_IMAGE_WIDTH || h > MAX_IMAGE_HEIGHT {
        let target_w = w.clamp(MIN_IMAGE_WIDTH, MAX_IMAGE_WIDTH) as u32;
        let target_h = h.clamp(MIN_IMAGE_HEIGHT, MAX_IMAGE_HEIGHT) as u32;
        let r = img.resize(target_w, target_h, Lanczos3);
        w = r.width() as usize;
        h = r.height() as usize;
        r
    } else {
        img
    };

    let rgba = resized.into_rgba8();
    Ok((rgba.as_bytes().to_vec(), w, h))
}

pub fn create_project(name: &str, image_bytes: &[u8]) -> Result<ProjectInfo, String> {
    validate_name(name)?;

    let dir = project_dir(name);
    if dir.exists() {
        return Err(format!("Project '{}' already exists", name));
    }

    // Decode and normalize before creating any dirs, so we fail early on bad images
    let (rgba, w, h) = load_and_normalize_image(image_bytes)?;

    std::fs::create_dir_all(&dir).map_err(|e| format!("Failed to create project directory: {}", e))?;

    // Save original uploaded bytes
    let original_path = dir.join("reference.original");
    std::fs::write(&original_path, image_bytes)
        .map_err(|e| format!("Failed to save original image: {}", e))?;

    // Save normalized RGBA as PNG
    let ref_path = project_reference_path(name);
    let mut png_buf = Vec::new();
    image::codecs::png::PngEncoder::new(&mut png_buf)
        .write_image(&rgba, w as u32, h as u32, image::ExtendedColorType::Rgba8)
        .map_err(|e| format!("Failed to encode reference PNG: {}", e))?;
    std::fs::write(&ref_path, &png_buf).map_err(|e| format!("Failed to save reference PNG: {}", e))?;

    Ok(ProjectInfo {
        name: name.to_string(),
        has_best: false,
        fitness: None,
        polygons: None,
    })
}

pub fn rename_project(old_name: &str, new_name: &str) -> Result<(), String> {
    validate_name(old_name)?;
    validate_name(new_name)?;
    let old_dir = project_dir(old_name);
    if !old_dir.exists() {
        return Err(format!("Project '{}' does not exist", old_name));
    }
    let new_dir = project_dir(new_name);
    if new_dir.exists() {
        return Err(format!("Project '{}' already exists", new_name));
    }
    std::fs::rename(&old_dir, &new_dir).map_err(|e| format!("Failed to rename project: {}", e))?;
    Ok(())
}

pub fn delete_project(name: &str) -> Result<(), String> {
    validate_name(name)?;
    let dir = project_dir(name);
    if !dir.exists() {
        return Err(format!("Project '{}' does not exist", name));
    }
    std::fs::remove_dir_all(&dir).map_err(|e| format!("Failed to delete project: {}", e))?;
    Ok(())
}

pub fn reset_project(name: &str) -> Result<(), String> {
    validate_name(name)?;
    let dir = project_dir(name);
    if !dir.exists() {
        return Err(format!("Project '{}' does not exist", name));
    }

    let best_json = project_best_json_path(name);
    if best_json.exists() {
        std::fs::remove_file(&best_json).map_err(|e| format!("Failed to remove best.json: {}", e))?;
    }

    let best_png = project_best_png_path(name);
    if best_png.exists() {
        std::fs::remove_file(&best_png).map_err(|e| format!("Failed to remove best.png: {}", e))?;
    }

    Ok(())
}

pub fn import_drawing(name: &str, json_bytes: &[u8]) -> Result<(), String> {
    validate_name(name)?;
    let dir = project_dir(name);
    if !dir.exists() {
        return Err(format!("Project '{}' does not exist", name));
    }

    // Validate that it's a valid Drawing
    let json_str =
        std::str::from_utf8(json_bytes).map_err(|e| format!("Invalid UTF-8 in drawing JSON: {}", e))?;
    serde_json::from_str::<crate::models::drawing::Drawing>(json_str)
        .map_err(|e| format!("Invalid drawing JSON: {}", e))?;

    let best_path = project_best_json_path(name);
    std::fs::write(&best_path, json_bytes).map_err(|e| format!("Failed to write best.json: {}", e))?;

    Ok(())
}

/// Ensure the projects directory exists.
pub fn ensure_projects_dir() {
    let root = projects_root();
    if !root.exists() {
        std::fs::create_dir_all(&root).expect("Failed to create projects directory");
    }
}

/// Migrate old-style files to a default project if projects/ is empty.
/// Looks for files matching *.jpg, *.png, *.jpeg in the current directory
/// that have a corresponding .best.json file.
pub fn migrate_legacy(legacy_image: Option<&str>) -> Option<String> {
    ensure_projects_dir();

    // If any projects already exist, skip migration
    if !list_projects().is_empty() {
        return None;
    }

    // Try the specified legacy image first
    if let Some(img_path) = legacy_image {
        let path = Path::new(img_path);
        if path.exists() {
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("default");
            let project_name = sanitize_name(stem);
            let image_bytes = match std::fs::read(path) {
                Ok(b) => b,
                Err(_) => return None,
            };
            if create_project(&project_name, &image_bytes).is_ok() {
                // Move best.json if it exists
                let best_json = format!("{}.best.json", stem);
                if Path::new(&best_json).exists() {
                    let dest = project_best_json_path(&project_name);
                    std::fs::copy(&best_json, &dest).ok();
                }
                println!("[Projects] Migrated '{}' -> project '{}'", img_path, project_name);
                return Some(project_name);
            }
        }
    }

    // Scan for any image files with a best.json
    let scan_extensions = ["jpg", "jpeg", "png"];
    for entry in std::fs::read_dir(".").into_iter().flatten().flatten() {
        let path = entry.path();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        if !scan_extensions.contains(&ext.as_str()) {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("default");
        let best_json = format!("{}.best.json", stem);
        if !Path::new(&best_json).exists() {
            continue;
        }
        let project_name = sanitize_name(stem);
        let Ok(image_bytes) = std::fs::read(&path) else {
            continue;
        };
        if create_project(&project_name, &image_bytes).is_ok() {
            let dest = project_best_json_path(&project_name);
            std::fs::copy(&best_json, &dest).ok();
            println!(
                "[Projects] Migrated '{}' + '{}' -> project '{}'",
                path.display(),
                best_json,
                project_name
            );
            return Some(project_name);
        }
    }

    None
}

fn sanitize_name(s: &str) -> String {
    let sanitized: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "default".to_string()
    } else {
        sanitized[..sanitized.len().min(64)].to_string()
    }
}
