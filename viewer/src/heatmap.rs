use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use wasm_bindgen::prelude::*;
use wasm_bindgen::Clamped;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

/// Decode a base64-encoded PNG into raw RGBA pixels + dimensions.
fn decode_png_b64(b64: &str) -> Option<(Vec<u8>, u32, u32)> {
    let bytes = BASE64.decode(b64).ok()?;
    let img = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png).ok()?;
    let rgba = img.to_rgba8();
    let w = rgba.width();
    let h = rgba.height();
    Some((rgba.into_raw(), w, h))
}

/// Compute per-pixel error heatmap between reference and generated images.
/// Decodes PNGs in Rust (synchronous, no DOM image elements).
/// Returns total error percent.
pub fn compute_heatmap(
    ref_b64: &str,
    gen_b64: &str,
    canvas: &HtmlCanvasElement,
) -> Option<f64> {
    let (ref_pixels, w, h) = decode_png_b64(ref_b64)?;
    let (gen_pixels, gw, gh) = decode_png_b64(gen_b64)?;

    // Use generated image dimensions if reference differs (shouldn't happen, but be safe)
    let (w, h) = if gw != w || gh != h { (gw, gh) } else { (w, h) };

    canvas.set_width(w);
    canvas.set_height(h);

    let pixel_count = (w * h) as usize;
    let mut error_pixels = vec![0u8; pixel_count * 4];
    let max_error = (255.0f64 * 255.0 * 3.0).sqrt();
    let mut total_error = 0.0f64;

    for i in 0..pixel_count {
        let idx = i * 4;
        let dr = ref_pixels[idx] as f64 - gen_pixels[idx] as f64;
        let dg = ref_pixels[idx + 1] as f64 - gen_pixels[idx + 1] as f64;
        let db = ref_pixels[idx + 2] as f64 - gen_pixels[idx + 2] as f64;
        let err = (dr * dr + dg * dg + db * db).sqrt();
        let normalized = err / max_error;
        total_error += normalized;

        let intensity = (normalized * 255.0) as u8;
        error_pixels[idx] = intensity;       // R
        error_pixels[idx + 1] = 0;           // G
        error_pixels[idx + 2] = 0;           // B
        error_pixels[idx + 3] = 255;         // A
    }

    let error_pct = (total_error / pixel_count as f64) * 100.0;

    let ctx: CanvasRenderingContext2d = canvas
        .get_context("2d")
        .ok()??
        .dyn_into()
        .ok()?;

    let img_data = web_sys::ImageData::new_with_u8_clamped_array_and_sh(
        Clamped(&error_pixels),
        w,
        h,
    )
    .ok()?;
    ctx.put_image_data(&img_data, 0.0, 0.0).ok()?;

    Some(error_pct)
}
