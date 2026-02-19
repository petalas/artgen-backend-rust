use wasm_bindgen::prelude::*;
use wasm_bindgen::Clamped;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, HtmlImageElement};

/// Compute per-pixel error heatmap between reference and generated images.
/// Returns (error_image_data_url, total_error_percent).
pub fn compute_heatmap(
    ref_b64: &str,
    gen_b64: &str,
    canvas: &HtmlCanvasElement,
) -> Option<f64> {
    let document = web_sys::window()?.document()?;

    let ref_img: HtmlImageElement = document.create_element("img").ok()?.dyn_into().ok()?;
    let gen_img: HtmlImageElement = document.create_element("img").ok()?.dyn_into().ok()?;

    ref_img.set_src(&format!("data:image/png;base64,{}", ref_b64));
    gen_img.set_src(&format!("data:image/png;base64,{}", gen_b64));

    let w = ref_img.natural_width();
    let h = ref_img.natural_height();

    if w == 0 || h == 0 {
        return None;
    }

    canvas.set_width(w);
    canvas.set_height(h);

    // Draw ref to offscreen canvas to get pixel data
    let offscreen: HtmlCanvasElement = document.create_element("canvas").ok()?.dyn_into().ok()?;
    offscreen.set_width(w);
    offscreen.set_height(h);
    let off_ctx: CanvasRenderingContext2d = offscreen
        .get_context("2d")
        .ok()??
        .dyn_into()
        .ok()?;

    off_ctx.draw_image_with_html_image_element(&ref_img, 0.0, 0.0).ok()?;
    let ref_data = off_ctx.get_image_data(0.0, 0.0, w as f64, h as f64).ok()?;
    let ref_pixels = ref_data.data();

    off_ctx.draw_image_with_html_image_element(&gen_img, 0.0, 0.0).ok()?;
    let gen_data = off_ctx.get_image_data(0.0, 0.0, w as f64, h as f64).ok()?;
    let gen_pixels = gen_data.data();

    let pixel_count = (w * h) as usize;
    let mut error_pixels = vec![0u8; pixel_count * 4];
    let max_error = (255.0f64 * 255.0 * 3.0).sqrt(); // sqrt(255^2 * 3)
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

    // Put heatmap onto visible canvas
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
