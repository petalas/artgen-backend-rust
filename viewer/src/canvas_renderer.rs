use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

use crate::models::Drawing;

/// Render a Drawing onto an HTML canvas at the canvas's current dimensions.
/// Polygons use normalized 0-1 coordinates, scaled to canvas size.
/// Uses Canvas 2D "source-over" compositing which matches the GPU's alpha blending.
pub fn render_drawing(drawing: &Drawing, canvas: &HtmlCanvasElement) -> bool {
    let w = canvas.width() as f64;
    let h = canvas.height() as f64;
    if w <= 0.0 || h <= 0.0 {
        return false;
    }

    let ctx: CanvasRenderingContext2d = match canvas.get_context("2d") {
        Ok(Some(ctx)) => match ctx.dyn_into() {
            Ok(ctx) => ctx,
            Err(_) => return false,
        },
        _ => return false,
    };

    // White background
    ctx.set_fill_style_str("white");
    ctx.fill_rect(0.0, 0.0, w, h);

    // Draw each polygon
    for polygon in &drawing.polygons {
        if polygon.points.len() < 3 {
            continue;
        }

        let c = &polygon.color;
        let alpha = c.a as f64 / 255.0;
        ctx.set_fill_style_str(&format!("rgba({},{},{},{})", c.r, c.g, c.b, alpha));

        ctx.begin_path();
        let p0 = &polygon.points[0];
        ctx.move_to(p0.x as f64 * w, p0.y as f64 * h);
        for p in &polygon.points[1..] {
            ctx.line_to(p.x as f64 * w, p.y as f64 * h);
        }
        ctx.close_path();
        ctx.fill();
    }

    true
}
