use leptos::prelude::*;
use web_sys::HtmlCanvasElement;

use crate::canvas_renderer::render_drawing;
use crate::heatmap::compute_heatmap;
use crate::models::Drawing;
use crate::ws::ViewerState;

/// Hi-res canvas render size (longest side).
const RENDER_SIZE: u32 = 1024;

#[component]
pub fn ImageRow(state: RwSignal<ViewerState>) -> impl IntoView {
    let gen_canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    let heatmap_canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    let error_pct = RwSignal::new(0.0f64);

    let ref_src = move || {
        let s = state.get();
        if s.reference_image_b64.is_empty() {
            String::new()
        } else {
            format!("data:image/png;base64,{}", s.reference_image_b64)
        }
    };

    let dimensions_text = move || {
        let s = state.get();
        if s.image_width > 0 && s.image_height > 0 {
            format!("{}x{}", s.image_width, s.image_height)
        } else {
            String::new()
        }
    };

    // Render drawing on canvas when drawing_json updates
    Effect::new(move || {
        let s = state.get();
        let json = match &s.drawing_json {
            Some(j) => j.clone(),
            None => return,
        };
        let drawing: Drawing = match serde_json::from_str(&json) {
            Ok(d) => d,
            Err(_) => return,
        };
        if let Some(canvas) = gen_canvas_ref.get() {
            let canvas_el: &HtmlCanvasElement = &canvas;
            // Set canvas resolution based on image aspect ratio
            let (cw, ch) = if s.image_width > 0 && s.image_height > 0 {
                if s.image_width >= s.image_height {
                    (RENDER_SIZE, RENDER_SIZE * s.image_height / s.image_width)
                } else {
                    (RENDER_SIZE * s.image_width / s.image_height, RENDER_SIZE)
                }
            } else {
                (RENDER_SIZE, RENDER_SIZE)
            };
            canvas_el.set_width(cw);
            canvas_el.set_height(ch);
            render_drawing(&drawing, canvas_el);
        }
    });

    // Compute heatmap when generated image updates (still uses server PNGs for pixel-accurate diff)
    Effect::new(move || {
        let s = state.get();
        if s.reference_image_b64.is_empty() || s.generated_image_b64.is_empty() {
            return;
        }
        if let Some(canvas) = heatmap_canvas_ref.get() {
            let canvas_el: &HtmlCanvasElement = &canvas;
            if let Some(pct) = compute_heatmap(
                &s.reference_image_b64,
                &s.generated_image_b64,
                canvas_el,
            ) {
                error_pct.set(pct);
            }
        }
    });

    let error_label = move || format!("{:.2}%", error_pct.get());

    let render_size_text = move || {
        let s = state.get();
        if s.image_width > 0 && s.image_height > 0 {
            let (cw, ch) = if s.image_width >= s.image_height {
                (RENDER_SIZE, RENDER_SIZE * s.image_height / s.image_width)
            } else {
                (RENDER_SIZE * s.image_width / s.image_height, RENDER_SIZE)
            };
            format!("{}x{}", cw, ch)
        } else {
            String::new()
        }
    };

    view! {
        <div class="image-row">
            <div class="image-container">
                <div class="card-header">
                    <span class="card-title">"Original"</span>
                    <span class="card-subtitle">{dimensions_text}</span>
                </div>
                <img class="panel-image" src={ref_src} alt="Reference image"/>
            </div>

            <div class="image-container">
                <div class="card-header">
                    <span class="card-title">"Generated"</span>
                    <span class="card-subtitle">{move || {
                        let s = state.get();
                        format!("{} polygons \u{00B7} {}", s.polygons, render_size_text())
                    }}</span>
                </div>
                <canvas class="panel-canvas" node_ref={gen_canvas_ref}></canvas>
            </div>

            <div class="image-container">
                <div class="card-header">
                    <span class="card-title">"Error Heatmap"</span>
                    <span class="card-subtitle">{move || format!("Error: {}", error_label())}</span>
                </div>
                <canvas class="panel-canvas" node_ref={heatmap_canvas_ref}></canvas>
                <div class="heatmap-gradient">
                    <span>"0%"</span>
                    <div class="gradient-bar"></div>
                    <span>"100%"</span>
                </div>
            </div>
        </div>
    }
}
