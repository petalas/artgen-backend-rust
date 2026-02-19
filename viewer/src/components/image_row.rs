use leptos::prelude::*;
use web_sys::HtmlCanvasElement;

use crate::heatmap::compute_heatmap;
use crate::ws::ViewerState;

#[component]
pub fn ImageRow(state: RwSignal<ViewerState>) -> impl IntoView {
    let canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    let error_pct = RwSignal::new(0.0f64);

    let ref_src = move || {
        let s = state.get();
        if s.reference_image_b64.is_empty() {
            String::new()
        } else {
            format!("data:image/png;base64,{}", s.reference_image_b64)
        }
    };

    let gen_src = move || {
        let s = state.get();
        if s.generated_image_b64.is_empty() {
            String::new()
        } else {
            format!("data:image/png;base64,{}", s.generated_image_b64)
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

    // Compute heatmap when generated image updates
    Effect::new(move || {
        let s = state.get();
        if s.reference_image_b64.is_empty() || s.generated_image_b64.is_empty() {
            return;
        }
        if let Some(canvas) = canvas_ref.get() {
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
                        format!("{} polygons", s.polygons)
                    }}</span>
                </div>
                <img class="panel-image" src={gen_src} alt="Generated image"/>
            </div>

            <div class="image-container">
                <div class="card-header">
                    <span class="card-title">"Error Heatmap"</span>
                    <span class="card-subtitle">{move || format!("Error: {}", error_label())}</span>
                </div>
                <canvas class="panel-canvas" node_ref={canvas_ref}></canvas>
                <div class="heatmap-gradient">
                    <span>"0%"</span>
                    <div class="gradient-bar"></div>
                    <span>"100%"</span>
                </div>
            </div>
        </div>
    }
}
