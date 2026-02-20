use leptos::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::HtmlCanvasElement;

use crate::benchmark::BenchmarkResult;

const COLORS: &[&str] = &["#009688", "#ff9800", "#1976d2", "#9c27b0", "#f44336", "#4caf50"];

pub fn color_for_index(i: usize) -> &'static str {
    COLORS[i % COLORS.len()]
}

#[component]
pub fn BenchmarkChart(results: Signal<Vec<BenchmarkResult>>) -> impl IntoView {
    let canvas_ref = NodeRef::<leptos::html::Canvas>::new();

    Effect::new(move |_| {
        let results = results.get();
        if results.is_empty() {
            return;
        }
        let Some(el) = canvas_ref.get() else { return };
        let canvas: HtmlCanvasElement = el.into();
        draw_chart(&canvas, &results);
    });

    view! {
        <canvas
            node_ref={canvas_ref}
            class="benchmark-chart-canvas"
            width="800"
            height="300"
        />
    }
}

fn draw_chart(canvas: &HtmlCanvasElement, results: &[BenchmarkResult]) {
    let ctx = canvas
        .get_context("2d")
        .ok()
        .flatten()
        .unwrap()
        .dyn_into::<web_sys::CanvasRenderingContext2d>()
        .unwrap();

    let w = canvas.width() as f64;
    let h = canvas.height() as f64;

    // Clear
    ctx.set_fill_style_str("#fff");
    ctx.fill_rect(0.0, 0.0, w, h);

    // Compute data ranges
    let max_time = results.iter().map(|r| r.duration_secs).max().unwrap_or(60) as f64;
    let mut min_fitness = f64::MAX;
    let mut max_fitness = f64::MIN;
    for r in results {
        for s in &r.samples {
            let f = s.best_fitness as f64;
            if f < min_fitness { min_fitness = f; }
            if f > max_fitness { max_fitness = f; }
        }
        let sf = r.start_fitness as f64;
        if sf < min_fitness { min_fitness = sf; }
    }

    if min_fitness >= max_fitness {
        max_fitness = min_fitness + 0.01;
    }

    // Add some padding to fitness range
    let range = max_fitness - min_fitness;
    min_fitness -= range * 0.05;
    max_fitness += range * 0.05;

    // Chart margins
    let left = 70.0;
    let right = 20.0;
    let top = 20.0;
    let bottom = 40.0;
    let cw = w - left - right;
    let ch = h - top - bottom;

    // Helper: data -> canvas coords
    let to_x = |t: f64| left + (t / max_time) * cw;
    let to_y = |f: f64| top + ch - ((f - min_fitness) / (max_fitness - min_fitness)) * ch;

    // Grid lines
    ctx.set_stroke_style_str("#eee");
    ctx.set_line_width(1.0);
    let y_ticks = 5;
    for i in 0..=y_ticks {
        let f = min_fitness + (max_fitness - min_fitness) * i as f64 / y_ticks as f64;
        let y = to_y(f);
        ctx.begin_path();
        ctx.move_to(left, y);
        ctx.line_to(w - right, y);
        ctx.stroke();

        // Y-axis label
        ctx.set_fill_style_str("#888");
        ctx.set_font("11px 'Ubuntu Mono', monospace");
        ctx.set_text_align("right");
        ctx.set_text_baseline("middle");
        ctx.fill_text(&format!("{:.2}%", f), left - 6.0, y).ok();
    }

    let x_ticks = 5.min(max_time as usize);
    for i in 0..=x_ticks {
        let t = max_time * i as f64 / x_ticks as f64;
        let x = to_x(t);
        ctx.begin_path();
        ctx.move_to(x, top);
        ctx.line_to(x, h - bottom);
        ctx.stroke();

        // X-axis label
        ctx.set_fill_style_str("#888");
        ctx.set_text_align("center");
        ctx.set_text_baseline("top");
        let mins = (t / 60.0).floor() as u32;
        let secs = (t % 60.0) as u32;
        ctx.fill_text(&format!("{}:{:02}", mins, secs), x, h - bottom + 6.0).ok();
    }

    // Axes
    ctx.set_stroke_style_str("#ccc");
    ctx.set_line_width(1.0);
    ctx.begin_path();
    ctx.move_to(left, top);
    ctx.line_to(left, h - bottom);
    ctx.line_to(w - right, h - bottom);
    ctx.stroke();

    // Draw lines for each result
    for (i, r) in results.iter().enumerate() {
        let color = color_for_index(i);
        ctx.set_stroke_style_str(color);
        ctx.set_line_width(2.0);
        ctx.begin_path();

        let mut first = true;
        for s in &r.samples {
            let x = to_x(s.elapsed_secs as f64);
            let y = to_y(s.best_fitness as f64);
            if first {
                ctx.move_to(x, y);
                first = false;
            } else {
                ctx.line_to(x, y);
            }
        }
        ctx.stroke();
    }

    // Legend
    let legend_x = left + 10.0;
    let mut legend_y = top + 14.0;
    for (i, r) in results.iter().enumerate() {
        let color = color_for_index(i);
        ctx.set_fill_style_str(color);
        ctx.fill_rect(legend_x, legend_y - 4.0, 16.0, 3.0);
        ctx.set_fill_style_str("#333");
        ctx.set_font("12px 'Ubuntu', sans-serif");
        ctx.set_text_align("left");
        ctx.set_text_baseline("middle");
        ctx.fill_text(&r.label, legend_x + 22.0, legend_y).ok();
        legend_y += 18.0;
    }
}
