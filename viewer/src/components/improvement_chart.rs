use std::cell::RefCell;
use std::rc::Rc;

use leptos::prelude::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::HtmlCanvasElement;

use crate::ws::ViewerState;

const WINDOW_SECS: usize = 30;
// We need WINDOW_SECS + 1 samples to compute WINDOW_SECS deltas
const MAX_SAMPLES: usize = WINDOW_SECS + 1;
const CSS_HEIGHT: f64 = 80.0;

struct Sample {
    improvements: u64,
}

#[component]
pub fn ImprovementChart(state: RwSignal<ViewerState>) -> impl IntoView {
    let canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    let samples: Rc<RefCell<Vec<Sample>>> = Rc::new(RefCell::new(Vec::with_capacity(MAX_SAMPLES)));

    // Tick every 1 second: record current improvements count, then redraw
    let samples_tick = Rc::clone(&samples);
    let canvas_tick = canvas_ref;
    Effect::new(move |_| {
        let samples = Rc::clone(&samples_tick);
        let canvas_ref = canvas_tick;

        let cb = Closure::<dyn Fn()>::new(move || {
            let improvements = state.with_untracked(|s| s.improvements);
            let mut buf = samples.borrow_mut();
            buf.push(Sample { improvements });
            let excess = buf.len().saturating_sub(MAX_SAMPLES);
            if excess > 0 {
                buf.drain(..excess);
            }

            // Compute rates from consecutive samples
            let rates: Vec<f64> = buf.windows(2)
                .map(|w| (w[1].improvements as f64 - w[0].improvements as f64).max(0.0))
                .collect();

            // Draw
            if let Some(el) = canvas_ref.get() {
                let canvas: HtmlCanvasElement = el.into();
                draw_chart(&canvas, &rates);
            }
        });

        let window = web_sys::window().unwrap();
        let id = window
            .set_interval_with_callback_and_timeout_and_arguments_0(
                cb.as_ref().unchecked_ref(),
                1000,
            )
            .unwrap();
        cb.forget();

        on_cleanup(move || {
            if let Some(w) = web_sys::window() {
                w.clear_interval_with_handle(id);
            }
        });
    });

    view! {
        <div class="improvement-chart-container">
            <canvas
                node_ref={canvas_ref}
                class="improvement-chart-canvas"
            />
        </div>
    }
}

fn draw_chart(canvas: &HtmlCanvasElement, rates: &[f64]) {
    let ctx = match canvas.get_context("2d").ok().flatten() {
        Some(c) => c.dyn_into::<web_sys::CanvasRenderingContext2d>().unwrap(),
        None => return,
    };

    // Match canvas backing resolution to CSS display size for crisp rendering
    let dpr = web_sys::window().map(|w| w.device_pixel_ratio()).unwrap_or(1.0);
    let css_w = canvas.client_width() as f64;
    let backing_w = (css_w * dpr) as u32;
    let backing_h = (CSS_HEIGHT * dpr) as u32;
    if canvas.width() != backing_w || canvas.height() != backing_h {
        canvas.set_width(backing_w);
        canvas.set_height(backing_h);
    }

    // Scale context so we can draw in CSS-pixel coordinates
    ctx.set_transform(dpr, 0.0, 0.0, dpr, 0.0, 0.0).ok();
    let w = css_w;
    let h = CSS_HEIGHT;

    // Clear
    ctx.clear_rect(0.0, 0.0, w, h);

    if rates.is_empty() {
        ctx.set_fill_style_str("#888");
        ctx.set_font("12px 'Ubuntu', sans-serif");
        ctx.set_text_align("center");
        ctx.set_text_baseline("middle");
        ctx.fill_text("Collecting data\u{2026}", w / 2.0, h / 2.0).ok();
        return;
    }

    let left = 44.0;
    let right = 8.0;
    let top = 16.0;
    let bottom = 18.0;
    let cw = w - left - right;
    let ch = h - top - bottom;

    // Pad rates to WINDOW_SECS length (prepend zeros for the unfilled portion)
    let n = WINDOW_SECS;
    let padded: Vec<f64> = if rates.len() < n {
        let mut v = vec![0.0; n - rates.len()];
        v.extend_from_slice(rates);
        v
    } else {
        rates[rates.len() - n..].to_vec()
    };

    let max_rate = padded.iter().cloned().fold(1.0_f64, f64::max);
    let current_rate = padded.last().copied().unwrap_or(0.0);

    let to_x = |i: usize| left + (i as f64 / (n - 1) as f64) * cw;
    let to_y = |v: f64| top + ch - (v / max_rate) * ch;

    // Y-axis gridline at 50%
    ctx.set_stroke_style_str("rgba(0,0,0,0.06)");
    ctx.set_line_width(1.0);
    let y = to_y(max_rate * 0.5);
    ctx.begin_path();
    ctx.move_to(left, y);
    ctx.line_to(w - right, y);
    ctx.stroke();

    // Area fill
    ctx.begin_path();
    ctx.move_to(to_x(0), to_y(0.0));
    for (i, &v) in padded.iter().enumerate() {
        ctx.line_to(to_x(i), to_y(v));
    }
    ctx.line_to(to_x(n - 1), to_y(0.0));
    ctx.close_path();
    ctx.set_fill_style_str("rgba(46, 125, 50, 0.12)");
    ctx.fill();

    // Line
    ctx.begin_path();
    for (i, &v) in padded.iter().enumerate() {
        if i == 0 {
            ctx.move_to(to_x(i), to_y(v));
        } else {
            ctx.line_to(to_x(i), to_y(v));
        }
    }
    ctx.set_stroke_style_str("#2e7d32");
    ctx.set_line_width(1.5);
    ctx.stroke();

    // Current rate text (top-right)
    ctx.set_fill_style_str("#2e7d32");
    ctx.set_font("bold 12px 'Ubuntu Mono', monospace");
    ctx.set_text_align("right");
    ctx.set_text_baseline("top");
    ctx.fill_text(&format!("{:.1}/s", current_rate), w - right, 1.0).ok();

    // Label (top-left)
    ctx.set_fill_style_str("#888");
    ctx.set_font("11px 'Ubuntu', sans-serif");
    ctx.set_text_align("left");
    ctx.set_text_baseline("top");
    ctx.fill_text("Improvements/sec (30s)", left, 1.0).ok();

    // Y-axis: 0 and max labels
    ctx.set_fill_style_str("#aaa");
    ctx.set_font("10px 'Ubuntu Mono', monospace");
    ctx.set_text_align("right");
    ctx.set_text_baseline("bottom");
    ctx.fill_text("0", left - 4.0, top + ch).ok();
    ctx.set_text_baseline("top");
    ctx.fill_text(&format!("{:.0}", max_rate), left - 4.0, top).ok();

    // X-axis: "-30s" and "now"
    ctx.set_fill_style_str("#aaa");
    ctx.set_font("10px 'Ubuntu', sans-serif");
    ctx.set_text_align("left");
    ctx.set_text_baseline("top");
    ctx.fill_text("-30s", left, top + ch + 3.0).ok();
    ctx.set_text_align("right");
    ctx.fill_text("now", w - right, top + ch + 3.0).ok();
}
