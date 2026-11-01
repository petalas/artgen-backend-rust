use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use leptos::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::HtmlCanvasElement;

use crate::ws::ViewerState;

/// Seconds of server time the chart spans.
const WINDOW_SECS: f64 = 30.0;
/// Each plotted rate covers this much server time, ending at its point.
const RATE_SPAN_SECS: f64 = 1.0;
const CSS_HEIGHT: f64 = 80.0;

/// The server's counter at one of its own timestamps.
#[derive(Clone, Copy)]
struct Sample {
    elapsed: f64,
    improvements: u64,
}

/// Rates from the server's own clock: every stats and update message carries
/// the run's elapsed time with the improvement counter, so the rate stays
/// correct however late the browser handles the message.
#[component]
pub fn ImprovementChart(state: RwSignal<ViewerState>) -> impl IntoView {
    let canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    let samples: Rc<RefCell<VecDeque<Sample>>> = Rc::new(RefCell::new(VecDeque::new()));
    let latest = Memo::new(move |_| {
        state.with(|s| (s.elapsed_secs.to_bits(), s.improvements))
    });

    Effect::new(move || {
        let (elapsed, improvements) = latest.get();
        let sample = Sample { elapsed: f64::from_bits(elapsed), improvements };
        let mut buf = samples.borrow_mut();
        match buf.back() {
            // A new or restarted run: its clock or counter went back.
            Some(last) if sample.elapsed < last.elapsed || sample.improvements < last.improvements => {
                buf.clear();
                buf.push_back(sample);
            }
            // Paused, or a message from the same instant.
            Some(last) if sample.elapsed <= last.elapsed => return,
            _ => buf.push_back(sample),
        }
        // Keep one sample older than the window so its first rate has a start.
        let cutoff = sample.elapsed - WINDOW_SECS - RATE_SPAN_SECS;
        while buf.len() > 2 && buf[1].elapsed <= cutoff {
            buf.pop_front();
        }
        let rates = rates(&buf);
        if let Some(el) = canvas_ref.get() {
            let canvas: HtmlCanvasElement = el.into();
            draw_chart(&canvas, &rates, sample.elapsed);
        }
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

/// `(elapsed, rate)` at each sample with at least `RATE_SPAN_SECS` of history:
/// improvements per second since the latest sample that span earlier.
fn rates(samples: &VecDeque<Sample>) -> Vec<(f64, f64)> {
    let mut rates = Vec::with_capacity(samples.len());
    let mut start = 0;
    for end in 0..samples.len() {
        let point = samples[end];
        while start + 1 < end && samples[start + 1].elapsed <= point.elapsed - RATE_SPAN_SECS {
            start += 1;
        }
        let from = samples[start];
        if point.elapsed - from.elapsed >= RATE_SPAN_SECS {
            let count = (point.improvements - from.improvements) as f64;
            rates.push((point.elapsed, count / (point.elapsed - from.elapsed)));
        }
    }
    rates
}

fn draw_chart(canvas: &HtmlCanvasElement, rates: &[(f64, f64)], now: f64) {
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

    let visible: Vec<(f64, f64)> =
        rates.iter().copied().filter(|&(t, _)| t >= now - WINDOW_SECS).collect();
    let max_rate = visible.iter().map(|&(_, r)| r).fold(1.0_f64, f64::max);
    let current_rate = visible.last().map_or(0.0, |&(_, r)| r);

    let to_x = |t: f64| left + (1.0 - (now - t) / WINDOW_SECS) * cw;
    let to_y = |v: f64| top + ch - (v / max_rate) * ch;

    // Y-axis gridline at 50%
    ctx.set_stroke_style_str("rgba(0,0,0,0.06)");
    ctx.set_line_width(1.0);
    let y = to_y(max_rate * 0.5);
    ctx.begin_path();
    ctx.move_to(left, y);
    ctx.line_to(w - right, y);
    ctx.stroke();

    let (Some(&(first, _)), Some(&(last, _))) = (visible.first(), visible.last()) else {
        return;
    };

    // Area fill
    ctx.begin_path();
    ctx.move_to(to_x(first), to_y(0.0));
    for &(t, v) in &visible {
        ctx.line_to(to_x(t), to_y(v));
    }
    ctx.line_to(to_x(last), to_y(0.0));
    ctx.close_path();
    ctx.set_fill_style_str("rgba(46, 125, 50, 0.12)");
    ctx.fill();

    // Line
    ctx.begin_path();
    for (i, &(t, v)) in visible.iter().enumerate() {
        if i == 0 {
            ctx.move_to(to_x(t), to_y(v));
        } else {
            ctx.line_to(to_x(t), to_y(v));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn samples(points: &[(f64, u64)]) -> VecDeque<Sample> {
        points.iter().map(|&(elapsed, improvements)| Sample { elapsed, improvements }).collect()
    }

    /// 20 improvements per server second, delivered in irregular batches as a
    /// lagging browser would handle them, still reads 20/s.
    #[test]
    fn rates_follow_server_time_not_message_spacing() {
        let buf = samples(&[(10.0, 200), (10.2, 204), (10.25, 205), (11.0, 220), (14.0, 280), (14.1, 282)]);
        let rates = rates(&buf);
        let times: Vec<f64> = rates.iter().map(|&(t, _)| t).collect();
        assert_eq!(times, [11.0, 14.0, 14.1]);
        for (t, rate) in rates {
            assert!((rate - 20.0).abs() < 1e-9, "{t}: {rate}");
        }
    }

    #[test]
    fn a_sample_less_than_a_span_after_the_first_has_no_rate() {
        assert!(rates(&samples(&[(5.0, 10), (5.5, 20)])).is_empty());
    }
}
