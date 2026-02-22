use leptos::prelude::*;

use crate::ws::ViewerState;

fn format_number(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn format_time(total_secs: f64) -> String {
    if total_secs < 60.0 {
        format!("{:.3}s", total_secs)
    } else if total_secs < 3600.0 {
        let mins = (total_secs / 60.0).floor() as u64;
        let secs = total_secs - (mins as f64 * 60.0);
        format!("{}m {:.3}s", mins, secs)
    } else {
        let hours = (total_secs / 3600.0).floor() as u64;
        let remainder = total_secs - (hours as f64 * 3600.0);
        let mins = (remainder / 60.0).floor() as u64;
        let secs = remainder - (mins as f64 * 60.0);
        format!("{}h {}m {:.3}s", hours, mins, secs)
    }
}

#[component]
pub fn Stats(state: RwSignal<ViewerState>) -> impl IntoView {
    view! {
        <div class="stats-section">
            <div class="stat-item stat-item-highlight">
                <div class="stat-value stat-value-similarity">{move || format!("{:.4}%", state.get().fitness)}</div>
                <div class="stat-label">"Similarity"</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{move || format!("{:.0}", state.get().evals_per_sec)}</div>
                <div class="stat-label">"Evals/sec"</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{move || format_number(state.get().total_evals)}</div>
                <div class="stat-label">"Total Evals"</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{move || format_number(state.get().improvements)}</div>
                <div class="stat-label">"Improvements"</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{move || {
                    let s = state.get();
                    if s.elapsed_secs > 0.0 {
                        format!("{:.1}", s.improvements as f64 / s.elapsed_secs)
                    } else {
                        "0".to_string()
                    }
                }}</div>
                <div class="stat-label">"Improv/sec"</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{move || format_time(state.get().elapsed_secs)}</div>
                <div class="stat-label">"Elapsed"</div>
            </div>
        </div>
    }
}
