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

fn format_time(secs: u64) -> String {
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        let s = secs % 60;
        format!("{}h {}m {}s", h, m, s)
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
                    if s.elapsed_secs > 0 {
                        format!("{:.1}", s.improvements as f64 / s.elapsed_secs as f64)
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
