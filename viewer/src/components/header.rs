use leptos::prelude::*;

use crate::components::project_selector::ProjectSelector;
use crate::ws::ViewerState;

#[component]
pub fn Header(state: RwSignal<ViewerState>) -> impl IntoView {
    let fitness_pct = move || {
        let s = state.get();
        format!("{:.4}%", s.fitness)
    };

    let status_class = move || {
        let s = state.get();
        if s.connected {
            "status-dot connected"
        } else if s.connecting {
            "status-dot connecting"
        } else {
            "status-dot disconnected"
        }
    };

    let status_text = move || {
        let s = state.get();
        if s.connected {
            "Connected"
        } else if s.connecting {
            "Connecting..."
        } else {
            "Disconnected"
        }
    };

    view! {
        <header class="app-header">
            <div class="header-left">
                <ProjectSelector state={state} />
            </div>
            <div class="header-center">
                <span class="similarity-label">"Similarity: "</span>
                <span class="similarity-value">{fitness_pct}</span>
            </div>
            <div class="header-right">
                <div class={status_class}></div>
                <span class="status-text">{status_text}</span>
            </div>
        </header>
    }
}
