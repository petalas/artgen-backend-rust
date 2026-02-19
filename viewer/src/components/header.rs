use leptos::prelude::*;

use crate::ws::ViewerState;

#[component]
pub fn Header(
    state: RwSignal<ViewerState>,
    page: RwSignal<String>,
) -> impl IntoView {
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

    let active_label = move || {
        state
            .get()
            .active_project
            .unwrap_or_else(|| "No project".to_string())
    };

    let go_evolution = move |_| page.set("evolution".into());
    let go_projects = move |_| page.set("projects".into());

    view! {
        <header class="app-header">
            <div class="header-left">
                <nav class="header-nav">
                    <a
                        class="nav-link"
                        class:active={move || page.get() == "evolution"}
                        on:click={go_evolution}
                    >
                        {active_label}
                    </a>
                    <a
                        class="nav-link"
                        class:active={move || page.get() == "projects"}
                        on:click={go_projects}
                    >
                        "Projects"
                    </a>
                </nav>
            </div>
            <div class="header-right">
                <div class={status_class}></div>
                <span class="status-text">{status_text}</span>
            </div>
        </header>
    }
}
