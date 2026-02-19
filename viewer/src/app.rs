use leptos::prelude::*;

use crate::components::controls::Controls;
use crate::components::header::Header;
use crate::components::image_row::ImageRow;
use crate::components::project_selector::ProjectsPage;
use crate::components::stats::Stats;
use crate::ws::{connect_ws, ViewerState};

#[component]
pub fn App() -> impl IntoView {
    let state = RwSignal::new(ViewerState::default());
    let page = RwSignal::new("evolution".to_string());

    // Connect WebSocket on mount
    connect_ws(state);

    view! {
        <div class="app">
            <Header state={state} page={page}/>
            <main class="main-content">
                {move || {
                    if page.get() == "projects" {
                        view! {
                            <ProjectsPage state={state} page={page}/>
                        }.into_any()
                    } else {
                        view! {
                            <ImageRow state={state}/>
                            <Stats state={state}/>
                            <Controls state={state}/>
                        }.into_any()
                    }
                }}
            </main>
        </div>
    }
}
