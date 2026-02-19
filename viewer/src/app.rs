use leptos::prelude::*;

use crate::components::controls::Controls;
use crate::components::header::Header;
use crate::components::image_row::ImageRow;
use crate::components::stats::Stats;
use crate::ws::{connect_ws, ViewerState};

#[component]
pub fn App() -> impl IntoView {
    let state = RwSignal::new(ViewerState::default());

    // Connect WebSocket on mount
    connect_ws(state);

    view! {
        <div class="app">
            <Header state={state}/>
            <main class="main-content">
                <ImageRow state={state}/>
                <Stats state={state}/>
                <Controls state={state}/>
            </main>
        </div>
    }
}
