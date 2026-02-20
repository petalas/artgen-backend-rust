mod app;
mod canvas_renderer;
mod components;
mod heatmap;
mod models;
mod mutation_params;
mod ws;

fn main() {
    leptos::mount::mount_to_body(app::App);
}
