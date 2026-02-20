use leptos::prelude::*;
use wasm_bindgen::JsCast;

use crate::mutation_params::MutationParams;
use crate::ws::{send_ws_json, ViewerState};

/// Format a probability as "1 in N" string.
fn format_prob(p: f32) -> String {
    if p <= 0.0 {
        "OFF".to_string()
    } else {
        let n = (1.0 / p).round() as u32;
        if n <= 1 {
            "1 in 1".to_string()
        } else {
            format!("1 in {}", n)
        }
    }
}

/// Convert a probability to a log-scale slider value.
/// Maps prob range [1/10000, 1/1] to slider [0, 1000].
/// Left (0) = rare, right (1000) = frequent.
fn prob_to_slider(p: f32) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    let inv = 1.0 / p;
    let log_val = inv.ln() as f64;
    let max_log = 10000_f64.ln(); // ≈ 9.21
    (1000.0 - (log_val / max_log) * 1000.0).clamp(0.0, 1000.0)
}

/// Convert a log-scale slider value back to a probability.
fn slider_to_prob(v: f64) -> f32 {
    let max_log = 10000_f64.ln();
    let log_val = ((1000.0 - v) / 1000.0) * max_log;
    (1.0 / log_val.exp()) as f32
}

fn send_params(params: &MutationParams) {
    send_ws_json(&serde_json::json!({
        "type": "update_params",
        "params": params,
    }));
}

#[component]
pub fn MutationPanel(state: RwSignal<ViewerState>) -> impl IntoView {
    let collapsed = RwSignal::new(true);

    let toggle = move |_| {
        collapsed.set(!collapsed.get());
    };

    view! {
        <div class="mutation-panel">
            <div class="mutation-panel-header" on:click={toggle}>
                <span class="mutation-panel-toggle">{move || if collapsed.get() { "\u{25B6}" } else { "\u{25BC}" }}</span>
                <span class="mutation-panel-title">"Evolution Parameters"</span>
            </div>
            {move || {
                if collapsed.get() {
                    view! { <div></div> }.into_any()
                } else {
                    view! { <MutationPanelBody state={state}/> }.into_any()
                }
            }}
        </div>
    }
}

#[component]
fn MutationPanelBody(state: RwSignal<ViewerState>) -> impl IntoView {
    let on_reset = move |_| {
        send_ws_json(&serde_json::json!({ "type": "reset_params" }));
        state.update(|s| s.mutation_params = MutationParams::default());
    };

    view! {
        <div class="mutation-panel-body">
            <div class="mutation-section">
                <div class="mutation-section-title">"Polygons"</div>
                <IntSlider state={state} label="Min polygons" get={|mp| mp.min_polygons as i64} set={|mp, v| { mp.min_polygons = v as u32; if mp.max_polygons < mp.min_polygons { mp.max_polygons = mp.min_polygons; } }} min=1 max=1000 step=1/>
                <IntSlider state={state} label="Max polygons" get={|mp| mp.max_polygons as i64} set={|mp, v| { mp.max_polygons = v as u32; if mp.min_polygons > mp.max_polygons { mp.min_polygons = mp.max_polygons; } }} min=1 max=1000 step=1/>
            </div>
            <div class="mutation-section">
                <div class="mutation-section-title">"Structure"</div>
                <ProbSlider state={state} label="Add polygon" field="add_polygon_prob"/>
                <ProbSlider state={state} label="Remove polygon" field="remove_polygon_prob"/>
                <ProbSlider state={state} label="Reorder polygon" field="reorder_polygon_prob"/>
            </div>
            <div class="mutation-section">
                <div class="mutation-section-title">"Movement"</div>
                <ProbSlider state={state} label="Offset polygon" field="offset_polygon_prob"/>
                <ProbSlider state={state} label="Move point" field="move_point_prob"/>
                <ProbSlider state={state} label="Micro adjust" field="micro_adjust_prob"/>
            </div>
            <div class="mutation-section">
                <div class="mutation-section-title">"Color"</div>
                <ProbSlider state={state} label="Change color" field="change_color_prob"/>
                <ProbSlider state={state} label="Lighten color" field="lighten_color_prob"/>
                <ProbSlider state={state} label="Darken color" field="darken_color_prob"/>
            </div>
            <div class="mutation-section">
                <div class="mutation-section-title">"Deltas"</div>
                <DeltaSlider state={state} label="Move point delta" field="move_point_max_delta" min=0.001 max=0.5 step=0.001/>
                <DeltaSlider state={state} label="Micro adjust delta" field="micro_adjust_delta" min=0.001 max=0.1 step=0.001/>
                <DeltaSlider state={state} label="New point distance" field="new_point_max_distance" min=0.001 max=0.2 step=0.001/>
                <DeltaSlider state={state} label="Offset magnitude" field="offset_polygon_magnitude" min=0.001 max=0.5 step=0.001/>
            </div>
            <div class="mutation-section">
                <div class="mutation-section-title">"Alpha"</div>
                <IntSlider state={state} label="Min alpha" get={|mp| mp.min_alpha as i64} set={|mp, v| { mp.min_alpha = v as u8; if mp.max_alpha < mp.min_alpha { mp.max_alpha = mp.min_alpha; } }} min=0 max=255 step=1/>
                <IntSlider state={state} label="Max alpha" get={|mp| mp.max_alpha as i64} set={|mp, v| { mp.max_alpha = v as u8; if mp.min_alpha > mp.max_alpha { mp.min_alpha = mp.max_alpha; } }} min=0 max=255 step=1/>
            </div>
            <div class="mutation-section mutation-section-actions">
                <button class="btn btn-secondary" on:click={on_reset}>"Reset to Defaults"</button>
            </div>
        </div>
    }
}

/// Probability slider with log scale and "1 in N" display.
#[component]
fn ProbSlider(
    state: RwSignal<ViewerState>,
    #[prop(into)] label: String,
    #[prop(into)] field: String,
) -> impl IntoView {
    let f1 = field.clone();
    let f2 = field.clone();
    let f3 = field.clone();

    let on_input = move |ev: web_sys::Event| {
        let target = ev.target().unwrap();
        let input: web_sys::HtmlInputElement = target.dyn_into().unwrap();
        let slider_val: f64 = input.value().parse().unwrap_or(0.0);
        let prob = slider_to_prob(slider_val);
        state.update(|s| set_prob_field(&mut s.mutation_params, &f1, prob));
        send_params(&state.get_untracked().mutation_params);
    };

    view! {
        <div class="mutation-row">
            <label class="mutation-label">{label}</label>
            <input
                type="range"
                class="mutation-slider"
                min="0"
                max="1000"
                step="1"
                prop:value={move || prob_to_slider(get_prob_field(&state.get().mutation_params, &f2)).to_string()}
                on:input={on_input}
            />
            <span class="mutation-value">{move || format_prob(get_prob_field(&state.get().mutation_params, &f3))}</span>
        </div>
    }
}

/// Linear delta/magnitude slider.
#[component]
fn DeltaSlider(
    state: RwSignal<ViewerState>,
    #[prop(into)] label: String,
    #[prop(into)] field: String,
    min: f64,
    max: f64,
    step: f64,
) -> impl IntoView {
    let f1 = field.clone();
    let f2 = field.clone();
    let f3 = field.clone();

    let on_input = move |ev: web_sys::Event| {
        let target = ev.target().unwrap();
        let input: web_sys::HtmlInputElement = target.dyn_into().unwrap();
        let val: f32 = input.value().parse().unwrap_or(0.0);
        state.update(|s| set_delta_field(&mut s.mutation_params, &f1, val));
        send_params(&state.get_untracked().mutation_params);
    };

    view! {
        <div class="mutation-row">
            <label class="mutation-label">{label}</label>
            <input
                type="range"
                class="mutation-slider"
                min={min.to_string()}
                max={max.to_string()}
                step={step.to_string()}
                prop:value={move || get_delta_field(&state.get().mutation_params, &f2).to_string()}
                on:input={on_input}
            />
            <span class="mutation-value">{move || format!("{:.3}", get_delta_field(&state.get().mutation_params, &f3))}</span>
        </div>
    }
}

/// Generic integer slider using getter/setter closures.
#[component]
fn IntSlider(
    state: RwSignal<ViewerState>,
    #[prop(into)] label: String,
    get: fn(&MutationParams) -> i64,
    set: fn(&mut MutationParams, i64),
    min: i64,
    max: i64,
    step: i64,
) -> impl IntoView {
    let on_input = move |ev: web_sys::Event| {
        let target = ev.target().unwrap();
        let input: web_sys::HtmlInputElement = target.dyn_into().unwrap();
        let val: i64 = input.value().parse().unwrap_or(0);
        state.update(|s| set(&mut s.mutation_params, val));
        send_params(&state.get_untracked().mutation_params);
    };

    view! {
        <div class="mutation-row">
            <label class="mutation-label">{label}</label>
            <input
                type="range"
                class="mutation-slider"
                min={min.to_string()}
                max={max.to_string()}
                step={step.to_string()}
                prop:value={move || get(&state.get().mutation_params).to_string()}
                on:input={on_input}
            />
            <span class="mutation-value">{move || get(&state.get().mutation_params).to_string()}</span>
        </div>
    }
}

// Field accessors — match on field name strings to get/set the right MutationParams field.

fn get_prob_field(mp: &MutationParams, field: &str) -> f32 {
    match field {
        "add_polygon_prob" => mp.add_polygon_prob,
        "remove_polygon_prob" => mp.remove_polygon_prob,
        "reorder_polygon_prob" => mp.reorder_polygon_prob,
        "offset_polygon_prob" => mp.offset_polygon_prob,
        "move_point_prob" => mp.move_point_prob,
        "remove_point_prob" => mp.remove_point_prob,
        "micro_adjust_prob" => mp.micro_adjust_prob,
        "change_color_prob" => mp.change_color_prob,
        "lighten_color_prob" => mp.lighten_color_prob,
        "darken_color_prob" => mp.darken_color_prob,
        _ => 0.0,
    }
}

fn set_prob_field(mp: &mut MutationParams, field: &str, val: f32) {
    match field {
        "add_polygon_prob" => mp.add_polygon_prob = val,
        "remove_polygon_prob" => mp.remove_polygon_prob = val,
        "reorder_polygon_prob" => mp.reorder_polygon_prob = val,
        "offset_polygon_prob" => mp.offset_polygon_prob = val,
        "move_point_prob" => mp.move_point_prob = val,
        "remove_point_prob" => mp.remove_point_prob = val,
        "micro_adjust_prob" => mp.micro_adjust_prob = val,
        "change_color_prob" => mp.change_color_prob = val,
        "lighten_color_prob" => mp.lighten_color_prob = val,
        "darken_color_prob" => mp.darken_color_prob = val,
        _ => {}
    }
}

fn get_delta_field(mp: &MutationParams, field: &str) -> f32 {
    match field {
        "move_point_max_delta" => mp.move_point_max_delta,
        "micro_adjust_delta" => mp.micro_adjust_delta,
        "new_point_max_distance" => mp.new_point_max_distance,
        "offset_polygon_magnitude" => mp.offset_polygon_magnitude,
        _ => 0.0,
    }
}

fn set_delta_field(mp: &mut MutationParams, field: &str, val: f32) {
    match field {
        "move_point_max_delta" => mp.move_point_max_delta = val,
        "micro_adjust_delta" => mp.micro_adjust_delta = val,
        "new_point_max_distance" => mp.new_point_max_distance = val,
        "offset_polygon_magnitude" => mp.offset_polygon_magnitude = val,
        _ => {}
    }
}
