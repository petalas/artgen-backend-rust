use leptos::prelude::*;
use wasm_bindgen::JsCast;

use crate::mutation_params::MutationParams;
use crate::ws::{send_ws_json, send_ws_loading, ViewerState};

/// Format a probability as a decimal string with adaptive precision.
fn format_prob(p: f32) -> String {
    if p <= 0.0 {
        "OFF".to_string()
    } else if p >= 0.01 {
        format!("{:.4}", p)
    } else {
        format!("{:.6}", p)
    }
}

/// Convert a probability to a log-scale slider value.
/// Maps prob range [0.00002, 0.5] to slider [0, 1000].
/// Left (0) = rarest, right (1000) = most frequent.
fn prob_to_slider(p: f32) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    let inv = (1.0 / p) as f64;
    let min_inv: f64 = 2.0;   // most frequent
    let max_inv: f64 = 50000.0; // rarest
    let log_range = max_inv.ln() - min_inv.ln();
    (1000.0 * (max_inv.ln() - inv.ln()) / log_range).clamp(0.0, 1000.0)
}

/// Convert a log-scale slider value back to a probability.
fn slider_to_prob(v: f64) -> f32 {
    let min_inv: f64 = 2.0;
    let max_inv: f64 = 50000.0;
    let log_range = max_inv.ln() - min_inv.ln();
    let log_inv = max_inv.ln() - (v / 1000.0) * log_range;
    (1.0 / log_inv.exp()) as f32
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
    let disabled = Signal::derive(move || {
        let s = state.get();
        !s.connected || s.engine_loading || !s.init_received
    });

    // Local params signal — initialized from state, synced via effect
    let params = RwSignal::new(state.get_untracked().mutation_params.clone());

    // Sync state → local params when server pushes updates (e.g. init)
    Effect::new(move |_| {
        let server_params = state.get().mutation_params.clone();
        if params.get_untracked() != server_params {
            params.set(server_params);
        }
    });

    // When local params change via slider, push to state + WS
    let on_change = Callback::new(move |_: ()| {
        let p = params.get_untracked();
        state.update(|s| s.mutation_params = p.clone());
        send_params(&p);
    });

    let on_reset = move |_| {
        send_ws_json(&serde_json::json!({ "type": "reset_params" }));
        let defaults = MutationParams::default();
        params.set(defaults.clone());
        state.update(|s| s.mutation_params = defaults);
    };

    let on_resolution_change = move |ev: web_sys::Event| {
        let target = ev.target().unwrap();
        let select: web_sys::HtmlSelectElement = target.dyn_into().unwrap();
        let val: u32 = select.value().parse().unwrap_or(0);
        send_ws_loading(state, &serde_json::json!({
            "type": "update_resolution",
            "resolution": val,
        }));
    };

    let resolution_presets: Vec<u32> = vec![0, 64, 128, 256, 384, 512, 768, 1024];

    view! {
        <div class="mutation-panel-body">
            <div class="mutation-section">
                <div class="mutation-section-title">"Resolution"</div>
                <div class="mutation-row">
                    <label class="mutation-label">"Internal resolution"</label>
                    <select
                        class="resolution-select"
                        disabled={move || disabled.get()}
                        on:change={on_resolution_change}
                        prop:value={move || state.get().target_resolution.to_string()}
                    >
                        {resolution_presets.into_iter().map(|r| {
                            let label = if r == 0 { "Auto (256-512)".to_string() } else { format!("{}px", r) };
                            let val = r.to_string();
                            view! {
                                <option value={val.clone()} selected={move || state.get().target_resolution == r}>{label}</option>
                            }
                        }).collect::<Vec<_>>()}
                    </select>
                </div>
            </div>
            <ParamsEditor params={params} on_change={on_change} disabled={disabled}/>
            <div class="mutation-section mutation-section-actions">
                <button class="btn btn-secondary" disabled={move || disabled.get()} on:click={on_reset}>"Reset to Defaults"</button>
            </div>
        </div>
    }
}

/// Reusable editor for all MutationParams fields.
/// Used by both the main mutation panel and the benchmark configuration page.
#[component]
pub fn ParamsEditor(
    params: RwSignal<MutationParams>,
    #[prop(optional)] on_change: Option<Callback<()>>,
    #[prop(optional)] disabled: Option<Signal<bool>>,
) -> impl IntoView {
    let notify = move || {
        if let Some(cb) = on_change {
            cb.run(());
        }
    };

    let is_disabled = move || disabled.map_or(false, |d| d.get());

    view! {
        <div class="mutation-section">
            <div class="mutation-section-title">"Mode"</div>
            <div class="mutation-row">
                <label class="bench-checkbox-label">
                    <input
                        type="checkbox"
                        disabled={move || is_disabled()}
                        prop:checked={move || params.get().single_mutation_mode}
                        on:change={move |_| {
                            params.update(|p| p.single_mutation_mode = !p.single_mutation_mode);
                            notify();
                        }}
                    />
                    "Single mutation per iteration"
                </label>
            </div>
            <div class="mutation-row">
                <label class="bench-checkbox-label">
                    <input
                        type="checkbox"
                        disabled={move || is_disabled()}
                        prop:checked={move || params.get().adaptive_mutation}
                        on:change={move |_| {
                            params.update(|p| p.adaptive_mutation = !p.adaptive_mutation);
                            notify();
                        }}
                    />
                    "Adaptive mutation scale (\u{03BB}>1 offspring only)"
                </label>
            </div>
            <div class="mutation-row">
                <label class="bench-checkbox-label">
                    <input
                        type="checkbox"
                        disabled={move || is_disabled()}
                        prop:checked={move || params.get().tile_culling}
                        on:change={move |_| {
                            params.update(|p| p.tile_culling = !p.tile_culling);
                            notify();
                        }}
                    />
                    "Tile culling (spatial polygon binning)"
                </label>
            </div>
            <div class="mutation-row">
                <label class="bench-checkbox-label">
                    <input
                        type="checkbox"
                        disabled={move || is_disabled()}
                        prop:checked={move || params.get().incremental_eval}
                        on:change={move |_| {
                            params.update(|p| p.incremental_eval = !p.incremental_eval);
                            notify();
                        }}
                    />
                    "Incremental eval (cached framebuffer)"
                </label>
            </div>
        </div>
        <div class="mutation-section">
            <div class="mutation-section-title">"Polygons"</div>
            <IntSlider params={params} on_change={on_change} label="Min polygons" get={|mp| mp.min_polygons as i64} set={|mp, v| { mp.min_polygons = v as u32; if mp.max_polygons < mp.min_polygons { mp.max_polygons = mp.min_polygons; } }} min=1 max=1000 step=1/>
            <IntSlider params={params} on_change={on_change} label="Max polygons" get={|mp| mp.max_polygons as i64} set={|mp, v| { mp.max_polygons = v as u32; if mp.min_polygons > mp.max_polygons { mp.min_polygons = mp.max_polygons; } }} min=1 max=1000 step=1/>
        </div>
        <div class="mutation-section">
            <div class="mutation-section-title">"Structure"</div>
            <ProbSlider params={params} on_change={on_change} label="Add polygon" field="add_polygon_prob"/>
            <ProbSlider params={params} on_change={on_change} label="Remove polygon" field="remove_polygon_prob"/>
            <ProbSlider params={params} on_change={on_change} label="Reorder polygon" field="reorder_polygon_prob"/>
            <ProbSlider params={params} on_change={on_change} label="Adjacent swap" field="adjacent_swap_prob"/>
            <ProbSlider params={params} on_change={on_change} label="Merge polygons" field="merge_polygon_prob"/>
            <ProbSlider params={params} on_change={on_change} label="Clone + jitter" field="clone_polygon_prob"/>
            <ProbSlider params={params} on_change={on_change} label="Swap colors" field="swap_colors_prob"/>
        </div>
        <div class="mutation-section">
            <div class="mutation-section-title">"Movement"</div>
            <ProbSlider params={params} on_change={on_change} label="Offset polygon" field="offset_polygon_prob"/>
            <ProbSlider params={params} on_change={on_change} label="Scale polygon" field="scale_polygon_prob"/>
            <ProbSlider params={params} on_change={on_change} label="Rotate polygon" field="rotate_polygon_prob"/>
            <ProbSlider params={params} on_change={on_change} label="Move point" field="move_point_prob"/>
            <ProbSlider params={params} on_change={on_change} label="Medium move" field="medium_move_prob"/>
            <ProbSlider params={params} on_change={on_change} label="Micro adjust" field="micro_adjust_prob"/>
        </div>
        <div class="mutation-section">
            <div class="mutation-section-title">"Color"</div>
            <ProbSlider params={params} on_change={on_change} label="Change color" field="change_color_prob"/>
            <ProbSlider params={params} on_change={on_change} label="Adjust brightness" field="adjust_brightness_prob"/>
            <ProbSlider params={params} on_change={on_change} label="Adjust saturation" field="adjust_saturation_prob"/>
        </div>
        <div class="mutation-section">
            <div class="mutation-section-title">"Deltas"</div>
            <DeltaSlider params={params} on_change={on_change} label="Move point delta" field="move_point_max_delta" min=0.03 max=0.3 step=0.005/>
            <DeltaSlider params={params} on_change={on_change} label="Micro adjust delta" field="micro_adjust_delta" min=0.0005 max=0.006 step=0.0005/>
            <DeltaSlider params={params} on_change={on_change} label="New point distance" field="new_point_max_distance" min=0.005 max=0.06 step=0.005/>
            <DeltaSlider params={params} on_change={on_change} label="Offset magnitude" field="offset_polygon_magnitude" min=0.005 max=0.16 step=0.005/>
            <DeltaSlider params={params} on_change={on_change} label="Medium move delta" field="medium_move_delta" min=0.005 max=0.06 step=0.005/>
            <DeltaSlider params={params} on_change={on_change} label="Merge centroid dist" field="merge_centroid_threshold" min=0.03 max=0.3 step=0.01/>
            <DeltaSlider params={params} on_change={on_change} label="Merge color dist" field="merge_color_threshold" min=0.04 max=0.4 step=0.01/>
        </div>
        <div class="mutation-section">
            <div class="mutation-section-title">"Crossover"</div>
            <ProbSlider params={params} on_change={on_change} label="Crossover prob" field="crossover_prob"/>
            <DeltaSlider params={params} on_change={on_change} label="Spatial weight" field="spatial_crossover_weight" min=0.0 max=1.0 step=0.05/>
            <IntSlider params={params} on_change={on_change} label="Tournament size" get={|mp| mp.tournament_size as i64} set={|mp, v| { mp.tournament_size = v as u32; }} min=1 max=16 step=1/>
            <Pow2Slider params={params} on_change={on_change} label="Chains" get={|mp| mp.chain_count} set={|mp, v| { mp.chain_count = v; }} min_exp=0 max_exp=10/>
            <Pow2Slider params={params} on_change={on_change} label="Lambda (\u{03BB})" get={|mp| mp.lambda} set={|mp, v| { mp.lambda = v; }} min_exp=0 max_exp=6/>
            <Pow2Slider params={params} on_change={on_change} label="Batch iters" get={|mp| mp.gpu_batch_iters} set={|mp, v| { mp.gpu_batch_iters = v; }} min_exp=0 max_exp=12/>
            <WgSelect params={params} on_change={on_change}/>
        </div>
        <div class="mutation-section">
            <div class="mutation-section-title">"Alpha"</div>
            <IntSlider params={params} on_change={on_change} label="Min alpha" get={|mp| mp.min_alpha as i64} set={|mp, v| { mp.min_alpha = v as u8; if mp.max_alpha < mp.min_alpha { mp.max_alpha = mp.min_alpha; } }} min=0 max=255 step=1/>
            <IntSlider params={params} on_change={on_change} label="Max alpha" get={|mp| mp.max_alpha as i64} set={|mp, v| { mp.max_alpha = v as u8; if mp.min_alpha > mp.max_alpha { mp.min_alpha = mp.max_alpha; } }} min=0 max=255 step=1/>
        </div>
    }
}

/// Rasterize workgroup size selector.
#[component]
fn WgSelect(
    params: RwSignal<MutationParams>,
    on_change: Option<Callback<()>>,
) -> impl IntoView {
    let notify = move || {
        if let Some(cb) = on_change {
            cb.run(());
        }
    };

    view! {
        <div class="mutation-row">
            <label class="mutation-label">"Rasterize WG"</label>
            <select
                class="resolution-select"
                prop:value={move || {
                    let wg = params.get().rasterize_wg;
                    format!("{}x{}", wg[0], wg[1])
                }}
                on:change={move |ev: web_sys::Event| {
                    let target = ev.target().unwrap();
                    let select: web_sys::HtmlSelectElement = target.dyn_into().unwrap();
                    let val = select.value();
                    let parts: Vec<u32> = val.split('x').filter_map(|s| s.parse().ok()).collect();
                    if parts.len() == 2 {
                        params.update(|p| p.rasterize_wg = [parts[0], parts[1]]);
                        notify();
                    }
                }}
            >
                <option value="32x16">"32x16 (512 threads)"</option>
                <option value="16x16">"16x16 (256 threads)"</option>
                <option value="32x8">"32x8 (256 threads)"</option>
                <option value="16x8">"16x8 (128 threads)"</option>
                <option value="8x8">"8x8 (64 threads)"</option>
            </select>
        </div>
    }
}

/// Probability slider with log scale and decimal display.
#[component]
fn ProbSlider(
    params: RwSignal<MutationParams>,
    on_change: Option<Callback<()>>,
    #[prop(into)] label: String,
    #[prop(into)] field: String,
) -> impl IntoView {
    let f1 = field.clone();
    let f2 = field.clone();
    let f3 = field.clone();

    let notify = move || {
        if let Some(cb) = on_change {
            cb.run(());
        }
    };

    let on_input = move |ev: web_sys::Event| {
        let target = ev.target().unwrap();
        let input: web_sys::HtmlInputElement = target.dyn_into().unwrap();
        let slider_val: f64 = input.value().parse().unwrap_or(0.0);
        let prob = slider_to_prob(slider_val);
        params.update(|p| set_prob_field(p, &f1, prob));
        notify();
    };

    view! {
        <div class="mutation-row">
            <label class="mutation-label">{label}</label>
            <input
                type="range"
                class="mutation-slider"
                min="0"
                max="1000"
                step="5"
                prop:value={move || prob_to_slider(get_prob_field(&params.get(), &f2)).to_string()}
                on:input={on_input}
            />
            <span class="mutation-value">{move || format_prob(get_prob_field(&params.get(), &f3))}</span>
        </div>
    }
}

/// Linear delta/magnitude slider.
#[component]
fn DeltaSlider(
    params: RwSignal<MutationParams>,
    on_change: Option<Callback<()>>,
    #[prop(into)] label: String,
    #[prop(into)] field: String,
    min: f64,
    max: f64,
    step: f64,
) -> impl IntoView {
    let f1 = field.clone();
    let f2 = field.clone();
    let f3 = field.clone();

    let notify = move || {
        if let Some(cb) = on_change {
            cb.run(());
        }
    };

    let on_input = move |ev: web_sys::Event| {
        let target = ev.target().unwrap();
        let input: web_sys::HtmlInputElement = target.dyn_into().unwrap();
        let val: f32 = input.value().parse().unwrap_or(0.0);
        params.update(|p| set_delta_field(p, &f1, val));
        notify();
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
                prop:value={move || get_delta_field(&params.get(), &f2).to_string()}
                on:input={on_input}
            />
            <span class="mutation-value">{move || format!("{:.3}", get_delta_field(&params.get(), &f3))}</span>
        </div>
    }
}

/// Generic integer slider using getter/setter closures.
#[component]
fn IntSlider(
    params: RwSignal<MutationParams>,
    on_change: Option<Callback<()>>,
    #[prop(into)] label: String,
    get: fn(&MutationParams) -> i64,
    set: fn(&mut MutationParams, i64),
    min: i64,
    max: i64,
    step: i64,
) -> impl IntoView {
    let notify = move || {
        if let Some(cb) = on_change {
            cb.run(());
        }
    };

    let on_input = move |ev: web_sys::Event| {
        let target = ev.target().unwrap();
        let input: web_sys::HtmlInputElement = target.dyn_into().unwrap();
        let val: i64 = input.value().parse().unwrap_or(0);
        params.update(|p| set(p, val));
        notify();
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
                prop:value={move || get(&params.get()).to_string()}
                on:input={on_input}
            />
            <span class="mutation-value">{move || get(&params.get()).to_string()}</span>
        </div>
    }
}

/// Power-of-2 slider: exponent maps to 2^exp.
#[component]
fn Pow2Slider(
    params: RwSignal<MutationParams>,
    on_change: Option<Callback<()>>,
    #[prop(into)] label: String,
    get: fn(&MutationParams) -> u32,
    set: fn(&mut MutationParams, u32),
    min_exp: u32,
    max_exp: u32,
) -> impl IntoView {
    let notify = move || {
        if let Some(cb) = on_change {
            cb.run(());
        }
    };

    let on_input = move |ev: web_sys::Event| {
        let target = ev.target().unwrap();
        let input: web_sys::HtmlInputElement = target.dyn_into().unwrap();
        let exp: u32 = input.value().parse().unwrap_or(min_exp);
        let val = 1u32 << exp;
        params.update(|p| set(p, val));
        notify();
    };

    view! {
        <div class="mutation-row">
            <label class="mutation-label">{label}</label>
            <input
                type="range"
                class="mutation-slider"
                min={min_exp.to_string()}
                max={max_exp.to_string()}
                step="1"
                prop:value={move || {
                    let v = get(&params.get()).max(1);
                    v.ilog2().clamp(min_exp, max_exp).to_string()
                }}
                on:input={on_input}
            />
            <span class="mutation-value">{move || get(&params.get()).to_string()}</span>
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
        "adjust_brightness_prob" => mp.adjust_brightness_prob,
        "adjust_saturation_prob" => mp.adjust_saturation_prob,
        "scale_polygon_prob" => mp.scale_polygon_prob,
        "rotate_polygon_prob" => mp.rotate_polygon_prob,
        "adjacent_swap_prob" => mp.adjacent_swap_prob,
        "merge_polygon_prob" => mp.merge_polygon_prob,
        "clone_polygon_prob" => mp.clone_polygon_prob,
        "medium_move_prob" => mp.medium_move_prob,
        "swap_colors_prob" => mp.swap_colors_prob,
        "crossover_prob" => mp.crossover_prob,
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
        "adjust_brightness_prob" => mp.adjust_brightness_prob = val,
        "adjust_saturation_prob" => mp.adjust_saturation_prob = val,
        "scale_polygon_prob" => mp.scale_polygon_prob = val,
        "rotate_polygon_prob" => mp.rotate_polygon_prob = val,
        "adjacent_swap_prob" => mp.adjacent_swap_prob = val,
        "merge_polygon_prob" => mp.merge_polygon_prob = val,
        "clone_polygon_prob" => mp.clone_polygon_prob = val,
        "medium_move_prob" => mp.medium_move_prob = val,
        "swap_colors_prob" => mp.swap_colors_prob = val,
        "crossover_prob" => mp.crossover_prob = val,
        _ => {}
    }
}

fn get_delta_field(mp: &MutationParams, field: &str) -> f32 {
    match field {
        "move_point_max_delta" => mp.move_point_max_delta,
        "micro_adjust_delta" => mp.micro_adjust_delta,
        "new_point_max_distance" => mp.new_point_max_distance,
        "offset_polygon_magnitude" => mp.offset_polygon_magnitude,
        "medium_move_delta" => mp.medium_move_delta,
        "merge_centroid_threshold" => mp.merge_centroid_threshold,
        "merge_color_threshold" => mp.merge_color_threshold,
        "spatial_crossover_weight" => mp.spatial_crossover_weight,
        _ => 0.0,
    }
}

fn set_delta_field(mp: &mut MutationParams, field: &str, val: f32) {
    match field {
        "move_point_max_delta" => mp.move_point_max_delta = val,
        "micro_adjust_delta" => mp.micro_adjust_delta = val,
        "new_point_max_distance" => mp.new_point_max_distance = val,
        "offset_polygon_magnitude" => mp.offset_polygon_magnitude = val,
        "medium_move_delta" => mp.medium_move_delta = val,
        "merge_centroid_threshold" => mp.merge_centroid_threshold = val,
        "merge_color_threshold" => mp.merge_color_threshold = val,
        "spatial_crossover_weight" => mp.spatial_crossover_weight = val,
        _ => {}
    }
}
