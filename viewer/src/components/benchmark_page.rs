use leptos::prelude::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use std::collections::HashSet;

use crate::benchmark::{BenchmarkExport, BenchmarkRequest, BenchmarkResult, BenchmarkSnapshot};
use crate::components::benchmark_chart::{color_for_index, BenchmarkChart};
use crate::components::controls::download_blob;
use crate::mutation_params::MutationParams;
use crate::ws::{send_ws_json, ViewerState};

#[component]
pub fn BenchmarkPage(state: RwSignal<ViewerState>) -> impl IntoView {
    view! {
        <div class="benchmark-page">
            <SnapshotsSection state={state}/>
            <ConfigureSection state={state}/>
            <QueueSection state={state}/>
            <ResultsSection state={state}/>
        </div>
    }
}

// ── Snapshots ──────────────────────────────────────────────

#[component]
fn SnapshotsSection(state: RwSignal<ViewerState>) -> impl IntoView {
    let snap_name = RwSignal::new(String::new());

    let save_snapshot = move |_| {
        let s = state.get();
        let Some(drawing_json) = s.drawing_json.as_ref() else { return };
        let name = snap_name.get();
        let name = if name.trim().is_empty() {
            format!("snap-{:.2}%", s.fitness)
        } else {
            name.trim().to_string()
        };
        let msg = serde_json::json!({
            "type": "save_snapshot",
            "name": name,
            "drawingJson": drawing_json,
            "fitness": s.fitness,
            "polygonCount": s.polygons,
        });
        send_ws_json(&msg);
        snap_name.set(String::new());
    };

    let delete_snapshot = move |id: String| {
        let msg = serde_json::json!({
            "type": "delete_snapshot",
            "snapshotId": id,
        });
        send_ws_json(&msg);
    };

    view! {
        <div class="bench-section">
            <h3 class="bench-section-title">"Snapshots"</h3>
            <div class="bench-snapshot-form">
                <input
                    type="text"
                    class="project-name-input"
                    placeholder="Snapshot name (optional)"
                    prop:value={move || snap_name.get()}
                    on:input={move |ev| snap_name.set(event_target_value(&ev))}
                />
                <button
                    class="btn btn-primary"
                    on:click={save_snapshot}
                    disabled={move || state.get().drawing_json.is_none()}
                >
                    "Save Current State"
                </button>
            </div>
            <div class="bench-snapshot-list">
                {move || {
                    let snaps = state.get().benchmark_snapshots.clone();
                    if snaps.is_empty() {
                        return view! {
                            <div class="projects-empty">"No snapshots saved. Save one to start benchmarking."</div>
                        }.into_any();
                    }
                    view! {
                        <div>
                            {snaps.into_iter().map(|snap| {
                                let id = snap.id.clone();
                                let del = move |_| delete_snapshot(id.clone());
                                let created_str = if snap.created_at.is_empty() {
                                    String::new()
                                } else {
                                    format!(" | {}", &snap.created_at[..16.min(snap.created_at.len())])
                                };
                                view! {
                                    <div class="bench-snapshot-item">
                                        <div class="bench-snapshot-info">
                                            <span class="bench-snapshot-name">{snap.name.clone()}</span>
                                            <span class="bench-snapshot-meta">
                                                {format!("{:.2}% | {} polygons{}", snap.fitness, snap.polygon_count, created_str)}
                                            </span>
                                        </div>
                                        <button class="btn-icon btn-icon-danger" on:click={del} title="Delete">
                                            "\u{2715}"
                                        </button>
                                    </div>
                                }
                            }).collect::<Vec<_>>()}
                        </div>
                    }.into_any()
                }}
            </div>
        </div>
    }
}

// ── Configure Run ──────────────────────────────────────────

// Chains: exponent 0..9 → 1,2,4,...,512
fn chains_from_exp(exp: u32) -> u32 { 1u32 << exp }

// Lambda: exponent 0..6 → 1,2,4,8,16,32,64
fn lambda_from_exp(exp: u32) -> u32 { 1u32 << exp }

// Workgroup size options: index -> [wg_x, wg_y]
const WG_OPTIONS: &[[u32; 2]] = &[[32, 16], [16, 16], [32, 8], [16, 8], [8, 8]];

fn wg_label(wg: &[u32; 2]) -> String {
    let threads = wg[0] * wg[1];
    format!("{}x{} ({})", wg[0], wg[1], threads)
}

fn build_benchmark_params(
    state: &ViewerState,
    chain_count: u32,
    lambda: u32,
    single_mutation_mode: bool,
    adaptive_mutation: bool,
    tile_culling: bool,
    rasterize_wg: [u32; 2],
    gpu_batch_iters: u32,
) -> crate::mutation_params::MutationParams {
    let mut params = state.mutation_params.clone();
    params.chain_count = chain_count;
    params.lambda = lambda;
    params.single_mutation_mode = single_mutation_mode;
    params.adaptive_mutation = adaptive_mutation;
    params.tile_culling = tile_culling;
    params.rasterize_wg = rasterize_wg;
    params.gpu_batch_iters = gpu_batch_iters;
    params
}

fn auto_label(chain_count: u32, lambda: u32, single_mutation_mode: bool, adaptive_mutation: bool, tile_culling: bool, rasterize_wg: [u32; 2], gpu_batch_iters: u32, resolution: u32) -> String {
    let mode = if single_mutation_mode { "single" } else { "multi" };
    let lambda_str = if lambda > 1 { format!("-{}\u{03BB}", lambda) } else { String::new() };
    let adaptive_str = if adaptive_mutation { "-adaptive" } else { "" };
    let tile_str = if tile_culling { "-tiled" } else { "" };
    let defaults = MutationParams::default();
    let wg_str = format!("-wg{}x{}", rasterize_wg[0], rasterize_wg[1]);
    let batch_str = if gpu_batch_iters != defaults.gpu_batch_iters {
        format!("-b{}", gpu_batch_iters)
    } else {
        String::new()
    };
    let res_str = format!("-{}px", resolution);
    format!("{}c{}-{}{}{}{}{}{}", chain_count, lambda_str, mode, adaptive_str, tile_str, wg_str, batch_str, res_str)
}

fn deduplicate_label(base: &str, state: &ViewerState) -> String {
    let existing: Vec<&str> = state.benchmark_results.iter().map(|r| r.label.as_str())
        .chain(state.benchmark_queue.iter().map(|r| r.label.as_str()))
        .collect();
    if !existing.contains(&base) {
        return base.to_string();
    }
    let mut n = 2u32;
    loop {
        let candidate = format!("{} #{}", base, n);
        if !existing.contains(&candidate.as_str()) {
            return candidate;
        }
        n += 1;
    }
}

#[component]
fn ConfigureSection(state: RwSignal<ViewerState>) -> impl IntoView {
    let selected_snap = RwSignal::new(0usize);
    let duration_secs = RwSignal::new(60u32);
    let label = RwSignal::new(String::new());
    let defaults = MutationParams::default();
    let chain_exp = RwSignal::new(defaults.chain_count.max(1).ilog2());
    let lambda_exp = RwSignal::new(defaults.lambda.max(1).ilog2());
    let single_mutation = RwSignal::new(defaults.single_mutation_mode);
    let adaptive_mutation = RwSignal::new(defaults.adaptive_mutation);
    let tile_culling = RwSignal::new(defaults.tile_culling);
    let default_wg_idx = WG_OPTIONS.iter().position(|w| *w == defaults.rasterize_wg).unwrap_or(0);
    let rasterize_wg_idx = RwSignal::new(default_wg_idx);
    let batch_iters = RwSignal::new(defaults.gpu_batch_iters);
    let resolution = RwSignal::new(state.get_untracked().target_resolution);

    let build_request = move || -> Option<(BenchmarkRequest, String)> {
        let s = state.get();
        let idx = selected_snap.get();
        let snap = s.benchmark_snapshots.get(idx)?;
        let cc = chains_from_exp(chain_exp.get());
        let lam = lambda_from_exp(lambda_exp.get());
        let sm = single_mutation.get();
        let am = adaptive_mutation.get();
        let tc = tile_culling.get();
        let wg = WG_OPTIONS[rasterize_wg_idx.get()];
        let bi = batch_iters.get();
        let res = resolution.get();
        let params = build_benchmark_params(&s, cc, lam, sm, am, tc, wg, bi);
        let lbl = label.get();
        let base = if lbl.trim().is_empty() { auto_label(cc, lam, sm, am, tc, wg, bi, res) } else { lbl.trim().to_string() };
        let lbl = deduplicate_label(&base, &s);
        let req = BenchmarkRequest {
            drawing_json: snap.drawing_json.clone(),
            params,
            duration_secs: duration_secs.get(),
            label: lbl,
            resolution: res,
            snapshot_id: snap.id.clone(),
        };
        Some((req, snap.id.clone()))
    };

    let add_to_queue = move |_| {
        if let Some((req, _)) = build_request() {
            state.update(|s| s.benchmark_queue.push(req));
            label.set(String::new());
        }
    };

    let run_now = move |_| {
        if let Some((req, _)) = build_request() {
            let msg = serde_json::json!({
                "type": "start_benchmark",
                "drawingJson": req.drawing_json,
                "params": req.params,
                "durationSecs": req.duration_secs,
                "label": req.label,
                "resolution": req.resolution,
                "snapshotId": req.snapshot_id,
            });
            state.update(|s| s.benchmark_initializing = true);
            send_ws_json(&msg);
            label.set(String::new());
        }
    };

    let has_snapshots = move || !state.get().benchmark_snapshots.is_empty();
    let is_active = move || { let s = state.get(); s.benchmark_active || s.benchmark_initializing };

    view! {
        <div class="bench-section">
            <h3 class="bench-section-title">"Configure Run"</h3>
            <div class="bench-config-grid">
                <div class="bench-config-row">
                    <label class="bench-config-label">"Snapshot"</label>
                    <select
                        class="bench-select"
                        prop:value={move || selected_snap.get().to_string()}
                        on:change={move |ev| {
                            if let Ok(v) = event_target_value(&ev).parse::<usize>() {
                                selected_snap.set(v);
                            }
                        }}
                        disabled={move || !has_snapshots()}
                    >
                        {move || {
                            state.get().benchmark_snapshots.iter().enumerate().map(|(i, snap)| {
                                view! {
                                    <option value={i.to_string()}>
                                        {format!("{} ({:.2}%)", snap.name, snap.fitness)}
                                    </option>
                                }
                            }).collect::<Vec<_>>()
                        }}
                    </select>
                </div>
                <div class="bench-config-row">
                    <label class="bench-config-label">"Duration"</label>
                    <div class="bench-radio-group">
                        {[30u32, 60, 120, 300, 600].into_iter().map(|d| {
                            let label_text = if d < 60 {
                                format!("{} sec", d)
                            } else {
                                let mins = d / 60;
                                if mins == 1 { "1 min".to_string() } else { format!("{} min", mins) }
                            };
                            view! {
                                <label class="bench-radio-label">
                                    <input
                                        type="radio"
                                        name="duration"
                                        value={d.to_string()}
                                        checked={move || duration_secs.get() == d}
                                        on:change={move |_| duration_secs.set(d)}
                                    />
                                    {label_text}
                                </label>
                            }
                        }).collect::<Vec<_>>()}
                    </div>
                </div>
                <div class="bench-config-row">
                    <label class="bench-config-label">"Label"</label>
                    <input
                        type="text"
                        class="project-name-input"
                        placeholder="Auto-generated from params"
                        prop:value={move || label.get()}
                        on:input={move |ev| label.set(event_target_value(&ev))}
                    />
                </div>
                <div class="bench-config-row">
                    <label class="bench-config-label">"Chains"</label>
                    <input
                        type="range"
                        class="mutation-slider"
                        min="0" max="10" step="1"
                        prop:value={move || chain_exp.get().to_string()}
                        on:input={move |ev| {
                            if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                chain_exp.set(v);
                            }
                        }}
                    />
                    <span class="mutation-value">{move || chains_from_exp(chain_exp.get()).to_string()}</span>
                </div>
                <div class="bench-config-row">
                    <label class="bench-config-label">"Lambda"</label>
                    <input
                        type="range"
                        class="mutation-slider"
                        min="0" max="6" step="1"
                        prop:value={move || lambda_exp.get().to_string()}
                        on:input={move |ev| {
                            if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                lambda_exp.set(v);
                            }
                        }}
                    />
                    <span class="mutation-value">{move || {
                        let lam = lambda_from_exp(lambda_exp.get());
                        if lam == 1 { "1 (1+1)".to_string() } else { format!("{} (1+\u{03BB})", lam) }
                    }}</span>
                </div>
                <div class="bench-config-row">
                    <label class="bench-config-label">"Mutation"</label>
                    <label class="bench-checkbox-label">
                        <input
                            type="checkbox"
                            prop:checked={move || single_mutation.get()}
                            on:change={move |_| {
                                single_mutation.set(!single_mutation.get_untracked());
                            }}
                        />
                        "Single mutation per iteration"
                    </label>
                </div>
                <div class="bench-config-row">
                    <label class="bench-config-label">"Adaptive"</label>
                    <label class="bench-checkbox-label">
                        <input
                            type="checkbox"
                            prop:checked={move || adaptive_mutation.get()}
                            on:change={move |_| {
                                adaptive_mutation.set(!adaptive_mutation.get_untracked());
                            }}
                        />
                        "Adaptive mutation scale (\u{03BB}>1 offspring only)"
                    </label>
                </div>
                <div class="bench-config-row">
                    <label class="bench-config-label">"Tile cull"</label>
                    <label class="bench-checkbox-label">
                        <input
                            type="checkbox"
                            prop:checked={move || tile_culling.get()}
                            on:change={move |_| {
                                tile_culling.set(!tile_culling.get_untracked());
                            }}
                        />
                        "Spatial tile culling (polygon binning)"
                    </label>
                </div>
                <div class="bench-config-row">
                    <label class="bench-config-label">"Workgroup"</label>
                    <select
                        class="bench-select"
                        prop:value={move || rasterize_wg_idx.get().to_string()}
                        on:change={move |ev| {
                            if let Ok(v) = event_target_value(&ev).parse::<usize>() {
                                rasterize_wg_idx.set(v);
                            }
                        }}
                    >
                        {WG_OPTIONS.iter().enumerate().map(|(i, wg)| {
                            let label_text = wg_label(wg);
                            let val = i.to_string();
                            view! {
                                <option value={val}>{label_text}</option>
                            }
                        }).collect::<Vec<_>>()}
                    </select>
                    <span class="mutation-value">{move || {
                        let wg = WG_OPTIONS[rasterize_wg_idx.get()];
                        format!("{} threads", wg[0] * wg[1])
                    }}</span>
                </div>
                <div class="bench-config-row">
                    <label class="bench-config-label">"Batch iters"</label>
                    <input
                        type="range"
                        class="mutation-slider"
                        min="0" max="12" step="1"
                        prop:value={move || batch_iters.get().max(1).ilog2().to_string()}
                        on:input={move |ev| {
                            if let Ok(exp) = event_target_value(&ev).parse::<u32>() {
                                batch_iters.set(1u32 << exp);
                            }
                        }}
                    />
                    <span class="mutation-value">{move || batch_iters.get().to_string()}</span>
                </div>
                <div class="bench-config-row">
                    <label class="bench-config-label">"Resolution"</label>
                    <select
                        class="resolution-select"
                        prop:value={move || resolution.get().to_string()}
                        on:change={move |ev| {
                            if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                resolution.set(v);
                            }
                        }}
                    >
                        {[64u32, 128, 256, 384, 512, 768, 1024].into_iter().map(|r| {
                            let label_text = format!("{}px", r);
                            let val = r.to_string();
                            view! {
                                <option value={val.clone()} selected={move || resolution.get() == r}>{label_text}</option>
                            }
                        }).collect::<Vec<_>>()}
                    </select>
                </div>
            </div>
            <div class="bench-config-actions">
                <button
                    class="btn btn-secondary"
                    on:click={add_to_queue}
                    disabled={move || !has_snapshots() || is_active()}
                >
                    "Add to Queue"
                </button>
                <button
                    class="btn btn-primary"
                    on:click={run_now}
                    disabled={move || !has_snapshots() || is_active()}
                >
                    "Run Now"
                </button>
                <button
                    class="btn btn-primary"
                    style="background: #6366f1;"
                    on:click={move |_| {
                        state.update(|s| s.benchmark_initializing = true);
                        send_ws_json(&serde_json::json!({
                            "type": "run_standard_benchmark",
                        }));
                    }}
                    disabled={move || is_active() || state.get().active_project.is_none()}
                    title="4 chains, 64 lambda, single mutation, adaptive, 33s from random start"
                >
                    {move || {
                        match &state.get().active_project {
                            Some(name) => format!("Std Bench: {} (33s)", name),
                            None => "Std Bench (no project)".to_string(),
                        }
                    }}
                </button>
            </div>
        </div>
    }
}

// ── Queue & Progress ───────────────────────────────────────

#[component]
fn QueueSection(state: RwSignal<ViewerState>) -> impl IntoView {
    let remove_from_queue = move |idx: usize| {
        state.update(|s| { s.benchmark_queue.remove(idx); });
    };

    let run_all = move |_| {
        let s = state.get();
        if s.benchmark_queue.is_empty() || s.benchmark_active {
            return;
        }
        // Take first from queue and send it
        let req = state.with_untracked(|s| s.benchmark_queue.first().cloned());
        if let Some(req) = req {
            state.update(|s| {
                s.benchmark_queue.remove(0);
                s.benchmark_initializing = true;
            });
            let msg = serde_json::json!({
                "type": "start_benchmark",
                "drawingJson": req.drawing_json,
                "params": req.params,
                "durationSecs": req.duration_secs,
                "label": req.label,
                "resolution": req.resolution,
                "snapshotId": req.snapshot_id,
            });
            send_ws_json(&msg);
        }
    };

    let has_queue = move || !state.get().benchmark_queue.is_empty();
    let is_active = move || { let s = state.get(); s.benchmark_active || s.benchmark_initializing };

    view! {
        <div class="bench-section">
            <h3 class="bench-section-title">"Queue & Progress"</h3>

            // Initializing spinner (shown while pipeline is being prepared)
            {move || {
                let s = state.get();
                s.benchmark_initializing.then(|| view! {
                    <div class="bench-progress">
                        <div class="bench-initializing">
                            <div class="spinner"></div>
                            <span>"Initializing pipeline\u{2026}"</span>
                        </div>
                    </div>
                })
            }}

            // Progress display
            {move || {
                let s = state.get();
                if let Some(prog) = &s.benchmark_progress {
                    let pct = if prog.duration_secs > 0 {
                        (prog.elapsed_secs / prog.duration_secs as f32 * 100.0).min(100.0)
                    } else {
                        0.0
                    };
                    view! {
                        <div class="bench-progress">
                            <div class="bench-progress-header">
                                <span class="bench-progress-label">
                                    {format!("Running: {}", prog.label)}
                                </span>
                                <span class="bench-progress-stats">
                                    {format!("{:.1}% | {} improvements", prog.best_fitness, prog.improvements)}
                                </span>
                            </div>
                            <div class="bench-progress-bar-bg">
                                <div
                                    class="bench-progress-bar-fill"
                                    style={format!("width: {}%", pct)}
                                />
                            </div>
                            <div class="bench-progress-time">
                                {format!(
                                    "{:.0}s / {}s",
                                    prog.elapsed_secs,
                                    prog.duration_secs,
                                )}
                            </div>
                        </div>
                    }.into_any()
                } else {
                    view! { <div></div> }.into_any()
                }
            }}

            // Queued items
            {move || {
                let queue = state.get().benchmark_queue.clone();
                if queue.is_empty() && !is_active() {
                    return view! {
                        <div class="bench-queue-empty">"No benchmarks queued."</div>
                    }.into_any();
                }
                view! {
                    <div class="bench-queue-list">
                        {queue.into_iter().enumerate().map(|(i, req)| {
                            let rm = move |_| remove_from_queue(i);
                            view! {
                                <div class="bench-queue-item">
                                    <span class="bench-queue-label">{req.label.clone()}</span>
                                    <span class="bench-queue-meta">
                                        {format!("{}s | {}px {}c {}\u{03BB} b{}{}", req.duration_secs, req.resolution, req.params.chain_count, req.params.lambda, req.params.gpu_batch_iters,
                                            if req.params.rasterize_wg != [16, 16] { format!(" wg{}x{}", req.params.rasterize_wg[0], req.params.rasterize_wg[1]) } else { String::new() }
                                        )}
                                    </span>
                                    <button class="btn-icon btn-icon-danger" on:click={rm} title="Remove">
                                        "\u{2715}"
                                    </button>
                                </div>
                            }
                        }).collect::<Vec<_>>()}
                    </div>
                }.into_any()
            }}

            <div class="bench-queue-actions">
                <button
                    class="btn btn-success"
                    on:click={run_all}
                    disabled={move || !has_queue() || is_active()}
                >
                    "Run All"
                </button>
            </div>
        </div>
    }
}

// ── Assign unassociated results ────────────────────────────

#[component]
fn UnassociatedHeader(
    snapshots: Vec<BenchmarkSnapshot>,
    result_ids: Vec<String>,
) -> impl IntoView {
    let assign_target = RwSignal::new(String::new());

    // Initialize with first snapshot id
    if let Some(first) = snapshots.first() {
        assign_target.set(first.id.clone());
    }

    let snaps = snapshots.clone();
    let ids = result_ids.clone();
    let assign = move |_| {
        let target = assign_target.get();
        if target.is_empty() {
            return;
        }
        let msg = serde_json::json!({
            "type": "assign_results_to_snapshot",
            "snapshotId": target,
            "resultIds": ids,
        });
        send_ws_json(&msg);
    };

    view! {
        <span class="bench-unassociated-header">
            <span>"Unassociated results"</span>
            <select
                class="bench-assign-select"
                prop:value={move || assign_target.get()}
                on:change={move |ev| assign_target.set(event_target_value(&ev))}
            >
                {snaps.iter().map(|s| {
                    let id = s.id.clone();
                    let label = format!("{} ({:.2}%)", s.name, s.fitness);
                    view! {
                        <option value={id}>{label}</option>
                    }
                }).collect::<Vec<_>>()}
            </select>
            <button class="btn btn-secondary btn-sm" on:click={assign}>"Assign"</button>
        </span>
    }
}

// ── Results ────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum SortCol {
    None,
    Label,
    Resolution,
    Chains,
    Lambda,
    Batch,
    Duration,
    Start,
    Final,
    Improvements,
    ImprovPerSec,
    EvalsPerSec,
}

fn result_evals_per_sec(r: &BenchmarkResult) -> f64 {
    let secs = if r.actual_duration_secs > 0.0 { r.actual_duration_secs as f64 } else { r.duration_secs as f64 };
    if secs > 0.0 { r.total_evals as f64 / secs } else { 0.0 }
}

fn result_actual_duration(r: &BenchmarkResult) -> f32 {
    if r.actual_duration_secs > 0.0 { r.actual_duration_secs } else { r.duration_secs as f32 }
}

fn sort_results(items: &mut [(usize, BenchmarkResult)], col: SortCol, ascending: bool) {
    items.sort_by(|(_, a), (_, b)| {
        let ord = match col {
            SortCol::Label => a.label.to_lowercase().cmp(&b.label.to_lowercase()),
            SortCol::Resolution => a.resolution.cmp(&b.resolution),
            SortCol::Chains => a.chain_count.cmp(&b.chain_count),
            SortCol::Lambda => a.lambda.cmp(&b.lambda),
            SortCol::Batch => a.gpu_batch_iters.cmp(&b.gpu_batch_iters),
            SortCol::Duration => result_actual_duration(a).partial_cmp(&result_actual_duration(b)).unwrap_or(std::cmp::Ordering::Equal),
            SortCol::Start => a.start_fitness.partial_cmp(&b.start_fitness).unwrap_or(std::cmp::Ordering::Equal),
            SortCol::Final => a.final_fitness.partial_cmp(&b.final_fitness).unwrap_or(std::cmp::Ordering::Equal),
            SortCol::Improvements => a.total_improvements.cmp(&b.total_improvements),
            SortCol::ImprovPerSec => a.improvements_per_sec.partial_cmp(&b.improvements_per_sec).unwrap_or(std::cmp::Ordering::Equal),
            SortCol::EvalsPerSec => result_evals_per_sec(a).partial_cmp(&result_evals_per_sec(b)).unwrap_or(std::cmp::Ordering::Equal),
            SortCol::None => std::cmp::Ordering::Equal,
        };
        if ascending { ord } else { ord.reverse() }
    });
}

#[component]
fn ResultsSection(state: RwSignal<ViewerState>) -> impl IntoView {
    let clear_results = move |_| {
        let msg = serde_json::json!({ "type": "clear_benchmarks" });
        send_ws_json(&msg);
    };

    let delete_result = move |id: String| {
        let msg = serde_json::json!({
            "type": "delete_benchmark_result",
            "resultId": id,
        });
        send_ws_json(&msg);
    };

    let results_signal = Signal::derive(move || state.get().benchmark_results.clone());
    let hidden_ids: RwSignal<HashSet<String>> = RwSignal::new(HashSet::new());
    let hidden_signal = Signal::derive(move || hidden_ids.get());

    let toggle_visibility = move |id: String| {
        hidden_ids.update(|set| {
            if !set.remove(&id) {
                set.insert(id);
            }
        });
    };

    let sort_col: RwSignal<SortCol> = RwSignal::new(SortCol::None);
    let sort_asc: RwSignal<bool> = RwSignal::new(true);

    let click_sort = move |col: SortCol| {
        if sort_col.get_untracked() == col {
            sort_asc.set(!sort_asc.get_untracked());
        } else {
            sort_col.set(col);
            // Default: descending for numeric, ascending for label
            sort_asc.set(col == SortCol::Label);
        }
    };

    // Derive display order (original indices in sorted order) for chart legend
    let display_order_signal = Signal::derive(move || {
        let results = state.get().benchmark_results;
        let col = sort_col.get();
        let asc = sort_asc.get();
        let mut indexed: Vec<(usize, BenchmarkResult)> = results.into_iter().enumerate().collect();
        if col != SortCol::None {
            sort_results(&mut indexed, col, asc);
        }
        indexed.into_iter().map(|(i, _)| i).collect::<Vec<usize>>()
    });

    view! {
        <div class="bench-section">
            <h3 class="bench-section-title">"Results"</h3>

            {move || {
                let results = state.get().benchmark_results.clone();
                let snapshots = state.get().benchmark_snapshots.clone();
                let cur_sort = sort_col.get();
                let cur_asc = sort_asc.get();

                if results.is_empty() {
                    return view! {
                        <div class="bench-queue-empty">"No results yet. Run a benchmark or import existing results."</div>
                    }.into_any();
                }

                // Build indexed list (original index for color stability)
                let mut indexed: Vec<(usize, BenchmarkResult)> = results.iter().enumerate()
                    .map(|(i, r)| (i, r.clone()))
                    .collect();

                let is_sorted = cur_sort != SortCol::None;
                if is_sorted {
                    sort_results(&mut indexed, cur_sort, cur_asc);
                }

                // Build table body: grouped when unsorted, flat when sorted
                let body_rows = if is_sorted {
                    // Flat sorted — no group headers
                    indexed.iter().map(|(i, r)| {
                        let rid = r.id.clone();
                        let rid2 = r.id.clone();
                        let del = move |_| delete_result(rid.clone());
                        let toggle = move |_: leptos::ev::MouseEvent| toggle_visibility(rid2.clone());
                        let is_hidden = {
                            let rid3 = r.id.clone();
                            Signal::derive(move || hidden_ids.get().contains(&rid3))
                        };
                        result_row(*i, r, del, toggle, is_hidden).into_any()
                    }).collect::<Vec<_>>()
                } else {
                    // Grouped by snapshot_id
                    let mut grouped: Vec<(Option<BenchmarkSnapshot>, Vec<(usize, BenchmarkResult)>)> = vec![];
                    let mut seen_snapshots: Vec<String> = vec![];
                    for (i, r) in &indexed {
                        let snap_id = &r.snapshot_id;
                        if let Some(pos) = seen_snapshots.iter().position(|id| id == snap_id) {
                            grouped[pos].1.push((*i, r.clone()));
                        } else {
                            seen_snapshots.push(snap_id.clone());
                            let snap = snapshots.iter().find(|s| s.id == *snap_id).cloned();
                            grouped.push((snap, vec![(*i, r.clone())]));
                        }
                    }

                    let mut rows: Vec<leptos::prelude::AnyView> = vec![];
                    for (snap, group_results) in grouped {
                        let snap_id = group_results.first().map(|(_, r)| r.snapshot_id.clone()).unwrap_or_default();
                        let is_unassociated = snap.is_none() && snap_id.is_empty();
                        let result_ids: Vec<String> = group_results.iter().map(|(_, r)| r.id.clone()).collect();

                        let header = if let Some(ref s) = snap {
                            view! {
                                <tr class="bench-snapshot-group-header">
                                    <td colspan="13">
                                        {format!("\u{1F4F7} {}", s.name)}
                                    </td>
                                </tr>
                            }.into_any()
                        } else if is_unassociated && !snapshots.is_empty() {
                            let snaps_for_assign = snapshots.clone();
                            view! {
                                <tr class="bench-snapshot-group-header">
                                    <td colspan="13">
                                        <UnassociatedHeader
                                            snapshots={snaps_for_assign}
                                            result_ids={result_ids}
                                        />
                                    </td>
                                </tr>
                            }.into_any()
                        } else if is_unassociated {
                            view! {
                                <tr class="bench-snapshot-group-header">
                                    <td colspan="13">"Unassociated results \u{2014} save a snapshot above to assign them"</td>
                                </tr>
                            }.into_any()
                        } else {
                            view! {
                                <tr class="bench-snapshot-group-header">
                                    <td colspan="13">{format!("(deleted snapshot {})", &snap_id[..8.min(snap_id.len())])}</td>
                                </tr>
                            }.into_any()
                        };
                        rows.push(header);

                        for (i, r) in &group_results {
                            let rid = r.id.clone();
                            let rid2 = r.id.clone();
                            let del = move |_| delete_result(rid.clone());
                            let toggle = move |_: leptos::ev::MouseEvent| toggle_visibility(rid2.clone());
                            let is_hidden = {
                                let rid3 = r.id.clone();
                                Signal::derive(move || hidden_ids.get().contains(&rid3))
                            };
                            rows.push(result_row(*i, r, del, toggle, is_hidden).into_any());
                        }
                    }
                    rows
                };

                let sort_indicator = move |col: SortCol| -> &'static str {
                    if cur_sort == col {
                        if cur_asc { " \u{25B2}" } else { " \u{25BC}" }
                    } else {
                        ""
                    }
                };

                view! {
                    <div>
                        <BenchmarkChart results={results_signal} hidden={hidden_signal} display_order={display_order_signal}/>
                        <div class="bench-results-table-wrap">
                            <table class="bench-results-table">
                                <thead>
                                    <tr>
                                        <th></th>
                                        <th class="bench-th-sortable" on:click={move |_| click_sort(SortCol::Label)}>
                                            {format!("Label{}", sort_indicator(SortCol::Label))}
                                        </th>
                                        <th class="bench-th-sortable" on:click={move |_| click_sort(SortCol::Resolution)}>
                                            {format!("Res{}", sort_indicator(SortCol::Resolution))}
                                        </th>
                                        <th class="bench-th-sortable" on:click={move |_| click_sort(SortCol::Chains)}>
                                            {format!("Chains{}", sort_indicator(SortCol::Chains))}
                                        </th>
                                        <th class="bench-th-sortable" on:click={move |_| click_sort(SortCol::Lambda)}>
                                            {format!("\u{03BB}{}", sort_indicator(SortCol::Lambda))}
                                        </th>
                                        <th class="bench-th-sortable" on:click={move |_| click_sort(SortCol::Batch)}>
                                            {format!("Batch{}", sort_indicator(SortCol::Batch))}
                                        </th>
                                        <th class="bench-th-sortable" on:click={move |_| click_sort(SortCol::Duration)}>
                                            {format!("Duration{}", sort_indicator(SortCol::Duration))}
                                        </th>
                                        <th class="bench-th-sortable" on:click={move |_| click_sort(SortCol::Start)}>
                                            {format!("Start{}", sort_indicator(SortCol::Start))}
                                        </th>
                                        <th class="bench-th-sortable" on:click={move |_| click_sort(SortCol::Final)}>
                                            {format!("Final{}", sort_indicator(SortCol::Final))}
                                        </th>
                                        <th class="bench-th-sortable" on:click={move |_| click_sort(SortCol::Improvements)}>
                                            {format!("Improv{}", sort_indicator(SortCol::Improvements))}
                                        </th>
                                        <th class="bench-th-sortable" on:click={move |_| click_sort(SortCol::ImprovPerSec)}>
                                            {format!("Improv/s{}", sort_indicator(SortCol::ImprovPerSec))}
                                        </th>
                                        <th class="bench-th-sortable" on:click={move |_| click_sort(SortCol::EvalsPerSec)}>
                                            {format!("Evals/s{}", sort_indicator(SortCol::EvalsPerSec))}
                                        </th>
                                        <th></th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {body_rows}
                                </tbody>
                            </table>
                        </div>
                    </div>
                }.into_any()
            }}

            // Always-visible action buttons (import works even with no results)
            <div class="bench-results-actions">
                <button class="btn btn-secondary" on:click={move |_| {
                    let results = state.get().benchmark_results.clone();
                    let text = export_results_text(&results);
                    copy_to_clipboard(&text);
                }}
                    disabled={move || state.get().benchmark_results.is_empty()}
                >
                    "Copy as Text"
                </button>
                <button class="btn btn-secondary" on:click={move |_| {
                    let s = state.get();
                    export_results_json(&s);
                }}
                    disabled={move || state.get().benchmark_results.is_empty()}
                >
                    "Export JSON"
                </button>
                <button class="btn btn-secondary" on:click={move |_| {
                    import_results_from_file();
                }}>
                    "Import JSON"
                </button>
                <button class="btn btn-danger" on:click={clear_results}
                    disabled={move || state.get().benchmark_results.is_empty() && state.get().benchmark_snapshots.is_empty()}
                >
                    "Clear All"
                </button>
            </div>
        </div>
    }
}

fn export_results_text(results: &[BenchmarkResult]) -> String {
    let mut out = String::new();

    // Summary table
    out.push_str("=== Benchmark Results ===\n\n");
    out.push_str(&format!("{:<24} {:>5} {:>6} {:>3} {:>5} {:>8} {:>8} {:>8} {:>6} {:>8} {:>8}\n",
        "Label", "Res", "Chains", "\u{03BB}", "Batch", "Duration", "Start", "Final", "Impr", "Impr/s", "Evals/s"));
    out.push_str(&"-".repeat(110));
    out.push('\n');
    for r in results {
        let actual_secs = if r.actual_duration_secs > 0.0 {
            r.actual_duration_secs as f64
        } else {
            r.duration_secs as f64
        };
        let evals_per_sec = if actual_secs > 0.0 {
            r.total_evals as f64 / actual_secs
        } else {
            0.0
        };
        let duration_str = if r.actual_duration_secs > 0.0 {
            format!("{:.0}s", r.actual_duration_secs)
        } else {
            format!("{}s", r.duration_secs)
        };
        let res_str = if r.resolution > 0 { format!("{}px", r.resolution) } else { "-".to_string() };
        out.push_str(&format!("{:<24} {:>5} {:>6} {:>3} {:>5} {:>8} {:>8} {:>8} {:>6} {:>8.2} {:>8.0}\n",
            r.label, res_str, r.chain_count, r.lambda, r.gpu_batch_iters, duration_str,
            r.start_fitness, r.final_fitness,
            r.total_improvements, r.improvements_per_sec, evals_per_sec));
    }

    // Time series per run
    for r in results {
        out.push_str(&format!("\n--- {} ({}c {}\u{03BB} {}s) ---\n",
            r.label, r.chain_count, r.lambda, r.duration_secs));
        out.push_str(&format!("{:>6} {:>10} {:>10} {:>10} {:>8} {:>12} {:>10}\n",
            "time", "best", "avg", "worst", "impr", "evals", "evals/s"));
        for s in &r.samples {
            out.push_str(&format!("{:>5.0}s {:>9.4}% {:>9.4}% {:>9.4}% {:>8} {:>12} {:>10.0}\n",
                s.elapsed_secs, s.best_fitness, s.avg_fitness, s.worst_fitness,
                s.improvements, s.total_evals, s.evals_per_sec));
        }
    }

    out
}

fn copy_to_clipboard(text: &str) {
    let escaped = text.replace('\\', "\\\\").replace('`', "\\`").replace('$', "\\$");
    let _ = js_sys::eval(&format!("navigator.clipboard.writeText(`{}`)", escaped));
}

fn export_results_json(viewer_state: &ViewerState) {
    let project_name = viewer_state.active_project.as_deref().unwrap_or("unknown").to_string();
    let export = BenchmarkExport {
        project_name: project_name.clone(),
        snapshots: viewer_state.benchmark_snapshots.clone(),
        results: viewer_state.benchmark_results.clone(),
    };
    let json = serde_json::to_string_pretty(&export).unwrap_or_default();
    let filename = format!("benchmarks-{}.json", project_name);
    download_blob(&json, &filename, "application/json");
}

fn import_results_from_file() {
    let window = web_sys::window().expect("no window");
    let document = window.document().expect("no document");

    let input: web_sys::HtmlInputElement = document
        .create_element("input")
        .expect("create input")
        .dyn_into()
        .expect("into input");
    input.set_type("file");
    input.set_attribute("accept", ".json").ok();

    let input_clone = input.clone();
    let onchange = Closure::<dyn Fn()>::new(move || {
        let Some(files) = input_clone.files() else { return };
        let Some(file) = files.get(0) else { return };

        let reader = web_sys::FileReader::new().unwrap();
        let reader_clone = reader.clone();

        let onload = Closure::<dyn Fn()>::new(move || {
            let result = reader_clone.result().unwrap();
            let text = result.as_string().unwrap_or_default();

            // Try BenchmarkExport first, then Vec<BenchmarkResult> for backwards compat
            let data: serde_json::Value = if let Ok(export) = serde_json::from_str::<BenchmarkExport>(&text) {
                serde_json::to_value(&export).unwrap()
            } else if let Ok(results) = serde_json::from_str::<Vec<BenchmarkResult>>(&text) {
                // Wrap old format — send as-is, backend handles it
                serde_json::Value::Array(results.into_iter().map(|r| serde_json::to_value(r).unwrap()).collect())
            } else {
                web_sys::console::error_1(&"Failed to parse benchmark JSON: not a BenchmarkExport or Vec<BenchmarkResult>".into());
                return;
            };

            let msg = serde_json::json!({
                "type": "import_benchmarks",
                "data": data,
            });
            send_ws_json(&msg);
        });

        reader.set_onload(Some(onload.as_ref().unchecked_ref()));
        onload.forget();
        reader.read_as_text(&file).ok();
    });

    input.set_onchange(Some(onchange.as_ref().unchecked_ref()));
    onchange.forget();
    input.click();
}

fn result_row<F, G>(
    i: usize,
    r: &BenchmarkResult,
    on_delete: F,
    on_toggle: G,
    is_hidden: Signal<bool>,
) -> impl IntoView
where
    F: Fn(leptos::ev::MouseEvent) + 'static,
    G: Fn(leptos::ev::MouseEvent) + 'static,
{
    let color = color_for_index(i);
    let actual = if r.actual_duration_secs > 0.0 { r.actual_duration_secs } else { r.duration_secs as f32 };
    let total_secs = actual.round() as u32;
    let mins = total_secs / 60;
    let secs = total_secs % 60;
    let duration_str = if secs == 0 {
        format!("{}:00", mins)
    } else {
        format!("{}:{:02}", mins, secs)
    };
    let actual_secs = if r.actual_duration_secs > 0.0 {
        r.actual_duration_secs as f64
    } else {
        r.duration_secs as f64
    };
    let evals_per_sec = if actual_secs > 0.0 {
        r.total_evals as f64 / actual_secs
    } else {
        0.0
    };
    let evals_str = if evals_per_sec >= 1000.0 {
        format!("{:.1}K", evals_per_sec / 1000.0)
    } else {
        format!("{:.0}", evals_per_sec)
    };

    let color_owned = color.to_string();
    let res_str = if r.resolution > 0 { format!("{}px", r.resolution) } else { "-".to_string() };
    view! {
        <tr class:bench-row-hidden={move || is_hidden.get()}>
            <td>
                <span
                    class="bench-color-dot bench-color-dot-toggle"
                    style={move || {
                        if is_hidden.get() {
                            format!("background: {}; opacity: 0.25", color_owned)
                        } else {
                            format!("background: {}", color_owned)
                        }
                    }}
                    on:click={on_toggle}
                    title="Toggle chart visibility"
                />
            </td>
            <td class="bench-cell-label">{r.label.clone()}</td>
            <td>{res_str}</td>
            <td>{r.chain_count.to_string()}</td>
            <td>{r.lambda.to_string()}</td>
            <td>{r.gpu_batch_iters.to_string()}</td>
            <td>{duration_str}</td>
            <td>{format!("{:.2}%", r.start_fitness)}</td>
            <td class="bench-cell-fitness">{format!("{:.2}%", r.final_fitness)}</td>
            <td>{r.total_improvements.to_string()}</td>
            <td>{format!("{:.2}", r.improvements_per_sec)}</td>
            <td>{evals_str}</td>
            <td>
                <button class="btn-icon btn-icon-danger" on:click={on_delete} title="Delete result">
                    "\u{2715}"
                </button>
            </td>
        </tr>
    }
}
