use leptos::prelude::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use crate::benchmark::{BenchmarkRequest, BenchmarkResult, BenchmarkSnapshot};
use crate::components::benchmark_chart::{color_for_index, BenchmarkChart};
use crate::components::controls::download_blob;
use crate::ws::{send_ws_json, send_ws_loading, ViewerState};

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
        let snapshot = BenchmarkSnapshot {
            name,
            drawing_json: drawing_json.clone(),
            fitness: s.fitness as f32,
            polygon_count: s.polygons,
        };
        state.update(|s| s.benchmark_snapshots.push(snapshot));
        snap_name.set(String::new());
    };

    let delete_snapshot = move |idx: usize| {
        state.update(|s| { s.benchmark_snapshots.remove(idx); });
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
                            {snaps.into_iter().enumerate().map(|(i, snap)| {
                                let del = move |_| delete_snapshot(i);
                                view! {
                                    <div class="bench-snapshot-item">
                                        <div class="bench-snapshot-info">
                                            <span class="bench-snapshot-name">{snap.name.clone()}</span>
                                            <span class="bench-snapshot-meta">
                                                {format!("{:.2}% | {} polygons", snap.fitness, snap.polygon_count)}
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
fn exp_from_chains(chains: u32) -> u32 { chains.max(1).ilog2() }

// Islands: exponent 0..5 → 1,2,4,8,16,32
fn islands_from_exp(exp: u32) -> u32 { 1u32 << exp }
fn exp_from_islands(islands: u32) -> u32 { islands.max(1).ilog2() }

// Lambda: exponent 0..6 → 1,2,4,8,16,32,64
fn lambda_from_exp(exp: u32) -> u32 { 1u32 << exp }

// Workgroup size options: index -> [wg_x, wg_y]
const WG_OPTIONS: &[[u32; 2]] = &[[16, 16], [16, 8], [8, 8]];

fn wg_label(wg: &[u32; 2]) -> String {
    let threads = wg[0] * wg[1];
    format!("{}x{} ({})", wg[0], wg[1], threads)
}

fn build_benchmark_params(
    state: &ViewerState,
    chain_count: u32,
    island_count: u32,
    lambda: u32,
    isolate_islands: bool,
    single_mutation_mode: bool,
    adaptive_mutation: bool,
    rasterize_wg: [u32; 2],
) -> crate::mutation_params::MutationParams {
    let mut params = state.mutation_params.clone();
    params.chain_count = chain_count;
    params.island_count = island_count;
    params.lambda = lambda;
    if isolate_islands {
        params.inter_island_interval = 0;
    }
    params.single_mutation_mode = single_mutation_mode;
    params.adaptive_mutation = adaptive_mutation;
    params.rasterize_wg = rasterize_wg;
    params
}

fn auto_label(chain_count: u32, island_count: u32, lambda: u32, isolate_islands: bool, single_mutation_mode: bool, adaptive_mutation: bool, rasterize_wg: [u32; 2]) -> String {
    let mode = if single_mutation_mode { "single" } else { "multi" };
    let lambda_str = if lambda > 1 { format!("-{}\u{03BB}", lambda) } else { String::new() };
    let adaptive_str = if adaptive_mutation { "-adaptive" } else { "" };
    let wg_str = if rasterize_wg != [16, 16] {
        format!("-wg{}x{}", rasterize_wg[0], rasterize_wg[1])
    } else {
        String::new()
    };
    if isolate_islands {
        format!("{}c{}-{}i-isolated-{}{}{}", chain_count, lambda_str, island_count, mode, adaptive_str, wg_str)
    } else {
        format!("{}c{}-{}i-{}{}{}", chain_count, lambda_str, island_count, mode, adaptive_str, wg_str)
    }
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
    let chain_exp = RwSignal::new(7u32); // 2^7 = 128
    let island_exp = RwSignal::new(3u32); // 2^3 = 8
    let lambda_exp = RwSignal::new(3u32); // 2^3 = 8 (default lambda=8)
    let isolate_islands = RwSignal::new(false);
    let single_mutation = RwSignal::new(false);
    let adaptive_mutation = RwSignal::new(true);
    let rasterize_wg_idx = RwSignal::new(0usize); // index into WG_OPTIONS, default 0 = 16x16

    let add_to_queue = move |_| {
        let s = state.get();
        let idx = selected_snap.get();
        let Some(snap) = s.benchmark_snapshots.get(idx) else { return };
        let cc = chains_from_exp(chain_exp.get());
        let ic = islands_from_exp(island_exp.get());
        let lam = lambda_from_exp(lambda_exp.get());
        let iso = isolate_islands.get();
        let sm = single_mutation.get();
        let am = adaptive_mutation.get();
        let wg = WG_OPTIONS[rasterize_wg_idx.get()];
        let params = build_benchmark_params(&s, cc, ic, lam, iso, sm, am, wg);
        let lbl = label.get();
        let base = if lbl.trim().is_empty() { auto_label(cc, ic, lam, iso, sm, am, wg) } else { lbl.trim().to_string() };
        let lbl = deduplicate_label(&base, &s);
        let req = BenchmarkRequest {
            drawing_json: snap.drawing_json.clone(),
            params,
            duration_secs: duration_secs.get(),
            label: lbl,
        };
        state.update(|s| s.benchmark_queue.push(req));
        label.set(String::new());
    };

    let run_now = move |_| {
        let s = state.get();
        let idx = selected_snap.get();
        let Some(snap) = s.benchmark_snapshots.get(idx) else { return };
        let cc = chains_from_exp(chain_exp.get());
        let ic = islands_from_exp(island_exp.get());
        let lam = lambda_from_exp(lambda_exp.get());
        let iso = isolate_islands.get();
        let sm = single_mutation.get();
        let am = adaptive_mutation.get();
        let wg = WG_OPTIONS[rasterize_wg_idx.get()];
        let params = build_benchmark_params(&s, cc, ic, lam, iso, sm, am, wg);
        let lbl = label.get();
        let base = if lbl.trim().is_empty() { auto_label(cc, ic, lam, iso, sm, am, wg) } else { lbl.trim().to_string() };
        let lbl = deduplicate_label(&base, &s);
        let msg = serde_json::json!({
            "type": "start_benchmark",
            "drawingJson": snap.drawing_json,
            "params": params,
            "durationSecs": duration_secs.get(),
            "label": lbl,
        });
        send_ws_json(&msg);
        label.set(String::new());
    };

    let has_snapshots = move || !state.get().benchmark_snapshots.is_empty();
    let is_active = move || state.get().benchmark_active;

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
                    <label class="bench-config-label">"Islands"</label>
                    <input
                        type="range"
                        class="mutation-slider"
                        min="0" max="5" step="1"
                        prop:value={move || island_exp.get().to_string()}
                        on:input={move |ev| {
                            if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                island_exp.set(v);
                            }
                        }}
                    />
                    <span class="mutation-value">{move || islands_from_exp(island_exp.get()).to_string()}</span>
                </div>
                <div class="bench-config-row">
                    <label class="bench-config-label">"Migration"</label>
                    <label class="bench-checkbox-label">
                        <input
                            type="checkbox"
                            prop:checked={move || isolate_islands.get()}
                            on:change={move |_| {
                                isolate_islands.set(!isolate_islands.get_untracked());
                            }}
                        />
                        "Isolate islands (no inter-island migration)"
                    </label>
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
                    <label class="bench-config-label">"Resolution"</label>
                    <select
                        class="resolution-select"
                        prop:value={move || state.get().target_resolution.to_string()}
                        on:change={move |ev: web_sys::Event| {
                            let target = ev.target().unwrap();
                            let select: web_sys::HtmlSelectElement = target.dyn_into().unwrap();
                            let val: u32 = select.value().parse().unwrap_or(0);
                            send_ws_loading(state, &serde_json::json!({
                                "type": "update_resolution",
                                "resolution": val,
                            }));
                        }}
                        disabled={move || state.get().engine_loading}
                    >
                        {[0u32, 64, 128, 256, 384, 512, 768, 1024].into_iter().map(|r| {
                            let label = if r == 0 { "Auto (256-512)".to_string() } else { format!("{}px", r) };
                            let val = r.to_string();
                            view! {
                                <option value={val.clone()} selected={move || state.get().target_resolution == r}>{label}</option>
                            }
                        }).collect::<Vec<_>>()}
                    </select>
                    <span class="mutation-value">{move || {
                        let s = state.get();
                        if s.image_width > 0 && s.image_height > 0 {
                            format!("{}x{}", s.image_width, s.image_height)
                        } else {
                            String::new()
                        }
                    }}</span>
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
            state.update(|s| { s.benchmark_queue.remove(0); });
            let msg = serde_json::json!({
                "type": "start_benchmark",
                "drawingJson": req.drawing_json,
                "params": req.params,
                "durationSecs": req.duration_secs,
                "label": req.label,
            });
            send_ws_json(&msg);
        }
    };

    let has_queue = move || !state.get().benchmark_queue.is_empty();
    let is_active = move || state.get().benchmark_active;

    view! {
        <div class="bench-section">
            <h3 class="bench-section-title">"Queue & Progress"</h3>

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
                                        {format!("{}s | {}c {}\u{03BB} {}i{}", req.duration_secs, req.params.chain_count, req.params.lambda, req.params.island_count,
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

// ── Results ────────────────────────────────────────────────

#[component]
fn ResultsSection(state: RwSignal<ViewerState>) -> impl IntoView {
    let clear_results = move |_| {
        state.update(|s| s.benchmark_results.clear());
        let msg = serde_json::json!({ "type": "clear_benchmarks" });
        send_ws_json(&msg);
    };

    let results_signal = Signal::derive(move || state.get().benchmark_results.clone());

    view! {
        <div class="bench-section">
            <h3 class="bench-section-title">"Results"</h3>

            {move || {
                let results = state.get().benchmark_results.clone();
                if results.is_empty() {
                    return view! {
                        <div class="bench-queue-empty">"No results yet. Run a benchmark to see comparisons."</div>
                    }.into_any();
                }
                view! {
                    <div>
                        <BenchmarkChart results={results_signal}/>
                        <div class="bench-results-table-wrap">
                            <table class="bench-results-table">
                                <thead>
                                    <tr>
                                        <th></th>
                                        <th>"Label"</th>
                                        <th>"Chains"</th>
                                        <th>"\u{03BB}"</th>
                                        <th>"Islands"</th>
                                        <th>"Duration"</th>
                                        <th>"Start"</th>
                                        <th>"Final"</th>
                                        <th>"Improv"</th>
                                        <th>"Improv/s"</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {results.iter().enumerate().map(|(i, r)| {
                                        result_row(i, r)
                                    }).collect::<Vec<_>>()}
                                </tbody>
                            </table>
                        </div>
                        <div class="bench-results-actions">
                            <button class="btn btn-secondary" on:click={move |_| {
                                let results = state.get().benchmark_results.clone();
                                let text = export_results_text(&results);
                                copy_to_clipboard(&text);
                            }}>
                                "Copy as Text"
                            </button>
                            <button class="btn btn-secondary" on:click={move |_| {
                                let results = state.get().benchmark_results.clone();
                                export_results_json(&results);
                            }}>
                                "Export JSON"
                            </button>
                            <button class="btn btn-secondary" on:click={move |_| {
                                import_results_from_file(state);
                            }}>
                                "Import JSON"
                            </button>
                            <button class="btn btn-danger" on:click={clear_results}>
                                "Clear Results"
                            </button>
                        </div>
                    </div>
                }.into_any()
            }}
        </div>
    }
}

fn export_results_text(results: &[BenchmarkResult]) -> String {
    let mut out = String::new();

    // Summary table
    out.push_str("=== Benchmark Results ===\n\n");
    out.push_str(&format!("{:<24} {:>6} {:>3} {:>7} {:>8} {:>8} {:>8} {:>6} {:>8}\n",
        "Label", "Chains", "\u{03BB}", "Islands", "Duration", "Start", "Final", "Impr", "Impr/s"));
    out.push_str(&"-".repeat(96));
    out.push('\n');
    for r in results {
        out.push_str(&format!("{:<24} {:>6} {:>3} {:>7} {:>7}s {:>7.2}% {:>7.2}% {:>6} {:>8.2}\n",
            r.label, r.chain_count, r.lambda, r.island_count, r.duration_secs,
            r.start_fitness, r.final_fitness,
            r.total_improvements, r.improvements_per_sec));
    }

    // Time series per run
    for r in results {
        out.push_str(&format!("\n--- {} ({}c {}\u{03BB} {}i {}s) ---\n",
            r.label, r.chain_count, r.lambda, r.island_count, r.duration_secs));
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

fn export_results_json(results: &[BenchmarkResult]) {
    let json = serde_json::to_string_pretty(results).unwrap_or_default();
    download_blob(&json, "benchmark-results.json", "application/json");
}

fn import_results_from_file(state: RwSignal<ViewerState>) {
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
            match serde_json::from_str::<Vec<BenchmarkResult>>(&text) {
                Ok(imported) => {
                    state.update(|s| s.benchmark_results.extend(imported));
                }
                Err(e) => {
                    web_sys::console::error_1(&format!("Failed to parse benchmark JSON: {}", e).into());
                }
            }
        });

        reader.set_onload(Some(onload.as_ref().unchecked_ref()));
        onload.forget();
        reader.read_as_text(&file).ok();
    });

    input.set_onchange(Some(onchange.as_ref().unchecked_ref()));
    onchange.forget();
    input.click();
}

fn result_row(i: usize, r: &BenchmarkResult) -> impl IntoView {
    let color = color_for_index(i);
    let mins = r.duration_secs / 60;
    let secs = r.duration_secs % 60;
    let duration_str = if secs == 0 {
        format!("{}:00", mins)
    } else {
        format!("{}:{:02}", mins, secs)
    };

    view! {
        <tr>
            <td>
                <span
                    class="bench-color-dot"
                    style={format!("background: {}", color)}
                />
            </td>
            <td class="bench-cell-label">{r.label.clone()}</td>
            <td>{r.chain_count.to_string()}</td>
            <td>{r.lambda.to_string()}</td>
            <td>{r.island_count.to_string()}</td>
            <td>{duration_str}</td>
            <td>{format!("{:.2}%", r.start_fitness)}</td>
            <td class="bench-cell-fitness">{format!("{:.2}%", r.final_fitness)}</td>
            <td>{r.total_improvements.to_string()}</td>
            <td>{format!("{:.2}", r.improvements_per_sec)}</td>
        </tr>
    }
}
