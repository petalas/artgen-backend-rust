use leptos::prelude::*;

use crate::benchmark::{BenchmarkRequest, BenchmarkResult, BenchmarkSnapshot};
use crate::components::benchmark_chart::{color_for_index, BenchmarkChart};
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

// Chains: exponent 4..9 → 16,32,64,128,256,512
fn chains_from_exp(exp: u32) -> u32 { 1u32 << exp }
fn exp_from_chains(chains: u32) -> u32 { chains.max(1).ilog2() }

// Islands: exponent 0..5 → 1,2,4,8,16,32
fn islands_from_exp(exp: u32) -> u32 { 1u32 << exp }
fn exp_from_islands(islands: u32) -> u32 { islands.max(1).ilog2() }

fn build_benchmark_params(
    state: &ViewerState,
    chain_count: u32,
    island_count: u32,
    isolate_islands: bool,
) -> crate::mutation_params::MutationParams {
    let mut params = state.mutation_params.clone();
    params.chain_count = chain_count;
    params.island_count = island_count;
    if isolate_islands {
        params.inter_island_interval = 0;
    }
    params
}

fn auto_label(chain_count: u32, island_count: u32, isolate_islands: bool) -> String {
    if isolate_islands {
        format!("{}c-{}i-isolated", chain_count, island_count)
    } else {
        format!("{}c-{}i", chain_count, island_count)
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
    let isolate_islands = RwSignal::new(false);

    let add_to_queue = move |_| {
        let s = state.get();
        let idx = selected_snap.get();
        let Some(snap) = s.benchmark_snapshots.get(idx) else { return };
        let cc = chains_from_exp(chain_exp.get());
        let ic = islands_from_exp(island_exp.get());
        let iso = isolate_islands.get();
        let params = build_benchmark_params(&s, cc, ic, iso);
        let lbl = label.get();
        let base = if lbl.trim().is_empty() { auto_label(cc, ic, iso) } else { lbl.trim().to_string() };
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
        let iso = isolate_islands.get();
        let params = build_benchmark_params(&s, cc, ic, iso);
        let lbl = label.get();
        let base = if lbl.trim().is_empty() { auto_label(cc, ic, iso) } else { lbl.trim().to_string() };
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
                        min="4" max="9" step="1"
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
                                        {format!("{}s | {}c {}i", req.duration_secs, req.params.chain_count, req.params.island_count)}
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
            <td>{r.island_count.to_string()}</td>
            <td>{duration_str}</td>
            <td>{format!("{:.2}%", r.start_fitness)}</td>
            <td class="bench-cell-fitness">{format!("{:.2}%", r.final_fitness)}</td>
            <td>{r.total_improvements.to_string()}</td>
            <td>{format!("{:.2}", r.improvements_per_sec)}</td>
        </tr>
    }
}
