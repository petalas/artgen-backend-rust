use leptos::prelude::*;

use crate::ws::{send_ws_json, ViewerState};

#[component]
pub fn AutoTuneSection(state: RwSignal<ViewerState>) -> impl IntoView {
    let collapsed = RwSignal::new(true);
    let selected_snap = RwSignal::new(0usize);
    let duration_secs = RwSignal::new(30u32);
    let replicates = RwSignal::new(1u32);
    let step_size_pct = RwSignal::new(15u32); // percentage, maps to initial_step_size
    let step_decay = RwSignal::new(50u32); // percentage, maps to step_decay (50 = 0.5)

    let has_snapshots = move || !state.get().benchmark_snapshots.is_empty();
    let is_busy = move || {
        let s = state.get();
        s.benchmark_active || s.benchmark_initializing
    };
    let is_tuning = move || {
        state.get().auto_tune_status.as_ref().map(|s| s.running).unwrap_or(false)
    };
    let has_saved_state = move || {
        state.get().auto_tune_status.as_ref().map(|s| s.trial_number > 0).unwrap_or(false)
    };

    let start_auto_tune = move |_| {
        let s = state.get();
        let snaps = &s.benchmark_snapshots;
        let idx = selected_snap.get();
        let Some(snap) = snaps.get(idx) else { return };
        let msg = serde_json::json!({
            "type": "start_auto_tune",
            "snapshotId": snap.id,
            "durationSecs": duration_secs.get(),
            "resolution": s.target_resolution,
            "replicates": replicates.get(),
            "initialStepSize": step_size_pct.get() as f64 / 100.0,
            "stepDecay": step_decay.get() as f64 / 100.0,
            "minStepSize": 0.03,
        });
        send_ws_json(&msg);
    };

    let stop_auto_tune = move |_| {
        send_ws_json(&serde_json::json!({ "type": "stop_auto_tune" }));
    };

    let resume_auto_tune = move |_| {
        send_ws_json(&serde_json::json!({ "type": "resume_auto_tune" }));
    };

    view! {
        <div class="bench-section">
            <div class="mutation-panel-header" on:click={move |_| collapsed.set(!collapsed.get())}>
                <span class="mutation-panel-toggle">{move || if collapsed.get() { "\u{25B6}" } else { "\u{25BC}" }}</span>
                <span class="mutation-panel-title">"Auto-Tune"</span>
                {move || {
                    if let Some(status) = &state.get().auto_tune_status {
                        if status.running {
                            let param_str = status.current_param.as_deref().unwrap_or("baseline");
                            let dir_str = status.current_direction.as_deref().unwrap_or("");
                            view! {
                                <span class="bench-snapshot-meta" style="margin-left: 8px;">
                                    {format!("Pass {}/{} | {} {} | Step: {:.0}% | Base: {:.2}%",
                                        status.pass,
                                        estimate_total_passes(status.step_size, status.config.step_decay, status.config.min_step_size, status.pass),
                                        param_str, dir_str,
                                        status.step_size * 100.0,
                                        status.base_fitness)}
                                </span>
                            }.into_any()
                        } else if status.phase == "done" {
                            view! {
                                <span class="bench-snapshot-meta" style="margin-left: 8px;">
                                    {format!("Complete ({} trials, best: {:.2}%)", status.trial_number, status.best_fitness)}
                                </span>
                            }.into_any()
                        } else if status.trial_number > 0 {
                            view! {
                                <span class="bench-snapshot-meta" style="margin-left: 8px;">
                                    {format!("Stopped ({} trials, best: {:.2}%)", status.trial_number, status.best_fitness)}
                                </span>
                            }.into_any()
                        } else {
                            view! { <span></span> }.into_any()
                        }
                    } else {
                        view! { <span></span> }.into_any()
                    }
                }}
            </div>

            {move || {
                if collapsed.get() {
                    return view! { <div></div> }.into_any();
                }

                view! {
                    <div style="padding: 8px 0;">
                        // Config row
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
                                    disabled={move || !has_snapshots() || is_tuning()}
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
                                <label class="bench-config-label">"Duration/trial"</label>
                                <div class="bench-radio-group">
                                    {[15u32, 30, 60, 120, 300].into_iter().map(|d| {
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
                                                    name="at_duration"
                                                    value={d.to_string()}
                                                    checked={move || duration_secs.get() == d}
                                                    on:change={move |_| duration_secs.set(d)}
                                                    disabled={move || is_tuning()}
                                                />
                                                {label_text}
                                            </label>
                                        }
                                    }).collect::<Vec<_>>()}
                                </div>
                            </div>
                            <div class="bench-config-row">
                                <label class="bench-config-label">"Replicates"</label>
                                <input
                                    type="number"
                                    class="project-name-input"
                                    style="width: 70px;"
                                    min="1"
                                    max="5"
                                    prop:value={move || replicates.get().to_string()}
                                    on:input={move |ev| {
                                        if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                            replicates.set(v.clamp(1, 5));
                                        }
                                    }}
                                    disabled={move || is_tuning()}
                                />
                            </div>
                            <div class="bench-config-row">
                                <label class="bench-config-label">"Step size %"</label>
                                <input
                                    type="number"
                                    class="project-name-input"
                                    style="width: 70px;"
                                    min="2"
                                    max="25"
                                    prop:value={move || step_size_pct.get().to_string()}
                                    on:input={move |ev| {
                                        if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                            step_size_pct.set(v.clamp(2, 25));
                                        }
                                    }}
                                    disabled={move || is_tuning()}
                                />
                            </div>
                            <div class="bench-config-row">
                                <label class="bench-config-label">"Step decay %"</label>
                                <input
                                    type="number"
                                    class="project-name-input"
                                    style="width: 70px;"
                                    min="30"
                                    max="80"
                                    prop:value={move || step_decay.get().to_string()}
                                    on:input={move |ev| {
                                        if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                            step_decay.set(v.clamp(30, 80));
                                        }
                                    }}
                                    disabled={move || is_tuning()}
                                />
                            </div>
                        </div>

                        // Actions
                        <div class="bench-config-actions">
                            <button
                                class="btn btn-primary"
                                on:click={start_auto_tune}
                                disabled={move || !has_snapshots() || is_busy() || is_tuning()}
                            >
                                "Start"
                            </button>
                            <button
                                class="btn btn-danger"
                                on:click={stop_auto_tune}
                                disabled={move || !is_tuning()}
                            >
                                "Stop"
                            </button>
                            <button
                                class="btn btn-secondary"
                                on:click={resume_auto_tune}
                                disabled={move || is_tuning() || is_busy() || !has_saved_state()}
                            >
                                "Resume"
                            </button>
                        </div>

                        // Status display
                        {move || {
                            let s = state.get();
                            let Some(status) = &s.auto_tune_status else {
                                return view! { <div></div> }.into_any();
                            };
                            if status.trial_number == 0 && !status.running {
                                return view! { <div></div> }.into_any();
                            }

                            let param_str = status.current_param.as_deref().unwrap_or("baseline");
                            let dir_str = status.current_direction.as_deref().unwrap_or("");

                            view! {
                                <div style="margin-top: 8px;">
                                    <div class="bench-progress-header">
                                        <span class="bench-progress-label">
                                            {format!("Trial {} | Pass {} | {} {} | Step: {:.0}% | Base: {:.2}% | Best: {:.2}% (#{})",
                                                status.trial_number, status.pass,
                                                param_str, dir_str,
                                                status.step_size * 100.0,
                                                status.base_fitness,
                                                status.best_fitness, status.best_trial)}
                                        </span>
                                    </div>

                                    // Parameter probe results
                                    {if status.param_results.is_empty() {
                                        view! { <div></div> }.into_any()
                                    } else {
                                        let items = status.param_results.clone();
                                        view! {
                                            <div style="margin-top: 8px;">
                                                <div style="font-size: 12px; color: #94a3b8; margin-bottom: 4px;">"Parameter Probe Results (current pass)"</div>
                                                <table class="auto-tune-results-table">
                                                    <thead>
                                                        <tr>
                                                            <th>"Param"</th>
                                                            <th>"Base"</th>
                                                            <th>"High"</th>
                                                            <th>"Low"</th>
                                                            <th>"Chosen"</th>
                                                            <th>"Delta"</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        {items.into_iter().map(|pr| {
                                                            let chosen_class = match pr.chosen.as_str() {
                                                                "high" | "low" => "auto-tune-improved",
                                                                "skip" => "auto-tune-skipped",
                                                                _ => "",
                                                            };
                                                            view! {
                                                                <tr class={chosen_class}>
                                                                    <td class="auto-tune-param-name">{pr.param_name}</td>
                                                                    <td>{format!("{:.3}%", pr.base_fitness)}</td>
                                                                    <td>{pr.high_fitness.map(|f| format!("{:.3}%", f)).unwrap_or_else(|| "\u{2014}".to_string())}</td>
                                                                    <td>{pr.low_fitness.map(|f| format!("{:.3}%", f)).unwrap_or_else(|| "\u{2014}".to_string())}</td>
                                                                    <td>{pr.chosen.clone()}</td>
                                                                    <td>{if pr.improvement.abs() > 0.0001 {
                                                                        format!("{:+.4}%", pr.improvement)
                                                                    } else {
                                                                        "\u{2014}".to_string()
                                                                    }}</td>
                                                                </tr>
                                                            }
                                                        }).collect::<Vec<_>>()}
                                                    </tbody>
                                                </table>
                                            </div>
                                        }.into_any()
                                    }}
                                </div>
                            }.into_any()
                        }}
                    </div>
                }.into_any()
            }}
        </div>
    }
}

/// Estimate total passes from current state.
fn estimate_total_passes(current_step: f32, decay: f32, min_step: f32, current_pass: u32) -> u32 {
    let mut step = current_step;
    let mut passes = current_pass;
    while step >= min_step {
        step *= decay;
        if step >= min_step {
            passes += 1;
        }
    }
    passes
}
