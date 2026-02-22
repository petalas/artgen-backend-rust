use leptos::prelude::*;

use crate::ws::{send_ws_json, ViewerState};

#[component]
pub fn AutoTuneSection(state: RwSignal<ViewerState>) -> impl IntoView {
    let collapsed = RwSignal::new(true);
    let selected_snap = RwSignal::new(0usize);
    let duration_secs = RwSignal::new(30u32);
    let exploration_trials = RwSignal::new(20u32);
    let elite_fraction = RwSignal::new(25u32); // percentage
    let exploration_rate = RwSignal::new(15u32); // percentage

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
            "explorationTrials": exploration_trials.get(),
            "eliteFraction": elite_fraction.get() as f64 / 100.0,
            "explorationRate": exploration_rate.get() as f64 / 100.0,
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
                            view! {
                                <span class="bench-snapshot-meta" style="margin-left: 8px;">
                                    {format!("Trial {} | Best: {:.2}% (#{}) | {}",
                                        status.trial_number, status.best_fitness, status.best_trial, status.phase)}
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
                                <label class="bench-config-label">"Exploration trials"</label>
                                <input
                                    type="number"
                                    class="project-name-input"
                                    style="width: 70px;"
                                    prop:value={move || exploration_trials.get().to_string()}
                                    on:input={move |ev| {
                                        if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                            exploration_trials.set(v.clamp(5, 200));
                                        }
                                    }}
                                    disabled={move || is_tuning()}
                                />
                            </div>
                            <div class="bench-config-row">
                                <label class="bench-config-label">"Elite %"</label>
                                <input
                                    type="number"
                                    class="project-name-input"
                                    style="width: 70px;"
                                    prop:value={move || elite_fraction.get().to_string()}
                                    on:input={move |ev| {
                                        if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                            elite_fraction.set(v.clamp(5, 50));
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

                            view! {
                                <div style="margin-top: 8px;">
                                    <div class="bench-progress-header">
                                        <span class="bench-progress-label">
                                            {format!("Trial {} | Phase: {} | Best: {:.2}% (trial #{})",
                                                status.trial_number, status.phase,
                                                status.best_fitness, status.best_trial)}
                                        </span>
                                    </div>

                                    // Parameter importance
                                    {if status.param_importance.is_empty() {
                                        view! { <div></div> }.into_any()
                                    } else {
                                        let items = status.param_importance.clone();
                                        view! {
                                            <div style="margin-top: 8px;">
                                                <div style="font-size: 12px; color: #94a3b8; margin-bottom: 4px;">"Parameter Importance (Spearman correlation)"</div>
                                                <div class="auto-tune-importance">
                                                    {items.into_iter().map(|(name, corr)| {
                                                        let bar_width = (corr.abs() * 100.0).min(100.0);
                                                        let bar_color = if corr >= 0.0 { "#22c55e" } else { "#ef4444" };
                                                        view! {
                                                            <div class="auto-tune-importance-row">
                                                                <span class="auto-tune-param-name">{name}</span>
                                                                <div class="auto-tune-bar-bg">
                                                                    <div
                                                                        class="auto-tune-bar-fill"
                                                                        style={format!("width: {}%; background: {}", bar_width, bar_color)}
                                                                    />
                                                                </div>
                                                                <span class="auto-tune-corr-value">{format!("{:.2}", corr)}</span>
                                                            </div>
                                                        }
                                                    }).collect::<Vec<_>>()}
                                                </div>
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
