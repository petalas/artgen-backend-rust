use leptos::prelude::*;

use crate::ws::ViewerState;

#[component]
pub fn GpuStatsPanel(state: RwSignal<ViewerState>) -> impl IntoView {
    let collapsed = RwSignal::new(true);

    let toggle = move |_| {
        collapsed.set(!collapsed.get());
    };

    let has_gpu = move || state.get().gpu_stats.is_some();

    view! {
        {move || {
            if !has_gpu() {
                return view! { <div></div> }.into_any();
            }
            view! {
                <div class="gpu-panel">
                    <div class="gpu-panel-header" on:click={toggle}>
                        <span class="mutation-panel-toggle">{move || if collapsed.get() { "\u{25B6}" } else { "\u{25BC}" }}</span>
                        {move || {
                            let s = state.get();
                            let gs = s.gpu_stats.as_ref();
                            match gs {
                                Some(gs) if gs.timings.total_ms > 0.0 => {
                                    let avg_fitness = if gs.chain_fitness.is_empty() {
                                        0.0
                                    } else {
                                        gs.chain_fitness.iter().sum::<f32>() / gs.chain_fitness.len() as f32
                                    };
                                    let evals_per_sec = s.evals_per_sec;
                                    let evals_k = evals_per_sec / 1000.0;
                                    view! {
                                        <span class="gpu-panel-title">
                                            {format!(
                                                "GPU Evolution \u{2014} {:.1}ms/iter \u{00B7} {:.0}K eval/s \u{00B7} avg {:.1}%",
                                                gs.timings.total_ms,
                                                evals_k,
                                                avg_fitness,
                                            )}
                                        </span>
                                    }.into_any()
                                }
                                _ => {
                                    view! {
                                        <span class="gpu-panel-title">"GPU Evolution"</span>
                                    }.into_any()
                                }
                            }
                        }}
                    </div>
                    {move || {
                        if collapsed.get() {
                            view! { <div></div> }.into_any()
                        } else {
                            view! { <GpuStatsPanelBody state={state}/> }.into_any()
                        }
                    }}
                </div>
            }.into_any()
        }}
    }
}

#[component]
fn GpuStatsPanelBody(state: RwSignal<ViewerState>) -> impl IntoView {
    view! {
        <div class="gpu-panel-body">
            <TimingBar state={state}/>
            <StatsGrid state={state}/>
            <TopChainsChart state={state}/>
        </div>
    }
}

#[component]
fn TimingBar(state: RwSignal<ViewerState>) -> impl IntoView {
    view! {
        {move || {
            let s = state.get();
            let gs = s.gpu_stats.as_ref();
            let gs = match gs {
                Some(g) if g.timings.total_ms > 0.0 => g,
                _ => return view! { <div></div> }.into_any(),
            };
            let t = &gs.timings;
            let total = t.total_ms;

            let segments: Vec<(&str, f32, f32, &str)> = vec![
                ("mutate", t.mutate_ms, t.mutate_pct, "#26a69a"),
                ("rasterize", t.rasterize_error_ms, t.rasterize_error_pct, "#ffb74d"),
                ("select", t.select_ms, t.select_pct, "#7986cb"),
                ("migrate", t.migrate_ms, t.migrate_pct, "#ab47bc"),
            ];

            view! {
                <div class="gpu-timing-section">
                    <div class="mutation-section-title">"Pipeline Timing"</div>
                    <div class="gpu-timing-bar">
                        {segments.iter().filter(|(_, _, pct, _)| *pct > 0.1).map(|(name, ms, pct, color)| {
                            let width_pct = format!("{}%", pct);
                            let bg = color.to_string();
                            let label = if *pct > 15.0 {
                                format!("{} {:.0}%", name, pct)
                            } else if *pct > 8.0 {
                                format!("{:.0}%", pct)
                            } else {
                                String::new()
                            };
                            let title = format!("{}: {:.2}ms ({:.1}%)", name, ms, pct);
                            view! {
                                <div
                                    class="gpu-timing-segment"
                                    style:width={width_pct}
                                    style:background={bg}
                                    title={title}
                                >
                                    {label}
                                </div>
                            }
                        }).collect::<Vec<_>>()}
                    </div>
                    <div class="gpu-timing-total">
                        {format!("{:.2}ms/iter", total)}
                    </div>
                </div>
            }.into_any()
        }}
    }
}

#[component]
fn StatsGrid(state: RwSignal<ViewerState>) -> impl IntoView {
    view! {
        {move || {
            let s = state.get();
            let gs = match s.gpu_stats.as_ref() {
                Some(g) => g.clone(),
                None => return view! { <div></div> }.into_any(),
            };

            let avg_fitness = if gs.chain_fitness.is_empty() {
                0.0
            } else {
                gs.chain_fitness.iter().sum::<f32>() / gs.chain_fitness.len() as f32
            };
            let best_fitness = gs.chain_fitness.first().copied().unwrap_or(0.0);
            let worst_fitness = gs.chain_fitness.last().copied().unwrap_or(0.0);

            view! {
                <div class="gpu-stats-section">
                    <div class="mutation-section-title">"Chain Stats"</div>
                    <div class="gpu-stats-grid">
                        <div class="gpu-stat-cell">
                            <span class="gpu-stat-label">"Chains"</span>
                            <span class="gpu-stat-value">{gs.chain_count.to_string()}</span>
                        </div>
                        <div class="gpu-stat-cell">
                            <span class="gpu-stat-label">"Workgroup"</span>
                            <span class="gpu-stat-value">"16\u{00D7}16"</span>
                        </div>
                        <div class="gpu-stat-cell">
                            <span class="gpu-stat-label">"GPU Memory"</span>
                            <span class="gpu-stat-value">{format!("{:.0} MB", gs.memory_mb)}</span>
                        </div>
                        <div class="gpu-stat-cell">
                            <span class="gpu-stat-label">"Avg Fitness"</span>
                            <span class="gpu-stat-value">{format!("{:.2}%", avg_fitness)}</span>
                        </div>
                        <div class="gpu-stat-cell">
                            <span class="gpu-stat-label">"Best Chain"</span>
                            <span class="gpu-stat-value">{format!("{:.2}%", best_fitness)}</span>
                        </div>
                        <div class="gpu-stat-cell">
                            <span class="gpu-stat-label">"Worst Chain"</span>
                            <span class="gpu-stat-value">{format!("{:.2}%", worst_fitness)}</span>
                        </div>
                    </div>
                </div>
            }.into_any()
        }}
    }
}

#[component]
fn TopChainsChart(state: RwSignal<ViewerState>) -> impl IntoView {
    view! {
        {move || {
            let s = state.get();
            let gs = match s.gpu_stats.as_ref() {
                Some(g) if !g.island_stats.is_empty() => g.clone(),
                _ => return view! { <div></div> }.into_any(),
            };

            let islands = &gs.island_stats;
            if islands.is_empty() {
                return view! { <div></div> }.into_any();
            }

            // Compute scale from all island best/avg values
            let best_max = islands.iter().map(|is| is.best_fitness).fold(f32::NEG_INFINITY, f32::max);
            let avg_min = islands.iter().map(|is| is.avg_fitness).fold(f32::INFINITY, f32::min);
            let floor = (avg_min - 2.0).max(0.0);
            let ceiling = best_max + 0.5;
            let range = ceiling - floor;

            // Color palette for islands (cycle if more than 8)
            let colors = [
                "#26a69a", "#ef5350", "#42a5f5", "#ffb74d",
                "#ab47bc", "#66bb6a", "#ec407a", "#5c6bc0",
            ];

            let island_views: Vec<_> = islands.iter().enumerate().map(|(i, is)| {
                let best_pct = if range > 0.0 {
                    ((is.best_fitness - floor) / range * 100.0).clamp(0.0, 100.0)
                } else {
                    100.0
                };
                let avg_pct = if range > 0.0 {
                    ((is.avg_fitness - floor) / range * 100.0).clamp(0.0, 100.0)
                } else {
                    100.0
                };
                let color = colors[i % colors.len()];
                let best_width = format!("{}%", best_pct);
                let avg_width = format!("{}%", avg_pct);
                let spread = is.best_fitness - is.avg_fitness;
                let title = format!(
                    "Island {}: best {:.2}%, avg {:.2}%, spread {:.2}, {} chains",
                    i, is.best_fitness, is.avg_fitness, spread, is.chain_count,
                );
                view! {
                    <div class="gpu-island-row" title={title}>
                        <span class="gpu-island-label">{format!("I{}", i)}</span>
                        <div class="gpu-island-bar-bg">
                            <div
                                class="gpu-island-bar-avg"
                                style:width={avg_width}
                                style:background={color}
                            ></div>
                            <div
                                class="gpu-island-bar-best"
                                style:width={best_width}
                                style:border-color={color}
                            ></div>
                        </div>
                        <span class="gpu-island-value">{format!("{:.2}%", is.best_fitness)}</span>
                    </div>
                }
            }).collect();

            view! {
                <div class="gpu-fitness-section">
                    <div class="mutation-section-title">{format!("Islands ({})", gs.island_count)}</div>
                    {island_views}
                    <div class="gpu-island-legend">
                        <span class="gpu-island-legend-item">
                            <span class="gpu-island-legend-fill"></span>
                            " avg"
                        </span>
                        <span class="gpu-island-legend-item">
                            <span class="gpu-island-legend-outline"></span>
                            " best"
                        </span>
                    </div>
                </div>
            }.into_any()
        }}
    }
}
