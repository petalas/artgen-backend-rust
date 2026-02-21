use std::cell::RefCell;

use leptos::prelude::*;
use wasm_bindgen::prelude::*;
use web_sys::{MessageEvent, WebSocket};

use crate::benchmark::{BenchmarkProgress, BenchmarkRequest, BenchmarkResult, BenchmarkSnapshot};
use crate::mutation_params::MutationParams;

#[derive(Clone, Debug, Default)]
pub struct GpuTimings {
    pub mutate_ms: f32,
    pub mutate_pct: f32,
    pub rasterize_error_ms: f32,
    pub rasterize_error_pct: f32,
    pub select_ms: f32,
    pub select_pct: f32,
    pub total_ms: f32,
}

#[derive(Clone, Debug, Default)]
pub struct GpuStats {
    pub chain_count: u32,
    pub memory_mb: f32,
    pub timings: GpuTimings,
    pub chain_fitness: Vec<f32>, // sorted desc
    pub rasterize_wg: [u32; 2],
}

#[allow(dead_code)]
#[derive(Clone, Debug, Default)]
pub struct ProjectInfo {
    pub name: String,
    pub has_best: bool,
    pub fitness: Option<f64>,
    pub polygons: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct ViewerState {
    pub reference_image_b64: String,
    pub generated_image_b64: String,
    pub fitness: f64,
    pub polygons: u32,
    pub improvements: u64,
    pub evals_per_sec: f64,
    pub total_evals: u64,
    pub elapsed_secs: u64,
    pub paused: bool,
    pub connected: bool,
    pub connecting: bool,
    pub drawing_json: Option<String>,
    pub image_width: u32,
    pub image_height: u32,
    // Engine state tracking
    pub engine_loading: bool,
    pub init_received: bool,
    // Resolution control
    pub target_resolution: u32,
    // Project management
    pub projects: Vec<ProjectInfo>,
    pub active_project: Option<String>,
    pub project_error: Option<String>,
    // Mutation parameters
    pub mutation_params: MutationParams,
    // GPU stats
    pub gpu_stats: Option<GpuStats>,
    // Benchmark
    pub benchmark_snapshots: Vec<BenchmarkSnapshot>,
    pub benchmark_results: Vec<BenchmarkResult>,
    pub benchmark_active: bool,
    pub benchmark_progress: Option<BenchmarkProgress>,
    pub benchmark_queue: Vec<BenchmarkRequest>,
}

impl Default for ViewerState {
    fn default() -> Self {
        Self {
            reference_image_b64: String::new(),
            generated_image_b64: String::new(),
            fitness: 0.0,
            polygons: 0,
            improvements: 0,
            evals_per_sec: 0.0,
            total_evals: 0,
            elapsed_secs: 0,
            paused: false,
            connected: false,
            connecting: true,
            drawing_json: None,
            image_width: 0,
            image_height: 0,
            engine_loading: false,
            init_received: false,
            target_resolution: 384,
            projects: vec![],
            active_project: None,
            project_error: None,
            mutation_params: MutationParams::default(),
            gpu_stats: None,
            benchmark_snapshots: vec![],
            benchmark_results: vec![],
            benchmark_active: false,
            benchmark_progress: None,
            benchmark_queue: vec![],
        }
    }
}

thread_local! {
    static WS_HANDLE: RefCell<Option<WebSocket>> = const { RefCell::new(None) };
}

pub fn send_ws_command(cmd_type: &str) {
    WS_HANDLE.with(|ws| {
        if let Some(ws) = ws.borrow().as_ref() {
            let msg = serde_json::json!({ "type": cmd_type });
            ws.send_with_str(&msg.to_string()).ok();
        }
    });
}

pub fn send_ws_json(value: &serde_json::Value) {
    WS_HANDLE.with(|ws| {
        if let Some(ws) = ws.borrow().as_ref() {
            ws.send_with_str(&value.to_string()).ok();
        }
    });
}

/// Send a WS command and set engine_loading = true on the viewer state.
pub fn send_ws_loading(state: RwSignal<ViewerState>, value: &serde_json::Value) {
    state.update(|s| s.engine_loading = true);
    send_ws_json(value);
}

pub fn connect_ws(state: RwSignal<ViewerState>) {
    let window = web_sys::window().expect("no window");
    let hostname = window.location().hostname().unwrap_or_else(|_| "localhost".into());
    let url = format!("ws://{}:9001", hostname);

    let ws = match WebSocket::new(&url) {
        Ok(ws) => ws,
        Err(_) => {
            schedule_reconnect(state);
            return;
        }
    };

    state.update(|s| {
        s.connecting = true;
        s.connected = false;
    });

    // onopen
    let state_open = state;
    let on_open = Closure::<dyn Fn()>::new(move || {
        state_open.update(|s| {
            s.connected = true;
            s.connecting = false;
        });
    });
    ws.set_onopen(Some(on_open.as_ref().unchecked_ref()));
    on_open.forget();

    // onmessage
    let state_msg = state;
    let on_message = Closure::<dyn Fn(MessageEvent)>::new(move |e: MessageEvent| {
        if let Some(text) = e.data().as_string() {
            if let Ok(data) = serde_json::from_str::<serde_json::Value>(&text) {
                handle_message(&data, state_msg);
            }
        }
    });
    ws.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
    on_message.forget();

    // onclose
    let state_close = state;
    let on_close = Closure::<dyn Fn()>::new(move || {
        state_close.update(|s| {
            s.connected = false;
            s.connecting = false;
            s.init_received = false;
            s.engine_loading = false;
        });
        WS_HANDLE.with(|h| h.borrow_mut().take());
        schedule_reconnect(state_close);
    });
    ws.set_onclose(Some(on_close.as_ref().unchecked_ref()));
    on_close.forget();

    // onerror — just let onclose handle reconnect
    let ws_err = ws.clone();
    let on_error = Closure::<dyn Fn()>::new(move || {
        ws_err.close().ok();
    });
    ws.set_onerror(Some(on_error.as_ref().unchecked_ref()));
    on_error.forget();

    // Store ws handle in thread-local
    WS_HANDLE.with(|h| *h.borrow_mut() = Some(ws));
}

fn schedule_reconnect(state: RwSignal<ViewerState>) {
    gloo_timers::callback::Timeout::new(1000, move || {
        connect_ws(state);
    })
    .forget();
}

fn parse_project_list(data: &serde_json::Value) -> Vec<ProjectInfo> {
    data["projects"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .map(|p| ProjectInfo {
                    name: p["name"].as_str().unwrap_or("").to_string(),
                    has_best: p["hasBest"].as_bool().unwrap_or(false),
                    fitness: p["fitness"].as_f64(),
                    polygons: p["polygons"].as_u64().map(|n| n as u32),
                })
                .collect()
        })
        .unwrap_or_default()
}

fn parse_gpu_stats(data: &serde_json::Value) -> Option<GpuStats> {
    let gs = &data["gpuStats"];
    if gs.is_null() {
        return None;
    }
    let timings = if let Some(t) = gs["timings"].as_object() {
        GpuTimings {
            mutate_ms: t.get("mutateMs").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            mutate_pct: t.get("mutatePct").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            rasterize_error_ms: t.get("rasterizeErrorMs").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            rasterize_error_pct: t.get("rasterizeErrorPct").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            select_ms: t.get("selectMs").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            select_pct: t.get("selectPct").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            total_ms: t.get("totalMs").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
        }
    } else {
        GpuTimings::default()
    };
    let chain_fitness = gs["chainFitness"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
        .unwrap_or_default();
    let rasterize_wg = gs["rasterizeWg"]
        .as_array()
        .and_then(|arr| {
            if arr.len() == 2 {
                Some([
                    arr[0].as_u64().unwrap_or(16) as u32,
                    arr[1].as_u64().unwrap_or(16) as u32,
                ])
            } else {
                None
            }
        })
        .unwrap_or([16, 16]);
    Some(GpuStats {
        chain_count: gs["chainCount"].as_u64().unwrap_or(0) as u32,
        memory_mb: gs["memoryMb"].as_f64().unwrap_or(0.0) as f32,
        timings,
        chain_fitness,
        rasterize_wg,
    })
}

fn handle_message(data: &serde_json::Value, state: RwSignal<ViewerState>) {
    let msg_type = data["type"].as_str().unwrap_or("");

    state.update(|s| {
        if let Some(paused) = data["paused"].as_bool() {
            s.paused = paused;
        }
        if let Some(fitness) = data["fitness"].as_f64() {
            s.fitness = fitness;
        }
        if let Some(polygons) = data["polygons"].as_u64() {
            s.polygons = polygons as u32;
        }
        if let Some(improvements) = data["improvements"].as_u64() {
            s.improvements = improvements;
        }
        if let Some(eps) = data["evalsPerSec"].as_f64() {
            s.evals_per_sec = eps;
        }
        if let Some(te) = data["totalEvals"].as_u64() {
            s.total_evals = te;
        }
        if let Some(elapsed) = data["elapsed"].as_u64() {
            s.elapsed_secs = elapsed;
        }
        if let Some(w) = data["imageWidth"].as_u64() {
            s.image_width = w as u32;
        }
        if let Some(h) = data["imageHeight"].as_u64() {
            s.image_height = h as u32;
        }

        match msg_type {
            "init" => {
                s.init_received = true;
                s.engine_loading = false;
                if let Some(ref_img) = data["referenceImage"].as_str() {
                    s.reference_image_b64 = ref_img.to_string();
                }
                if let Some(img) = data["image"].as_str() {
                    s.generated_image_b64 = img.to_string();
                }
                if let Some(dj) = data["drawingJson"].as_str() {
                    s.drawing_json = Some(dj.to_string());
                }
                if let Some(ap) = data["activeProject"].as_str() {
                    s.active_project = Some(ap.to_string());
                } else if data["activeProject"].is_null() {
                    s.active_project = None;
                }
                if let Ok(mp) = serde_json::from_value::<MutationParams>(data["mutationParams"].clone()) {
                    s.mutation_params = mp;
                }
                if let Some(gs) = parse_gpu_stats(data) {
                    s.gpu_stats = Some(gs);
                }
                if let Some(res) = data["targetResolution"].as_u64() {
                    s.target_resolution = res as u32;
                }
            }
            "update" => {
                if let Some(img) = data["image"].as_str() {
                    s.generated_image_b64 = img.to_string();
                }
                if let Some(dj) = data["drawingJson"].as_str() {
                    s.drawing_json = Some(dj.to_string());
                }
                if let Some(gs) = parse_gpu_stats(data) {
                    s.gpu_stats = Some(gs);
                }
            }
            "stats" => {
                // stats-only update, no image data
                if let Some(gs) = parse_gpu_stats(data) {
                    s.gpu_stats = Some(gs);
                }
            }
            "project_list" => {
                s.projects = parse_project_list(data);
                if let Some(ap) = data["activeProject"].as_str() {
                    s.active_project = Some(ap.to_string());
                } else if data["activeProject"].is_null() {
                    s.active_project = None;
                }
            }
            "project_switched" => {
                s.init_received = true;
                s.engine_loading = false;
                // Treat like init — update images and active project
                if let Some(ref_img) = data["referenceImage"].as_str() {
                    s.reference_image_b64 = ref_img.to_string();
                }
                if let Some(img) = data["image"].as_str() {
                    s.generated_image_b64 = img.to_string();
                }
                if let Some(dj) = data["drawingJson"].as_str() {
                    s.drawing_json = Some(dj.to_string());
                }
                if let Some(ap) = data["project"].as_str() {
                    s.active_project = Some(ap.to_string());
                }
                if let Ok(mp) = serde_json::from_value::<MutationParams>(data["mutationParams"].clone()) {
                    s.mutation_params = mp;
                }
                if let Some(res) = data["targetResolution"].as_u64() {
                    s.target_resolution = res as u32;
                }
            }
            "project_error" => {
                s.engine_loading = false;
                if let Some(err) = data["error"].as_str() {
                    s.project_error = Some(err.to_string());
                }
            }
            "benchmark_started" => {
                s.benchmark_active = true;
                let label = data["label"].as_str().unwrap_or("").to_string();
                s.benchmark_progress = Some(BenchmarkProgress {
                    label,
                    elapsed_secs: 0.0,
                    duration_secs: 0,
                    best_fitness: 0.0,
                    improvements: 0,
                });
            }
            "benchmark_progress" => {
                s.benchmark_progress = Some(BenchmarkProgress {
                    label: data["label"].as_str().unwrap_or("").to_string(),
                    elapsed_secs: data["elapsedSecs"].as_f64().unwrap_or(0.0) as f32,
                    duration_secs: data["durationSecs"].as_u64().unwrap_or(0) as u32,
                    best_fitness: data["bestFitness"].as_f64().unwrap_or(0.0) as f32,
                    improvements: data["improvements"].as_u64().unwrap_or(0),
                });
            }
            "benchmark_complete" => {
                s.benchmark_active = false;
                s.benchmark_progress = None;
                if let Ok(result) = serde_json::from_value::<BenchmarkResult>(data["result"].clone()) {
                    s.benchmark_results.push(result);
                }
                // Process next queued benchmark if any
                if !s.benchmark_queue.is_empty() {
                    let next = s.benchmark_queue.remove(0);
                    // We need to send it after the update closure completes
                    // Store it back and handle after update
                    s.benchmark_queue.insert(0, next);
                }
            }
            _ => {}
        }
    });

    // After state update: if benchmark just completed and queue has items, send next
    let should_send_next = state.with_untracked(|s| {
        !s.benchmark_active && !s.benchmark_queue.is_empty() && msg_type == "benchmark_complete"
    });
    if should_send_next {
        let next = state.with_untracked(|s| s.benchmark_queue.first().cloned());
        if let Some(req) = next {
            state.update(|s| { s.benchmark_queue.remove(0); });
            let msg = serde_json::json!({
                "type": "start_benchmark",
                "drawingJson": req.drawing_json,
                "params": req.params,
                "durationSecs": req.duration_secs,
                "label": req.label,
                "resolution": req.resolution,
            });
            send_ws_json(&msg);
        }
    }
}
