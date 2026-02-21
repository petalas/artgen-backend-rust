use artgen_backend_rust::{
    benchmark::{BenchmarkRequest, BenchmarkResult, BenchmarkSample},
    engine::{Engine, Rasterizer},
    evaluator::{Evaluator, EvaluatorPayload},
    gpu_evolver::GpuEvolver,
    models::drawing::Drawing,
    mutation_params::MutationParams,
    projects,
    settings::{
        DISPLAY_H, DISPLAY_W, MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH, MIN_IMAGE_HEIGHT, MIN_IMAGE_WIDTH,
        TARGET_FRAMETIME,
    },
    utils::print_stats,
};

use sdl2::keyboard::Keycode;
use sdl2::{event::Event, pixels::PixelFormatEnum};
use std::{
    env,
    net::TcpListener,
    path::Path,
    process::exit,
    sync::{
        mpsc::{self, channel},
        Arc, Condvar, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use image::ImageEncoder;
use tokio::sync::broadcast;

const DEFAULT_REF_IMAGE_FILENAME: &str = "ff.jpg";
const DEFAULT_SAVE_PATH: &str = "ff.best.json";

fn evaluate(work_sender: mpsc::Sender<EvaluatorPayload>, mut evaluator: Evaluator) {
    loop {
        let update = evaluator.produce_new_best();
        evaluator.reset(update.best.clone());
        let _ = work_sender.send(update);
    }
}

fn initialize_engine(ref_image_filename: &str) -> Engine {
    let mut engine = Engine {
        raster_mode: Rasterizer::HalfSpace,
        ..Default::default()
    };
    engine.init(
        ref_image_filename,
        MIN_IMAGE_WIDTH,
        MIN_IMAGE_HEIGHT,
        MAX_IMAGE_WIDTH,
        MAX_IMAGE_HEIGHT,
    );
    engine
}

/// Initialize an Engine from raw RGBA data + dimensions (for project mode).
fn initialize_engine_from_rgba(rgba: &[u8], w: usize, h: usize) -> Engine {
    let mut engine = Engine {
        raster_mode: Rasterizer::HalfSpace,
        ..Default::default()
    };
    engine.w = w;
    engine.h = h;
    let size = w * h * 4;
    engine.ref_image_data = rgba.to_vec();
    engine.working_data = vec![0u8; size];
    engine.error_data = vec![0u8; size];
    engine.initialized = true;
    engine
}

fn main() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn,artgen_backend_rust=info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let args: Vec<String> = env::args().collect();

    let use_gpu = args.iter().any(|a| a == "--gpu");
    let headless = args.iter().any(|a| a == "--headless");
    let run_bench = args.iter().any(|a| a == "--bench");

    // Parse --gpu-batch-iters N (number of iterations per GPU batch submission)
    let gpu_batch_iters_override: Option<u32> = args
        .windows(2)
        .find(|w| w[0] == "--gpu-batch-iters")
        .and_then(|w| w[1].parse().ok());

    // Filter out flags and their values to get positional args
    let mut positional: Vec<&String> = Vec::new();
    let mut skip_next = false;
    for (_i, a) in args.iter().enumerate().skip(1) {
        if skip_next {
            skip_next = false;
            continue;
        }
        if a == "--gpu-batch-iters" {
            skip_next = true;
            continue;
        }
        if !a.starts_with("--") {
            positional.push(a);
        }
    }

    if run_bench {
        // Quick GPU benchmark mode — no WS server, no project system
        let project = positional.first().map(|s| s.as_str()).unwrap_or("ff");
        run_cli_benchmark(project, gpu_batch_iters_override);
        return;
    }

    if use_gpu && headless {
        // Project management mode — image arg is optional
        let legacy_image = positional.first().map(|s| s.as_str());
        println!("Starting GPU evolution pipeline (headless, project management mode)...");
        gpu_main_loop_headless(legacy_image, gpu_batch_iters_override);
        return;
    }

    // Legacy mode: require image arg
    let ref_image_filename = if !positional.is_empty() {
        positional[0].as_str()
    } else {
        DEFAULT_REF_IMAGE_FILENAME
    };

    let json_filename = if !positional.is_empty() {
        format!(
            "{}.best.json",
            Path::new(ref_image_filename)
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap()
        )
    } else {
        DEFAULT_SAVE_PATH.to_string()
    };

    println!("Using {:?} -> {:?}", ref_image_filename, json_filename);

    let engine = initialize_engine(ref_image_filename);

    let best = if Path::new(&json_filename).exists() {
        Drawing::from_file(&json_filename)
    } else {
        println!(
            "Could not find saved best for {:?}, starting from scratch...",
            json_filename
        );
        Drawing::new_random()
    };

    let (sdl_context, mut canvas, texture_creator) = initialize_sdl(DISPLAY_W, DISPLAY_H);
    let mut texture = texture_creator
        .create_texture_streaming(PixelFormatEnum::ABGR8888, DISPLAY_W, DISPLAY_H)
        .unwrap();

    canvas.clear();
    canvas.present();

    if use_gpu {
        println!("Starting GPU evolution pipeline...");
        gpu_main_loop(
            &sdl_context,
            &mut texture,
            &mut canvas,
            &engine,
            best,
            &json_filename,
        );
    } else {
        cpu_main(
            &sdl_context,
            &mut texture,
            &mut canvas,
            &engine,
            best,
            &json_filename,
        );
    }
}

fn cpu_main(
    sdl_context: &sdl2::Sdl,
    texture: &mut sdl2::render::Texture,
    canvas: &mut sdl2::render::WindowCanvas,
    engine: &Engine,
    best: Drawing,
    json_filename: &str,
) {
    let num_threads = num_cpus::get();

    // channel to send work to worker threads
    let (work_sender, work_receiver) = channel::<EvaluatorPayload>();

    // broadcast channel to send out new best to all workers
    let (best_sender, best_receiver) = broadcast::channel::<Drawing>(num_threads);

    // spawn a bunch of worker threads, giving each a sender
    let workers = (0..num_threads)
        .map(|_| {
            let ws = work_sender.clone();
            let br = best_sender.subscribe();
            let evaluator = Evaluator::new(
                engine.ref_image_data.clone(),
                engine.w,
                engine.h,
                best.clone(),
                br,
            );
            thread::spawn(move || evaluate(ws, evaluator))
        })
        .collect::<Vec<_>>();

    drop(work_sender);
    drop(best_receiver);

    let global_best = best.clone();

    main_loop(
        sdl_context,
        &work_receiver,
        &best_sender,
        texture,
        canvas,
        global_best,
        json_filename,
    );

    for worker in workers {
        worker.join().expect("worker panicked");
    }
}

fn gpu_main_loop(
    sdl_context: &sdl2::Sdl,
    texture: &mut sdl2::render::Texture,
    canvas: &mut sdl2::render::WindowCanvas,
    engine: &Engine,
    initial_best: Drawing,
    json_filename: &str,
) {
    let mut event_pump = sdl_context.event_pump().unwrap();
    let frametime = Duration::from_millis(TARGET_FRAMETIME);

    // Initialize GPU evolver
    let mut evolver = futures_lite::future::block_on(GpuEvolver::new(
        &engine.ref_image_data,
        engine.w as u32,
        engine.h as u32,
        &initial_best,
    ));

    let mut global_best = initial_best;
    let mut last_draw_timestamp = Instant::now() - frametime;
    let mut last_save_timestamp = Instant::now();
    let mut last_stats_timestamp = Instant::now();

    let default_params = MutationParams::default();
    loop {
        // Run a batch of GPU iterations
        if let Some(new_best) = evolver.run_batch(&default_params, true) {
            if new_best.fitness > global_best.fitness {
                global_best = new_best;

                let since_last_save = last_save_timestamp.elapsed().as_secs();
                if since_last_save >= 10 {
                    global_best.to_file(json_filename);
                    last_save_timestamp = Instant::now();
                }
            }
        }

        // Display at ~30fps
        let elapsed = last_draw_timestamp.elapsed();
        if elapsed >= frametime {
            // Draw upscaled image
            let mut upscale_buf = vec![0u8; DISPLAY_W as usize * DISPLAY_H as usize * 4];
            global_best.draw(
                &mut upscale_buf,
                DISPLAY_W as usize,
                DISPLAY_H as usize,
                Rasterizer::HalfSpace,
            );

            texture
                .update(None, &upscale_buf, DISPLAY_W as usize * 4)
                .unwrap();
            canvas.copy(texture, None, None).unwrap();
            canvas.present();
            last_draw_timestamp = Instant::now();

            exhaust_event_pump(&mut event_pump);
        }

        // Print stats periodically
        if last_stats_timestamp.elapsed().as_secs() >= 2 {
            print_gpu_stats(&evolver, &global_best);
            evolver.pass_timings().print_averages();
            evolver.reset_pass_timings();
            last_stats_timestamp = Instant::now();
        }
    }
}

struct GpuPassTimingsWs {
    mutate_ms: f32,
    mutate_pct: f32,
    rasterize_error_ms: f32,
    rasterize_error_pct: f32,
    select_ms: f32,
    select_pct: f32,
    total_ms: f32,
}

struct GpuStatsWs {
    chain_count: u32,
    memory_mb: f32,
    timings: Option<GpuPassTimingsWs>,
    chain_fitness: Vec<f32>, // sorted desc
    rasterize_wg: [u32; 2],
}

impl GpuStatsWs {
    fn to_json(&self) -> serde_json::Value {
        let timings = self.timings.as_ref().map(|t| {
            serde_json::json!({
                "mutateMs": t.mutate_ms,
                "mutatePct": t.mutate_pct,
                "rasterizeErrorMs": t.rasterize_error_ms,
                "rasterizeErrorPct": t.rasterize_error_pct,
                "selectMs": t.select_ms,
                "selectPct": t.select_pct,
                "totalMs": t.total_ms,
            })
        });
        serde_json::json!({
            "chainCount": self.chain_count,
            "memoryMb": self.memory_mb,
            "timings": timings,
            "chainFitness": self.chain_fitness,
            "rasterizeWg": self.rasterize_wg,
        })
    }
}

struct WsState {
    ref_png: Vec<u8>,
    best_png: Vec<u8>,
    fitness: f32,
    polygons: usize,
    improvements: u64,
    evals_per_sec: f64,
    total_evals: u64,
    elapsed_secs: u64,
    generation: u64,
    image_generation: u64,
    ref_image_generation: u64, // bumped when reference image changes (project switch)
    paused: bool,
    drawing_json: String,
    image_width: u32,
    image_height: u32,
    // Project management
    active_project: Option<String>,
    project_list_generation: u64,
    switch_request: Option<String>,
    reset_active: bool,   // skip saving on switch when true (reset deletes best.json)
    pending_delete: Option<String>, // project dir to delete after inner loop breaks
    // Resolution control (0 = use native clamped resolution)
    target_resolution: u32,
    // Mutation parameters (runtime-configurable)
    mutation_params: MutationParams,
    // GPU stats
    gpu_stats: Option<GpuStatsWs>,
    // Benchmark
    benchmark_request: Option<BenchmarkRequest>,
    benchmark_results: Vec<BenchmarkResult>,
    benchmark_active: bool,
    benchmark_events: Vec<serde_json::Value>, // queued events for WS clients
}

type SharedWsState = Arc<(Mutex<WsState>, Condvar)>;

fn encode_rgba_as_png(rgba: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut buf = Vec::new();
    image::codecs::png::PngEncoder::new(&mut buf)
        .write_image(rgba, w as u32, h as u32, image::ExtendedColorType::Rgba8)
        .expect("PNG encoding failed");
    buf
}

fn ws_server(state: SharedWsState) {
    let listener = TcpListener::bind("0.0.0.0:9001").expect("Failed to bind WS port 9001");
    println!("[WS] Server listening on 0.0.0.0:9001");

    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[WS] Accept error: {}", e);
                continue;
            }
        };
        let state = state.clone();
        thread::spawn(move || ws_handle_client(stream, state));
    }
}

fn build_project_list_msg(active_project: &Option<String>) -> serde_json::Value {
    let project_list = projects::list_projects();
    serde_json::json!({
        "type": "project_list",
        "projects": project_list,
        "activeProject": active_project,
    })
}

fn ws_handle_client(stream: std::net::TcpStream, state: SharedWsState) {
    let peer = stream.peer_addr().ok();

    // Configure WebSocket to accept large frames (up to 20MB for image uploads)
    let mut config = tungstenite::protocol::WebSocketConfig::default();
    config.max_message_size = Some(20 * 1024 * 1024);
    config.max_frame_size = Some(20 * 1024 * 1024);

    let mut ws = match tungstenite::accept_with_config(stream, Some(config)) {
        Ok(ws) => ws,
        Err(e) => {
            eprintln!("[WS] Handshake error: {}", e);
            return;
        }
    };

    println!("[WS] Client connected: {:?}", peer);

    let (lock, _cvar) = &*state;

    // Throttle: stats at ~30fps, images at ~5fps (large base64 payloads)
    let stats_interval = Duration::from_millis(33);
    let image_interval = Duration::from_millis(200);
    let mut last_send_time = Instant::now();
    let mut last_image_send_time = Instant::now();

    // Send init message + project list (blocking mode)
    let mut last_gen;
    let mut last_image_gen;
    let mut last_ref_image_gen;
    let mut last_project_list_gen;
    {
        let s = lock.lock().unwrap();
        last_gen = s.generation;
        last_image_gen = s.image_generation;
        last_ref_image_gen = s.ref_image_generation;
        last_project_list_gen = s.project_list_generation;

        // Send init message
        let gpu_stats_json = s.gpu_stats.as_ref().map(|g| g.to_json());
        let msg = serde_json::json!({
            "type": "init",
            "referenceImage": BASE64.encode(&s.ref_png),
            "image": BASE64.encode(&s.best_png),
            "fitness": s.fitness,
            "polygons": s.polygons,
            "improvements": s.improvements,
            "evalsPerSec": s.evals_per_sec,
            "totalEvals": s.total_evals,
            "elapsed": s.elapsed_secs,
            "paused": s.paused,
            "drawingJson": s.drawing_json,
            "imageWidth": s.image_width,
            "imageHeight": s.image_height,
            "activeProject": s.active_project,
            "mutationParams": s.mutation_params,
            "gpuStats": gpu_stats_json,
            "targetResolution": s.target_resolution,
        });
        if ws
            .send(tungstenite::Message::Text(msg.to_string().into()))
            .is_err()
        {
            println!("[WS] Client disconnected during init: {:?}", peer);
            return;
        }

        // Send project list
        let pl_msg = build_project_list_msg(&s.active_project);
        if ws
            .send(tungstenite::Message::Text(pl_msg.to_string().into()))
            .is_err()
        {
            println!("[WS] Client disconnected during project_list: {:?}", peer);
            return;
        }
    }

    // Switch to non-blocking so we can interleave reads (commands) and writes (updates)
    if ws.get_ref().set_nonblocking(true).is_err() {
        return;
    }

    loop {
        // 1. Drain incoming commands from client
        loop {
            match ws.read() {
                Ok(tungstenite::Message::Text(text)) => {
                    if let Ok(cmd) = serde_json::from_str::<serde_json::Value>(&text) {
                        let reply = handle_ws_command(&cmd, &state, &peer);
                        if let Some(reply_msg) = reply {
                            // Send reply in blocking mode
                            ws.get_ref().set_nonblocking(false).ok();
                            let result = ws.send(tungstenite::Message::Text(reply_msg.to_string().into()));
                            ws.get_ref().set_nonblocking(true).ok();
                            if result.is_err() {
                                println!("[WS] Client disconnected: {:?}", peer);
                                return;
                            }
                        }
                    }
                }
                Ok(tungstenite::Message::Close(_)) => {
                    println!("[WS] Client disconnected: {:?}", peer);
                    return;
                }
                Err(tungstenite::Error::Io(ref e))
                    if e.kind() == std::io::ErrorKind::WouldBlock =>
                {
                    break; // no more data
                }
                Err(_) => {
                    println!("[WS] Client disconnected: {:?}", peer);
                    return;
                }
                _ => {}
            }
        }

        // 2. Throttle: sleep to cap at ~30 stats updates/sec
        let since_last_send = last_send_time.elapsed();
        if since_last_send < stats_interval {
            thread::sleep(stats_interval - since_last_send);
        }
        last_send_time = Instant::now();

        // 3. Read current state and send appropriate message
        let msg_string = {
            let mut s = lock.lock().unwrap();

            // Project list changed (small message, send immediately)
            if s.project_list_generation != last_project_list_gen {
                last_project_list_gen = s.project_list_generation;
                let pl_msg = build_project_list_msg(&s.active_project);
                drop(s);
                ws.get_ref().set_nonblocking(false).ok();
                let result = ws.send(tungstenite::Message::Text(pl_msg.to_string().into()));
                ws.get_ref().set_nonblocking(true).ok();
                if result.is_err() {
                    break;
                }
                continue;
            }

            // Drain benchmark events (small messages, send immediately)
            if !s.benchmark_events.is_empty() {
                let events: Vec<serde_json::Value> = s.benchmark_events.drain(..).collect();
                drop(s);
                ws.get_ref().set_nonblocking(false).ok();
                let mut failed = false;
                for evt in events {
                    if ws.send(tungstenite::Message::Text(evt.to_string().into())).is_err() {
                        failed = true;
                        break;
                    }
                }
                ws.get_ref().set_nonblocking(true).ok();
                if failed {
                    break;
                }
                continue;
            }

            // Reference image changed (project switch — always send immediately)
            if s.ref_image_generation != last_ref_image_gen {
                last_ref_image_gen = s.ref_image_generation;
                last_gen = s.generation;
                last_image_gen = s.image_generation;
                last_image_send_time = Instant::now();
                let msg = serde_json::json!({
                    "type": "project_switched",
                    "referenceImage": BASE64.encode(&s.ref_png),
                    "image": BASE64.encode(&s.best_png),
                    "project": s.active_project,
                    "fitness": s.fitness,
                    "polygons": s.polygons,
                    "improvements": s.improvements,
                    "evalsPerSec": s.evals_per_sec,
                    "totalEvals": s.total_evals,
                    "elapsed": s.elapsed_secs,
                    "paused": s.paused,
                    "drawingJson": s.drawing_json,
                    "imageWidth": s.image_width,
                    "imageHeight": s.image_height,
                    "mutationParams": s.mutation_params,
                    "targetResolution": s.target_resolution,
                });
                msg.to_string()
            } else if s.generation == last_gen {
                // Nothing changed — skip send
                continue;
            } else {
                // Stats or image update
                // Images are large (~100KB base64), throttle separately to ~5fps
                let has_new_image = s.image_generation > last_image_gen
                    && last_image_send_time.elapsed() >= image_interval;
                last_gen = s.generation;

                let gpu_stats_json = s.gpu_stats.as_ref().map(|g| g.to_json());
                let msg = if has_new_image {
                    last_image_gen = s.image_generation;
                    last_image_send_time = Instant::now();
                    serde_json::json!({
                        "type": "update",
                        "image": BASE64.encode(&s.best_png),
                        "fitness": s.fitness,
                        "polygons": s.polygons,
                        "improvements": s.improvements,
                        "evalsPerSec": s.evals_per_sec,
                        "totalEvals": s.total_evals,
                        "elapsed": s.elapsed_secs,
                        "paused": s.paused,
                        "drawingJson": s.drawing_json,
                        "gpuStats": gpu_stats_json,
                    })
                } else {
                    serde_json::json!({
                        "type": "stats",
                        "fitness": s.fitness,
                        "polygons": s.polygons,
                        "improvements": s.improvements,
                        "evalsPerSec": s.evals_per_sec,
                        "totalEvals": s.total_evals,
                        "elapsed": s.elapsed_secs,
                        "paused": s.paused,
                        "gpuStats": gpu_stats_json,
                    })
                };
                msg.to_string()
            }
        };

        // Send in blocking mode to ensure delivery
        ws.get_ref().set_nonblocking(false).ok();
        let result = ws.send(tungstenite::Message::Text(msg_string.into()));
        ws.get_ref().set_nonblocking(true).ok();
        if result.is_err() {
            break;
        }
    }

    println!("[WS] Client disconnected: {:?}", peer);
}

/// Handle a WS command from a client. Returns an optional reply to send back to the client.
fn handle_ws_command(
    cmd: &serde_json::Value,
    state: &SharedWsState,
    peer: &Option<std::net::SocketAddr>,
) -> Option<serde_json::Value> {
    let (lock, cvar) = &**state;
    match cmd["type"].as_str() {
        Some("pause") | Some("resume") => {
            let pause = cmd["type"].as_str() == Some("pause");
            let mut s = lock.lock().unwrap();
            s.paused = pause;
            s.generation += 1;
            cvar.notify_all();
            println!(
                "[WS] {} by {:?}",
                if pause { "Paused" } else { "Resumed" },
                peer
            );
            None
        }
        Some("create_project") => {
            let name = cmd["name"].as_str().unwrap_or("");
            let ref_image_b64 = cmd["referenceImage"].as_str().unwrap_or("");
            let image_bytes = match BASE64.decode(ref_image_b64) {
                Ok(b) => b,
                Err(e) => {
                    return Some(serde_json::json!({
                        "type": "project_error",
                        "error": format!("Invalid base64: {}", e),
                    }));
                }
            };
            match projects::create_project(name, &image_bytes) {
                Ok(_info) => {
                    println!("[WS] Project '{}' created by {:?}", name, peer);
                    let mut s = lock.lock().unwrap();
                    s.project_list_generation += 1;
                    // Auto-switch if no active project
                    if s.active_project.is_none() {
                        s.switch_request = Some(name.to_string());
                    }
                    s.generation += 1;
                    cvar.notify_all();
                    None
                }
                Err(e) => Some(serde_json::json!({
                    "type": "project_error",
                    "error": e,
                })),
            }
        }
        Some("delete_project") => {
            let name = cmd["name"].as_str().unwrap_or("");
            let mut s = lock.lock().unwrap();
            let is_active = s.active_project.as_deref() == Some(name);

            if is_active {
                // Active project: defer deletion until the inner loop breaks.
                // Set switch_request so the loop exits, then delete from the outer loop.
                let remaining: Vec<_> = projects::list_projects()
                    .into_iter()
                    .filter(|p| p.name != name)
                    .collect();
                s.switch_request = remaining.first().map(|p| p.name.clone());
                // Store name to delete after inner loop breaks
                s.pending_delete = Some(name.to_string());
                if s.switch_request.is_none() {
                    s.active_project = None;
                }
                s.project_list_generation += 1;
                s.generation += 1;
                cvar.notify_all();
                println!("[WS] Project '{}' (active) delete requested by {:?}", name, peer);
                None
            } else {
                drop(s); // release lock before I/O
                match projects::delete_project(name) {
                    Ok(()) => {
                        println!("[WS] Project '{}' deleted by {:?}", name, peer);
                        let mut s = lock.lock().unwrap();
                        s.project_list_generation += 1;
                        s.generation += 1;
                        cvar.notify_all();
                        None
                    }
                    Err(e) => Some(serde_json::json!({
                        "type": "project_error",
                        "error": e,
                    })),
                }
            }
        }
        Some("reset_project") => {
            let name = cmd["name"].as_str().unwrap_or("");
            // Validate project exists
            let dir = std::path::Path::new("projects").join(name);
            if !dir.exists() {
                return Some(serde_json::json!({
                    "type": "project_error",
                    "error": format!("Project '{}' does not exist", name),
                }));
            }
            println!("[WS] Project '{}' reset requested by {:?}", name, peer);
            let mut s = lock.lock().unwrap();
            s.project_list_generation += 1;
            // If active, trigger reload and auto-pause
            // File deletion happens in the evolution loop (race-free)
            if s.active_project.as_deref() == Some(name) {
                s.switch_request = Some(name.to_string());
                s.reset_active = true;
                s.paused = true;
            } else {
                // Not active — safe to delete files directly
                projects::reset_project(name).ok();
            }
            s.generation += 1;
            cvar.notify_all();
            None
        }
        Some("switch_project") => {
            let name = cmd["name"].as_str().unwrap_or("");
            println!("[WS] Switch to project '{}' requested by {:?}", name, peer);
            let mut s = lock.lock().unwrap();
            s.switch_request = Some(name.to_string());
            s.generation += 1;
            cvar.notify_all();
            None
        }
        Some("rename_project") => {
            let old_name = cmd["oldName"].as_str().unwrap_or("");
            let new_name = cmd["newName"].as_str().unwrap_or("");
            match projects::rename_project(old_name, new_name) {
                Ok(()) => {
                    println!("[WS] Project '{}' renamed to '{}' by {:?}", old_name, new_name, peer);
                    let mut s = lock.lock().unwrap();
                    if s.active_project.as_deref() == Some(old_name) {
                        s.active_project = Some(new_name.to_string());
                    }
                    s.project_list_generation += 1;
                    s.generation += 1;
                    cvar.notify_all();
                    None
                }
                Err(e) => Some(serde_json::json!({
                    "type": "project_error",
                    "error": e,
                })),
            }
        }
        Some("update_params") => {
            if let Ok(mut params) = serde_json::from_value::<MutationParams>(cmd["params"].clone()) {
                params.sanitize();
                let mut s = lock.lock().unwrap();
                s.mutation_params = params;
                println!("[WS] Mutation params updated by {:?}", peer);
            }
            None
        }
        Some("reset_params") => {
            let mut s = lock.lock().unwrap();
            s.mutation_params = MutationParams::default();
            println!("[WS] Mutation params reset by {:?}", peer);
            None
        }
        Some("start_benchmark") => {
            if let Ok(req) = serde_json::from_value::<BenchmarkRequest>(cmd.clone()) {
                let mut s = lock.lock().unwrap();
                // If benchmark specifies a different resolution, trigger project reload first
                if req.resolution > 0 && req.resolution != s.target_resolution {
                    println!("[WS] Benchmark needs resolution {} (current {}), triggering reload",
                        req.resolution, s.target_resolution);
                    s.target_resolution = req.resolution;
                    if let Some(name) = s.active_project.clone() {
                        s.switch_request = Some(name);
                    }
                }
                s.benchmark_request = Some(req);
                s.generation += 1;
                cvar.notify_all();
                println!("[WS] Benchmark requested by {:?}", peer);
            }
            None
        }
        Some("run_standard_benchmark") => {
            // Fixed-settings benchmark from blank start — deterministic baseline
            let blank_drawing = Drawing { polygons: vec![], is_dirty: false, fitness: 0.0 };
            let drawing_json = serde_json::to_string(&blank_drawing).unwrap();
            let mut params = MutationParams {
                chain_count: 4,
                lambda: 64,
                single_mutation_mode: true,
                adaptive_mutation: true,
                ..Default::default()
            };
            params.sanitize();
            let mut s = lock.lock().unwrap();
            let req = BenchmarkRequest {
                drawing_json,
                params,
                duration_secs: 33,
                label: "standard-bench".to_string(),
                resolution: s.target_resolution,
            };
            s.benchmark_request = Some(req);
            s.generation += 1;
            cvar.notify_all();
            println!("[WS] Standard benchmark requested by {:?}", peer);
            None
        }
        Some("clear_benchmarks") => {
            let mut s = lock.lock().unwrap();
            s.benchmark_results.clear();
            s.generation += 1;
            cvar.notify_all();
            println!("[WS] Benchmarks cleared by {:?}", peer);
            None
        }
        Some("update_resolution") => {
            let resolution = cmd["resolution"].as_u64().unwrap_or(0) as u32;
            if resolution != 0 && !(64..=1024).contains(&resolution) {
                return Some(serde_json::json!({
                    "type": "project_error",
                    "error": "Resolution must be between 64 and 1024",
                }));
            }
            let mut s = lock.lock().unwrap();
            s.target_resolution = resolution;
            // Trigger self-switch to reload at new resolution
            if let Some(name) = s.active_project.clone() {
                s.switch_request = Some(name);
            }
            s.generation += 1;
            cvar.notify_all();
            println!("[WS] Resolution set to {} by {:?}", resolution, peer);
            None
        }
        Some("import_drawing") => {
            let name = cmd["name"].as_str().unwrap_or("");
            let drawing_json = cmd["drawingJson"].as_str().unwrap_or("");
            match projects::import_drawing(name, drawing_json.as_bytes()) {
                Ok(()) => {
                    println!("[WS] Drawing imported into '{}' by {:?}", name, peer);
                    let mut s = lock.lock().unwrap();
                    s.project_list_generation += 1;
                    // If active, trigger reload
                    if s.active_project.as_deref() == Some(name) {
                        s.switch_request = Some(name.to_string());
                    }
                    s.generation += 1;
                    cvar.notify_all();
                    None
                }
                Err(e) => Some(serde_json::json!({
                    "type": "project_error",
                    "error": e,
                })),
            }
        }
        _ => None,
    }
}

fn build_gpu_stats(evolver: &GpuEvolver) -> GpuStatsWs {
    let timings = evolver.pass_timings();
    let ws_timings = if timings.sample_count > 0 {
        let n = timings.sample_count as f64;
        let mutate = timings.mutate_ns / n / 1_000_000.0;
        let rasterize_error = timings.rasterize_error_ns / n / 1_000_000.0;
        let select = timings.select_ns / n / 1_000_000.0;
        let total = mutate + rasterize_error + select;
        if total > 0.0 {
            Some(GpuPassTimingsWs {
                mutate_ms: mutate as f32,
                mutate_pct: (mutate / total * 100.0) as f32,
                rasterize_error_ms: rasterize_error as f32,
                rasterize_error_pct: (rasterize_error / total * 100.0) as f32,
                select_ms: select as f32,
                select_pct: (select / total * 100.0) as f32,
                total_ms: total as f32,
            })
        } else {
            None
        }
    } else {
        None
    };

    let raw_fitness = evolver.chain_fitness();
    let chain_count = evolver.chain_count();

    let mut fitness: Vec<f32> = raw_fitness.to_vec();
    fitness.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    GpuStatsWs {
        chain_count,
        memory_mb: evolver.estimated_memory_bytes() as f32 / (1024.0 * 1024.0),
        timings: ws_timings,
        chain_fitness: fitness,
        rasterize_wg: evolver.rasterize_wg(),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_benchmark(
    ws_state: &SharedWsState,
    evolver: &mut GpuEvolver,
    req: &BenchmarkRequest,
    global_best: &mut Drawing,
    improvements: &mut u64,
    w: usize,
    h: usize,
    render_buf: &mut [u8],
) {
    // Parse snapshot drawing
    let drawing: Drawing = match serde_json::from_str(&req.drawing_json) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[Benchmark] Failed to parse drawing: {}", e);
            return;
        }
    };

    let mut bench_params = req.params.clone();
    bench_params.sanitize();

    // Signal benchmark started
    {
        let (lock, cvar) = &**ws_state;
        let mut s = lock.lock().unwrap();
        s.benchmark_active = true;
        s.benchmark_events.push(serde_json::json!({
            "type": "benchmark_started",
            "label": req.label,
        }));
        s.generation += 1;
        cvar.notify_all();
    }
    println!("[Benchmark] Started: '{}' ({}s, {} chains)",
        req.label, req.duration_secs, bench_params.chain_count);

    // Prepare: reinit chains, recreate pipeline if needed, run 2 warmup batches,
    // then reset counters. This ensures pipeline compilation and warmup time
    // are NOT counted in the benchmark duration.
    let gpu_start_fitness = evolver.prepare_for_benchmark(&drawing, &bench_params);
    let start_fitness = gpu_start_fitness;

    let bench_start = Instant::now();
    let duration = Duration::from_secs(req.duration_secs as u64);
    let mut samples: Vec<BenchmarkSample> = vec![];
    let mut bench_improvements = 0u64;
    let mut last_sample = Instant::now();
    // Use GPU-evaluated fitness as the threshold — CPU and GPU compute different
    // fitness values (L2 vs L1, different float accumulation per workgroup size),
    // so using CPU fitness here would cause improvements to be undercounted or zero.
    let mut bench_best = drawing.clone();
    bench_best.fitness = gpu_start_fitness;
    let sample_interval = Duration::from_secs(1);
    let image_render_interval = Duration::from_millis(200);
    let mut last_image_render = Instant::now();

    loop {
        if bench_start.elapsed() >= duration {
            break;
        }

        // Check for abort (switch_request)
        if ws_state.0.lock().unwrap().switch_request.is_some() {
            println!("[Benchmark] Aborted due to switch request");
            break;
        }

        // Run evolution batch with benchmark params
        if let Some(new_best) = evolver.run_batch(&bench_params, true) {
            if new_best.fitness > bench_best.fitness {
                bench_improvements += 1;
                bench_best = new_best;
            }
        }

        // Recapture elapsed AFTER the batch completes so evals/s is accurate
        let elapsed = bench_start.elapsed();

        // Collect sample every second
        if last_sample.elapsed() >= sample_interval {
            let elapsed_secs = elapsed.as_secs_f32();
            let chain_fitness = evolver.chain_fitness();
            let best_f = chain_fitness.iter().cloned().fold(0.0f32, f32::max);
            let worst_f = chain_fitness.iter().cloned().fold(f32::MAX, f32::min);
            let avg_f = chain_fitness.iter().sum::<f32>() / chain_fitness.len().max(1) as f32;
            let total_evals = evolver.total_evaluations();
            let evals_per_sec = if elapsed_secs > 0.0 { total_evals as f64 / elapsed_secs as f64 } else { 0.0 };

            samples.push(BenchmarkSample {
                elapsed_secs,
                best_fitness: best_f,
                avg_fitness: avg_f,
                worst_fitness: worst_f,
                improvements: bench_improvements,
                total_evals,
                evals_per_sec,
            });

            // Send progress via WS state
            {
                let (lock, cvar) = &**ws_state;
                let mut s = lock.lock().unwrap();
                s.fitness = bench_best.fitness;
                s.polygons = bench_best.polygons.len();
                s.improvements = *improvements + bench_improvements;
                s.evals_per_sec = evals_per_sec;
                s.total_evals = total_evals;
                s.elapsed_secs = elapsed_secs as u64;
                s.benchmark_events.push(serde_json::json!({
                    "type": "benchmark_progress",
                    "label": req.label,
                    "elapsedSecs": elapsed_secs,
                    "durationSecs": req.duration_secs,
                    "bestFitness": best_f,
                    "improvements": bench_improvements,
                }));
                s.generation += 1;
                cvar.notify_all();
            }

            last_sample = Instant::now();
        }

        // Render image periodically so clients see visual progress
        if bench_best.fitness > global_best.fitness && last_image_render.elapsed() >= image_render_interval {
            bench_best.draw(render_buf, w, h, Rasterizer::HalfSpace);
            let png = encode_rgba_as_png(render_buf, w, h);
            let (lock, cvar) = &**ws_state;
            let mut s = lock.lock().unwrap();
            s.best_png = png;
            s.drawing_json = serde_json::to_string(&bench_best).unwrap();
            s.image_generation += 1;
            cvar.notify_all();
            last_image_render = Instant::now();
        }
    }

    // Build result
    let total_elapsed = bench_start.elapsed();
    let total_evals = evolver.total_evaluations();
    let result = BenchmarkResult {
        label: req.label.clone(),
        start_fitness,
        final_fitness: bench_best.fitness,
        total_improvements: bench_improvements,
        total_evals,
        duration_secs: req.duration_secs,
        actual_duration_secs: total_elapsed.as_secs_f32(),
        improvements_per_sec: if total_elapsed.as_secs_f64() > 0.0 {
            bench_improvements as f64 / total_elapsed.as_secs_f64()
        } else {
            0.0
        },
        samples,
        chain_count: bench_params.chain_count,
        lambda: bench_params.lambda,
        gpu_batch_iters: bench_params.gpu_batch_iters,
        resolution: if w >= h { w as u32 } else { h as u32 },
    };

    println!(
        "[Benchmark] Complete: '{}' | {:.4} -> {:.4} | {} improvements | {} evals",
        result.label, result.start_fitness, result.final_fitness,
        result.total_improvements, result.total_evals,
    );

    // Update global best if benchmark found something better
    if bench_best.fitness > global_best.fitness {
        *improvements += bench_improvements;
        *global_best = bench_best;
    }

    // Store result and signal completion
    {
        let (lock, cvar) = &**ws_state;
        let mut s = lock.lock().unwrap();
        s.benchmark_events.push(serde_json::json!({
            "type": "benchmark_complete",
            "result": serde_json::to_value(&result).unwrap(),
        }));
        s.benchmark_results.push(result);
        s.benchmark_active = false;
        s.generation += 1;
        cvar.notify_all();
    }
}

/// Standalone CLI benchmark — no WS server, no project system.
/// Loads the reference image from a project directory, starts from a random drawing,
/// evolves for 33 seconds with fixed settings, and prints results.
///
/// Usage: cargo run --release -- --bench [project_name]
///   project_name defaults to "ff"
fn run_cli_benchmark(project: &str, gpu_batch_iters_override: Option<u32>) {
    const DURATION_SECS: u64 = 33;
    const BENCH_CHAINS: u32 = 4;
    const BENCH_LAMBDA: u32 = 64;
    const BENCH_RESOLUTION: u32 = 256;

    println!("=== GPU CLI Benchmark ===");
    println!("Project: {}, Duration: {}s, Chains: {}, Lambda: {}, Resolution: {}, BatchIters: {}",
        project, DURATION_SECS, BENCH_CHAINS, BENCH_LAMBDA, BENCH_RESOLUTION,
        gpu_batch_iters_override.unwrap_or(artgen_backend_rust::settings::GPU_DEFAULT_BATCH_ITERS));

    // Load reference image from project directory
    projects::ensure_projects_dir();
    let original_path = projects::project_original_path(project);
    let ref_path = projects::project_reference_path(project);
    let image_path = if original_path.exists() { original_path } else { ref_path };
    if !image_path.exists() {
        eprintln!("Error: Reference image not found for project '{}' at {:?}", project, image_path);
        eprintln!("Make sure the project exists in projects/{}/", project);
        exit(1);
    }

    let ref_image_bytes = std::fs::read(&image_path).expect("Failed to read reference image");
    let (rgba, w, h) = projects::load_and_normalize_image(&ref_image_bytes, Some(BENCH_RESOLUTION))
        .expect("Failed to normalize image");
    println!("Image: {}x{}", w, h);

    // Start from blank — deterministic baseline, no random variance
    let initial = Drawing { polygons: vec![], is_dirty: false, fitness: 0.0 };

    // Initialize GPU evolver
    let mut evolver = futures_lite::future::block_on(GpuEvolver::new(
        &rgba,
        w as u32,
        h as u32,
        &initial,
    ));

    // Configure benchmark params: 4 chains, 64 lambda, single mutation, adaptive scale
    let mut params = MutationParams {
        chain_count: BENCH_CHAINS,
        lambda: BENCH_LAMBDA,
        single_mutation_mode: true,
        adaptive_mutation: true,
        ..Default::default()
    };
    if let Some(iters) = gpu_batch_iters_override {
        params.gpu_batch_iters = iters;
    }
    params.sanitize();

    let evals_per_batch = params.gpu_batch_iters as u64
        * params.chain_count as u64
        * params.lambda as u64;

    // Warm up: pipeline setup + 2 batches to fill double-buffer
    println!("Warming up...");
    let gpu_start_fitness = evolver.prepare_for_benchmark(&initial, &params);
    println!("GPU start fitness: {:.4}%", gpu_start_fitness * 100.0);

    // Run benchmark
    println!("Running evolution for {}s...\n", DURATION_SECS);
    let bench_start = Instant::now();
    let duration = Duration::from_secs(DURATION_SECS);
    let mut improvements = 0u64;
    let mut best = initial;
    best.fitness = gpu_start_fitness;
    let mut batches = 0u64;
    let mut last_print = Instant::now();

    loop {
        if bench_start.elapsed() >= duration {
            break;
        }

        if let Some(new_best) = evolver.run_batch(&params, false) {
            if new_best.fitness > best.fitness {
                improvements += 1;
                best = new_best;
            }
        }
        batches += 1;

        // Recapture elapsed AFTER the batch so evals/s is accurate
        let elapsed = bench_start.elapsed();

        // Print progress every 5 seconds
        if last_print.elapsed() >= Duration::from_secs(5) {
            let secs = elapsed.as_secs_f64();
            let total_evals = evolver.total_evaluations();
            let evals_per_sec = if secs > 0.0 { total_evals as f64 / secs } else { 0.0 };
            println!("  {:>3.0}s  {:.4}%  {:>8} evals/s  {:>4} improvements",
                secs, best.fitness * 100.0, evals_per_sec as u64, improvements);
            last_print = Instant::now();
        }
    }

    let total_elapsed = bench_start.elapsed();
    let total_secs = total_elapsed.as_secs_f64();
    let total_evals = evolver.total_evaluations();
    let evals_per_sec = if total_secs > 0.0 { total_evals as f64 / total_secs } else { 0.0 };
    let improvements_per_sec = if total_secs > 0.0 { improvements as f64 / total_secs } else { 0.0 };

    println!("\n=== Results ===");
    println!("  Final fitness:    {:.4}%", best.fitness * 100.0);
    println!("  Evals/sec:        {:.0}", evals_per_sec);
    println!("  Total evals:      {}", total_evals);
    println!("  Improvements:     {}", improvements);
    println!("  Improvements/sec: {:.1}", improvements_per_sec);
    println!("  Batches:          {}", batches);
    println!("  Elapsed:          {:.1}s", total_secs);
    println!("  Polygons:         {}", best.polygons.len());
    println!("  Evals/batch:      {}", evals_per_batch);

    // Print JSON for machine-readable output
    let result = serde_json::json!({
        "finalFitness": best.fitness,
        "evalsPerSec": evals_per_sec as u64,
        "totalEvals": total_evals,
        "improvements": improvements,
        "improvementsPerSec": improvements_per_sec,
        "batches": batches,
        "elapsedSecs": total_secs,
        "polygons": best.polygons.len(),
        "chains": BENCH_CHAINS,
        "lambda": BENCH_LAMBDA,
        "resolution": format!("{}x{}", w, h),
    });
    println!("\nJSON: {}", serde_json::to_string(&result).unwrap());
}

fn gpu_main_loop_headless(legacy_image: Option<&str>, gpu_batch_iters_override: Option<u32>) {
    projects::ensure_projects_dir();

    // Legacy migration
    let migrated_project = projects::migrate_legacy(legacy_image);

    // Apply CLI overrides to default mutation params
    let mut initial_params = MutationParams::default();
    if let Some(iters) = gpu_batch_iters_override {
        initial_params.gpu_batch_iters = iters;
    }
    initial_params.sanitize();

    // Create shared state for WS server
    let ws_state: SharedWsState = Arc::new((
        Mutex::new(WsState {
            ref_png: vec![],
            best_png: vec![],
            fitness: 0.0,
            polygons: 0,
            improvements: 0,
            evals_per_sec: 0.0,
            total_evals: 0,
            elapsed_secs: 0,
            generation: 0,
            image_generation: 0,
            ref_image_generation: 0,
            paused: true,
            drawing_json: String::new(),
            image_width: 0,
            image_height: 0,
            active_project: None,
            project_list_generation: 0,
            switch_request: migrated_project,
            reset_active: false,
            pending_delete: None,
            target_resolution: 256,
            mutation_params: initial_params,
            gpu_stats: None,
            benchmark_request: None,
            benchmark_results: vec![],
            benchmark_active: false,
            benchmark_events: vec![],
        }),
        Condvar::new(),
    ));

    // Spawn WS server thread
    let ws_clone = ws_state.clone();
    thread::spawn(move || ws_server(ws_clone));

    // Outer loop: manage project switching
    loop {
        // Handle deferred project deletion (after inner loop has stopped)
        {
            let mut s = ws_state.0.lock().unwrap();
            if let Some(name) = s.pending_delete.take() {
                drop(s); // release lock before I/O
                if let Err(e) = projects::delete_project(&name) {
                    eprintln!("[Projects] Failed to delete '{}': {}", name, e);
                }
                println!("[Projects] Deleted project '{}'", name);
                let mut s = ws_state.0.lock().unwrap();
                s.project_list_generation += 1;
                s.generation += 1;
            }
        }

        // Determine which project to activate
        let project_name = loop {
            // Handle deferred deletes while idle too
            {
                let mut s = ws_state.0.lock().unwrap();
                if let Some(name) = s.pending_delete.take() {
                    drop(s);
                    if let Err(e) = projects::delete_project(&name) {
                        eprintln!("[Projects] Failed to delete '{}': {}", name, e);
                    }
                    println!("[Projects] Deleted project '{}'", name);
                    let mut s = ws_state.0.lock().unwrap();
                    s.project_list_generation += 1;
                    s.generation += 1;
                }
            }
            let switch = ws_state.0.lock().unwrap().switch_request.take();
            if let Some(name) = switch {
                break name;
            }
            // No switch request — idle
            thread::sleep(Duration::from_millis(100));
        };

        // Load project reference image — prefer original (full quality) for resolution changes
        let original_path = projects::project_original_path(&project_name);
        let ref_path = projects::project_reference_path(&project_name);
        let image_path = if original_path.exists() { &original_path } else { &ref_path };
        if !image_path.exists() {
            eprintln!("[Projects] Reference image not found for '{}', skipping", project_name);
            continue;
        }

        let ref_image_bytes = match std::fs::read(image_path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("[Projects] Failed to read reference for '{}': {}", project_name, e);
                continue;
            }
        };

        let target_res = ws_state.0.lock().unwrap().target_resolution;
        let max_dim = if target_res > 0 { Some(target_res) } else { None };
        let (rgba, w, h) = match projects::load_and_normalize_image(&ref_image_bytes, max_dim) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[Projects] Failed to normalize image for '{}': {}", project_name, e);
                continue;
            }
        };

        let engine = initialize_engine_from_rgba(&rgba, w, h);

        // Load best drawing or start fresh, respecting current evolution params
        let best_path = projects::project_best_json_path(&project_name);
        let mp = ws_state.0.lock().unwrap().mutation_params.clone();
        let max_poly = mp.max_polygons as usize;
        let mut initial_best = if best_path.exists() {
            let best_str = best_path.to_str().unwrap_or("");
            println!("[Projects] Loading existing best for '{}'", project_name);
            Drawing::from_file(best_str)
        } else {
            println!("[Projects] Starting fresh for '{}'", project_name);
            Drawing::new_random_capped(max_poly)
        };
        // Truncate loaded drawings that exceed current max_polygons
        if initial_best.polygons.len() > max_poly {
            initial_best.polygons.truncate(max_poly);
        }

        let json_filename = best_path.to_string_lossy().to_string();
        let png_path = projects::project_best_png_path(&project_name).to_string_lossy().to_string();

        // Initialize GPU evolver
        let mut evolver = futures_lite::future::block_on(GpuEvolver::new(
            &engine.ref_image_data,
            engine.w as u32,
            engine.h as u32,
            &initial_best,
        ));

        // Evaluate initial fitness on GPU so it matches GPU's error metric
        // (important after resolution changes where CPU fitness is stale)
        let gpu_fitness = evolver.evaluate_chain_fitness(mp.chain_count);
        initial_best.fitness = gpu_fitness;

        let mut render_buf = vec![0u8; w * h * 4];

        // Encode reference image as PNG for WS clients
        let ref_png = encode_rgba_as_png(&engine.ref_image_data, w, h);

        // Encode initial best as PNG
        initial_best.draw(&mut render_buf, w, h, Rasterizer::HalfSpace);
        let initial_png = encode_rgba_as_png(&render_buf, w, h);
        if let Err(e) = std::fs::write(&png_path, &initial_png) {
            eprintln!("Failed to save PNG: {}", e);
        }

        // Update WS state with new project info
        {
            let (lock, cvar) = &*ws_state;
            let mut s = lock.lock().unwrap();
            s.ref_png = ref_png;
            s.best_png = initial_png;
            s.fitness = initial_best.fitness;
            s.polygons = initial_best.polygons.len();
            s.improvements = 0;
            s.evals_per_sec = 0.0;
            s.total_evals = 0;
            s.elapsed_secs = 0;
            s.drawing_json = serde_json::to_string(&initial_best).unwrap_or_default();
            s.image_width = w as u32;
            s.image_height = h as u32;
            s.active_project = Some(project_name.clone());
            s.paused = true;
            s.generation += 1;
            s.image_generation += 1;
            s.ref_image_generation += 1;
            s.project_list_generation += 1;
            cvar.notify_all();
        }

        println!("[Projects] Active project: '{}'", project_name);

        let mut global_best = initial_best;
        let mut last_save_timestamp = Instant::now();
        let mut last_stats_timestamp = Instant::now();
        let mut last_image_render = Instant::now();
        let image_render_interval = Duration::from_millis(200); // render at ~5fps
        let mut improvements = 0u64;
        let mut batches = 0u64;
        let mut image_dirty = false; // true when global_best changed but not yet rendered
        let mut paused_duration = Duration::ZERO;
        let mut pause_start: Option<Instant> = Some(Instant::now()); // starts paused

        // Inner loop: evolution for current project
        loop {
            // Check for switch request
            {
                let mut s = ws_state.0.lock().unwrap();
                if s.switch_request.is_some() {
                    let deleting = s.pending_delete.is_some();
                    if s.reset_active {
                        // Reset: delete best files now (race-free since we hold the lock
                        // and are about to break — no more periodic saves can happen)
                        s.reset_active = false;
                        std::fs::remove_file(&json_filename).ok();
                        std::fs::remove_file(&png_path).ok();
                        println!("[Projects] Resetting '{}'", project_name);
                    } else if deleting {
                        // About to delete this project — don't save
                        println!("[Projects] Stopping '{}' (pending delete)", project_name);
                    } else {
                        // Normal switch: save current state
                        drop(s); // release lock before I/O
                        global_best.to_file(&json_filename);
                        global_best.draw(&mut render_buf, w, h, Rasterizer::HalfSpace);
                        let png = encode_rgba_as_png(&render_buf, w, h);
                        std::fs::write(&png_path, &png).ok();
                        println!("[Projects] Saving and switching from '{}'", project_name);
                    }
                    break; // break inner loop, outer loop will handle the switch
                }
                // Check if active project was cleared (no projects left)
                if s.active_project.is_none() {
                    if s.pending_delete.is_none() {
                        // Only save if we're not about to delete
                        drop(s);
                        global_best.to_file(&json_filename);
                    }
                    println!("[Projects] Going idle");
                    break;
                }
            }

            // Check for benchmark request (runs even when paused)
            let bench_req = ws_state.0.lock().unwrap().benchmark_request.take();
            if let Some(req) = bench_req {
                // Accumulate paused time before benchmark
                if let Some(start) = pause_start.take() {
                    paused_duration += start.elapsed();
                }
                run_benchmark(&ws_state, &mut evolver, &req, &mut global_best, &mut improvements, w, h, &mut render_buf);
                // Benchmark resets evolver.start_time, so paused_duration must reset too
                // to avoid underflow in `evolver.elapsed() - paused_duration`
                paused_duration = Duration::ZERO;
                // After benchmark, stay paused if we were paused
                if ws_state.0.lock().unwrap().paused {
                    pause_start = Some(Instant::now());
                }
            }

            // Check if paused — sleep and skip evolution, stats are frozen
            if ws_state.0.lock().unwrap().paused {
                if pause_start.is_none() {
                    pause_start = Some(Instant::now());
                    // Flush any pending image render before freezing
                    if image_dirty {
                        image_dirty = false;
                        global_best.draw(&mut render_buf, w, h, Rasterizer::HalfSpace);
                        let png = encode_rgba_as_png(&render_buf, w, h);
                        let (lock, cvar) = &*ws_state;
                        let mut s = lock.lock().unwrap();
                        s.best_png = png;
                        s.drawing_json = serde_json::to_string(&global_best).unwrap();
                        s.image_generation += 1;
                        cvar.notify_all();
                    }
                    // Push final stats snapshot so client has accurate frozen values
                    let active_elapsed = evolver.elapsed() - paused_duration;
                    let active_secs = active_elapsed.as_secs_f64();
                    let evals = evolver.total_evaluations();
                    let evals_per_sec = if active_secs > 0.0 { evals as f64 / active_secs } else { 0.0 };
                    let (lock, cvar) = &*ws_state;
                    let mut s = lock.lock().unwrap();
                    s.evals_per_sec = evals_per_sec;
                    s.total_evals = evals;
                    s.elapsed_secs = active_elapsed.as_secs();
                    s.fitness = global_best.fitness;
                    s.polygons = global_best.polygons.len();
                    s.improvements = improvements;
                    s.generation += 1;
                    cvar.notify_all();
                }
                thread::sleep(Duration::from_millis(50));
                continue;
            }

            // Accumulate paused time when resuming
            if let Some(start) = pause_start.take() {
                paused_duration += start.elapsed();
            }

            batches += 1;
            let mp = ws_state.0.lock().unwrap().mutation_params.clone();
            if let Some(new_best) = evolver.run_batch(&mp, true) {
                if new_best.fitness > global_best.fitness {
                    improvements += 1;
                    let delta = new_best.fitness - global_best.fitness;
                    println!(
                        "[GPU] improvement #{}: {:.4} -> {:.4} (+{:.6}) | polygons: {} | batch: {}",
                        improvements,
                        global_best.fitness,
                        new_best.fitness,
                        delta,
                        new_best.polygons.len(),
                        batches,
                    );
                    global_best = new_best;
                    image_dirty = true;
                }
            }

            // Render image + update WS at ~5fps (avoids wasting CPU on PNGs that won't be sent)
            if image_dirty && last_image_render.elapsed() >= image_render_interval {
                image_dirty = false;
                last_image_render = Instant::now();

                global_best.draw(&mut render_buf, w, h, Rasterizer::HalfSpace);
                let png = encode_rgba_as_png(&render_buf, w, h);

                {
                    let active_elapsed = evolver.elapsed() - paused_duration;
                    let active_secs = active_elapsed.as_secs_f64();
                    let evals = evolver.total_evaluations();
                    let evals_per_sec = if active_secs > 0.0 { evals as f64 / active_secs } else { 0.0 };
                    let (lock, cvar) = &*ws_state;
                    let mut s = lock.lock().unwrap();
                    s.best_png = png.clone();
                    s.fitness = global_best.fitness;
                    s.polygons = global_best.polygons.len();
                    s.improvements = improvements;
                    s.evals_per_sec = evals_per_sec;
                    s.total_evals = evals;
                    s.elapsed_secs = active_elapsed.as_secs();
                    s.drawing_json = serde_json::to_string(&global_best).unwrap();
                    s.generation += 1;
                    s.image_generation += 1;
                    cvar.notify_all();
                }

                // Save to disk periodically
                if last_save_timestamp.elapsed().as_secs() >= 10 {
                    global_best.to_file(&json_filename);
                    if let Err(e) = std::fs::write(&png_path, &png) {
                        eprintln!("Failed to save PNG: {}", e);
                    } else {
                        println!("[GPU] Saved preview: {}", png_path);
                    }
                    last_save_timestamp = Instant::now();
                }
            }

            // Push lightweight stats periodically (no image)
            if last_stats_timestamp.elapsed().as_secs() >= 2 {
                let active_elapsed = evolver.elapsed() - paused_duration;
                let active_secs = active_elapsed.as_secs_f64();
                let evals = evolver.total_evaluations();
                let evals_per_sec = if active_secs > 0.0 { evals as f64 / active_secs } else { 0.0 };
                let gpu_stats = build_gpu_stats(&evolver);
                {
                    let (lock, cvar) = &*ws_state;
                    let mut s = lock.lock().unwrap();
                    s.evals_per_sec = evals_per_sec;
                    s.total_evals = evals;
                    s.elapsed_secs = active_elapsed.as_secs();
                    s.fitness = global_best.fitness;
                    s.polygons = global_best.polygons.len();
                    s.improvements = improvements;
                    s.gpu_stats = Some(gpu_stats);
                    s.generation += 1;
                    cvar.notify_all();
                }

                print_gpu_stats_active(&evolver, &global_best, active_elapsed);
                evolver.pass_timings().print_averages();
                evolver.reset_pass_timings();
                last_stats_timestamp = Instant::now();
            }
        }
    }
}

fn print_gpu_stats(evolver: &GpuEvolver, best: &Drawing) {
    let evals = evolver.total_evaluations();
    let evals_per_sec = evolver.evals_per_sec();
    let elapsed_secs = evolver.elapsed().as_secs();
    println!(
        "[GPU] fitness: {:.4} | polygons: {} | evals: {} | evals/s: {:.0} | elapsed: {}s",
        best.fitness,
        best.polygons.len(),
        evals,
        evals_per_sec,
        elapsed_secs,
    );
}

fn print_gpu_stats_active(evolver: &GpuEvolver, best: &Drawing, active_elapsed: Duration) {
    let evals = evolver.total_evaluations();
    let active_secs = active_elapsed.as_secs_f64();
    let evals_per_sec = if active_secs > 0.0 { evals as f64 / active_secs } else { 0.0 };
    println!(
        "[GPU] fitness: {:.4} | polygons: {} | evals: {} | evals/s: {:.0} | elapsed: {}s",
        best.fitness,
        best.polygons.len(),
        evals,
        evals_per_sec,
        active_elapsed.as_secs(),
    );
}

fn main_loop(
    sdl_context: &sdl2::Sdl,
    work_receiver: &mpsc::Receiver<EvaluatorPayload>,
    best_sender: &broadcast::Sender<Drawing>,
    texture: &mut sdl2::render::Texture,
    canvas: &mut sdl2::render::WindowCanvas,
    mut global_best: Drawing,
    json_filename: &str,
) {
    let mut event_pump = sdl_context.event_pump().unwrap();
    let frametime = Duration::from_millis(TARGET_FRAMETIME);
    let mut last_draw_timestamp = Instant::now() - frametime;
    let mut stats = EvaluatorPayload::default();
    let mut real_elapsed = Duration::from_millis(0);
    let mut t0 = Instant::now();
    let mut last_save_timestamp = Instant::now();

    // start receiving messages over the channel
    while let Ok(update) = work_receiver.recv() {
        // keep track of real time
        real_elapsed += t0.elapsed();
        t0 = Instant::now();

        // cummulative stats from all threads
        stats.elapsed += update.elapsed;
        stats.evaluations += update.evaluations;
        stats.mutations += update.mutations;
        stats.best = update.best;

        // broadcast new potentially global best to all workers
        if stats.best.fitness > global_best.fitness {
            global_best = stats.best.clone();

            let since_last_save = last_save_timestamp.elapsed().as_secs();
            if since_last_save >= 10 {
                global_best.to_file(json_filename);
                last_save_timestamp = Instant::now();
            }
        }

        // need to always send the current global best to all workers, not only when it changes
        best_sender.send(global_best.clone()).unwrap();

        // everything below here is optional (display new best if enough time has passed)
        let elapsed = last_draw_timestamp.elapsed();
        if elapsed.lt(&frametime) {
            continue;
        }

        // draw upscaled image
        let mut upscale_buf = vec![0u8; DISPLAY_W as usize * DISPLAY_H as usize * 4];
        global_best.draw(
            &mut upscale_buf,
            DISPLAY_W as usize,
            DISPLAY_H as usize,
            Rasterizer::HalfSpace,
        );

        texture
            .update(None, &upscale_buf, DISPLAY_W as usize * 4)
            .unwrap();
        canvas.copy(texture, None, None).unwrap();

        // Present the canvas to display the upscaled image
        canvas.present();
        last_draw_timestamp = Instant::now();
        print_stats(stats.clone(), real_elapsed);
        exhaust_event_pump(&mut event_pump);
    }
}

fn exhaust_event_pump(event_pump: &mut sdl2::EventPump) {
    // TODO: add pause/resume functionality (when pressing space bar)
    for event in event_pump.poll_iter() {
        match event {
            Event::Quit { .. }
            | Event::KeyDown {
                keycode: Some(Keycode::Escape),
                ..
            } => {
                // println!("Exiting, results saved in {:?}");
                exit(0);
            }
            _ => {}
        }
    }
}

fn initialize_sdl(
    w: u32,
    h: u32,
) -> (
    sdl2::Sdl,
    sdl2::render::Canvas<sdl2::video::Window>,
    sdl2::render::TextureCreator<sdl2::video::WindowContext>,
) {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    let window = video_subsystem
        .window("polygon renderer", w, h)
        .position_centered()
        .build()
        .unwrap();

    let canvas = window.into_canvas().build().unwrap();
    let texture_creator = canvas.texture_creator();

    (sdl_context, canvas, texture_creator)
}
