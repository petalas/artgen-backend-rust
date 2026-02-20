use artgen_backend_rust::{
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
    tracing_subscriber::fmt().init();

    let args: Vec<String> = env::args().collect();

    let use_gpu = args.iter().any(|a| a == "--gpu");
    let headless = args.iter().any(|a| a == "--headless");

    // Filter out flags to get positional args
    let positional: Vec<&String> = args.iter().skip(1).filter(|a| !a.starts_with("--")).collect();

    if use_gpu && headless {
        // Project management mode — image arg is optional
        let legacy_image = positional.first().map(|s| s.as_str());
        println!("Starting GPU evolution pipeline (headless, project management mode)...");
        gpu_main_loop_headless(legacy_image);
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
        if let Some(new_best) = evolver.run_batch(&default_params) {
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
            last_stats_timestamp = Instant::now();
        }
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
    // Mutation parameters (runtime-configurable)
    mutation_params: MutationParams,
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
    let config = tungstenite::protocol::WebSocketConfig {
        max_message_size: Some(20 * 1024 * 1024),
        max_frame_size: Some(20 * 1024 * 1024),
        ..Default::default()
    };

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
                    if let Ok(cmd) = serde_json::from_str::<serde_json::Value>(&*text) {
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
            let s = lock.lock().unwrap();

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
        Some("update_params") => {
            if let Ok(params) = serde_json::from_value::<MutationParams>(cmd["params"].clone()) {
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

fn gpu_main_loop_headless(legacy_image: Option<&str>) {
    projects::ensure_projects_dir();

    // Legacy migration
    let migrated_project = projects::migrate_legacy(legacy_image);

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
            mutation_params: MutationParams::default(),
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

        // Load project reference image
        let ref_path = projects::project_reference_path(&project_name);
        if !ref_path.exists() {
            eprintln!("[Projects] Reference image not found for '{}', skipping", project_name);
            continue;
        }

        let ref_image_bytes = match std::fs::read(&ref_path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("[Projects] Failed to read reference for '{}': {}", project_name, e);
                continue;
            }
        };

        let (rgba, w, h) = match projects::load_and_normalize_image(&ref_image_bytes) {
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
            if let Some(new_best) = evolver.run_batch(&mp) {
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
                {
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

                print_gpu_stats_active(&evolver, &global_best, active_elapsed);
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
