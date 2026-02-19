use artgen_backend_rust::{
    engine::{Engine, Rasterizer},
    evaluator::{Evaluator, EvaluatorPayload},
    gpu_evolver::GpuEvolver,
    models::drawing::Drawing,
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

fn main() {
    tracing_subscriber::fmt().init();

    let args: Vec<String> = env::args().collect();

    let use_gpu = args.iter().any(|a| a == "--gpu");
    let headless = args.iter().any(|a| a == "--headless");

    // Filter out flags to get positional args
    let positional: Vec<&String> = args.iter().skip(1).filter(|a| !a.starts_with("--")).collect();

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

    if use_gpu && headless {
        println!("Starting GPU evolution pipeline (headless)...");
        gpu_main_loop_headless(&engine, best, &json_filename);
        return;
    }

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

    loop {
        // Run a batch of GPU iterations
        if let Some(new_best) = evolver.run_batch() {
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
    paused: bool,
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

fn ws_handle_client(stream: std::net::TcpStream, state: SharedWsState) {
    let peer = stream.peer_addr().ok();
    let mut ws = match tungstenite::accept(stream) {
        Ok(ws) => ws,
        Err(e) => {
            eprintln!("[WS] Handshake error: {}", e);
            return;
        }
    };

    println!("[WS] Client connected: {:?}", peer);

    let (lock, cvar) = &*state;

    // Send init message (blocking mode)
    let mut last_gen;
    let mut last_image_gen;
    {
        let s = lock.lock().unwrap();
        last_gen = s.generation;
        last_image_gen = s.image_generation;
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
        });
        if ws
            .send(tungstenite::Message::Text(msg.to_string().into()))
            .is_err()
        {
            println!("[WS] Client disconnected during init: {:?}", peer);
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
                        match cmd["type"].as_str() {
                            Some("pause") | Some("resume") => {
                                let pause = cmd["type"].as_str() == Some("pause");
                                let mut s = lock.lock().unwrap();
                                s.paused = pause;
                                s.generation += 1;
                                cvar.notify_all();
                                println!("[WS] {} by {:?}", if pause { "Paused" } else { "Resumed" }, peer);
                            }
                            _ => {}
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

        // 2. Check for new state to send
        let cur_gen = lock.lock().unwrap().generation;
        if cur_gen == last_gen {
            thread::sleep(Duration::from_millis(50));
            continue;
        }

        // 3. Build message while holding lock, then drop before sending
        let msg_string = {
            let s = lock.lock().unwrap();
            let has_new_image = s.image_generation > last_image_gen;
            last_gen = s.generation;

            let msg = if has_new_image {
                last_image_gen = s.image_generation;
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

fn gpu_main_loop_headless(engine: &Engine, initial_best: Drawing, json_filename: &str) {
    let mut evolver = futures_lite::future::block_on(GpuEvolver::new(
        &engine.ref_image_data,
        engine.w as u32,
        engine.h as u32,
        &initial_best,
    ));

    let w = engine.w;
    let h = engine.h;
    let mut render_buf = vec![0u8; w * h * 4];
    let png_path = json_filename.replace(".best.json", ".best.png");

    // Encode reference image as PNG for WS clients
    let ref_png = encode_rgba_as_png(&engine.ref_image_data, w, h);

    // Encode and save initial best as PNG
    initial_best.draw(&mut render_buf, w, h, Rasterizer::HalfSpace);
    let initial_png = encode_rgba_as_png(&render_buf, w, h);
    if let Err(e) = std::fs::write(&png_path, &initial_png) {
        eprintln!("Failed to save PNG: {}", e);
    } else {
        println!("[GPU] Saved preview: {}", png_path);
    }

    // Create shared state for WS server
    let ws_state: SharedWsState = Arc::new((
        Mutex::new(WsState {
            ref_png,
            best_png: initial_png,
            fitness: initial_best.fitness,
            polygons: initial_best.polygons.len(),
            improvements: 0,
            evals_per_sec: 0.0,
            total_evals: 0,
            elapsed_secs: 0,
            generation: 0,
            image_generation: 0,
            paused: false,
        }),
        Condvar::new(),
    ));

    // Spawn WS server thread
    let ws_clone = ws_state.clone();
    thread::spawn(move || ws_server(ws_clone));

    let mut global_best = initial_best;
    let mut last_save_timestamp = Instant::now();
    let mut last_stats_timestamp = Instant::now();
    let mut improvements = 0u64;
    let mut batches = 0u64;

    loop {
        // Check if paused
        if ws_state.0.lock().unwrap().paused {
            thread::sleep(Duration::from_millis(50));
            // Still send stats while paused
            if last_stats_timestamp.elapsed().as_secs() >= 2 {
                {
                    let (lock, cvar) = &*ws_state;
                    let mut s = lock.lock().unwrap();
                    s.evals_per_sec = evolver.evals_per_sec();
                    s.total_evals = evolver.total_evaluations();
                    s.elapsed_secs = evolver.elapsed().as_secs();
                    s.fitness = global_best.fitness;
                    s.polygons = global_best.polygons.len();
                    s.improvements = improvements;
                    s.generation += 1;
                    cvar.notify_all();
                }
                print_gpu_stats(&evolver, &global_best);
                last_stats_timestamp = Instant::now();
            }
            continue;
        }

        batches += 1;
        if let Some(new_best) = evolver.run_batch() {
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

                // Render and encode PNG for WS + disk save
                global_best.draw(&mut render_buf, w, h, Rasterizer::HalfSpace);
                let png = encode_rgba_as_png(&render_buf, w, h);

                // Update WS state
                {
                    let (lock, cvar) = &*ws_state;
                    let mut s = lock.lock().unwrap();
                    s.best_png = png.clone();
                    s.fitness = global_best.fitness;
                    s.polygons = global_best.polygons.len();
                    s.improvements = improvements;
                    s.evals_per_sec = evolver.evals_per_sec();
                    s.total_evals = evolver.total_evaluations();
                    s.elapsed_secs = evolver.elapsed().as_secs();
                    s.generation += 1;
                    s.image_generation += 1;
                    cvar.notify_all();
                }

                // Save to disk periodically
                let since_last_save = last_save_timestamp.elapsed().as_secs();
                if since_last_save >= 10 {
                    global_best.to_file(json_filename);
                    if let Err(e) = std::fs::write(&png_path, &png) {
                        eprintln!("Failed to save PNG: {}", e);
                    } else {
                        println!("[GPU] Saved preview: {}", png_path);
                    }
                    last_save_timestamp = Instant::now();
                }
            }
        }

        if last_stats_timestamp.elapsed().as_secs() >= 2 {
            // Update WS state with latest stats
            {
                let (lock, cvar) = &*ws_state;
                let mut s = lock.lock().unwrap();
                s.evals_per_sec = evolver.evals_per_sec();
                s.total_evals = evolver.total_evaluations();
                s.elapsed_secs = evolver.elapsed().as_secs();
                s.fitness = global_best.fitness;
                s.polygons = global_best.polygons.len();
                s.improvements = improvements;
                s.generation += 1;
                cvar.notify_all();
            }

            print_gpu_stats(&evolver, &global_best);
            last_stats_timestamp = Instant::now();
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
