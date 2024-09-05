use artgen_backend_rust::{
    engine::{Engine, Rasterizer},
    evaluator::{Evaluator, EvaluatorPayload},
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
    path::Path,
    process::exit,
    sync::mpsc::{self, channel},
    thread,
    time::{Duration, Instant},
};
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
    let mut engine = Engine::default();
    engine.raster_mode = Rasterizer::HalfSpace;
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

    let ref_image_filename = if args.len() > 1 {
        &args[1]
    } else {
        DEFAULT_REF_IMAGE_FILENAME
    };

    let json_filename = if args.len() > 1 {
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

    let num_threads = num_cpus::get();
    let mut engine = initialize_engine(ref_image_filename);

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

    // channel to send work to worker threads
    let (work_sender, work_receiver) = channel::<EvaluatorPayload>();

    // broadcast channel to send out new best to all workers
    let (best_sender, best_receiver) = broadcast::channel::<Drawing>(num_threads);

    // 2. spawn a bunch of worker threads, giving each a sender
    let workers = (0..num_threads)
        .map(|_| {
            let ws = work_sender.clone(); // to send new work to threads
            let br = best_sender.subscribe();
            // TODO: remove refs to engine, calculate ref_image_data, w, h in main
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
        &sdl_context,
        &work_receiver,
        &best_sender,
        &mut texture,
        &mut canvas,
        global_best,
        &json_filename,
    );

    for worker in workers {
        worker.join().expect("worker panicked");
    }
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
        canvas.copy(&texture, None, None).unwrap();

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
