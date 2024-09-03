use artgen_backend_rust::{
    engine::{Engine, Rasterizer},
    evaluator::{Evaluator, EvaluatorPayload},
    models::drawing::Drawing,
    settings::{MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH, TARGET_FRAMETIME},
    utils::print_stats,
};

use sdl2::keyboard::Keycode;
use sdl2::{event::Event, pixels::PixelFormatEnum};
use std::{
    sync::mpsc::{self, channel},
    thread,
    time::{Duration, Instant},
};
use tokio::sync::broadcast;

fn evaluate(work_sender: mpsc::Sender<EvaluatorPayload>, mut evaluator: Evaluator) {
    loop {
        let update = evaluator.produce_new_best();
        evaluator.reset(update.best.clone());
        let _ = work_sender.send(update);
    }
}

fn main() {
    tracing_subscriber::fmt().init();

    println!("{}", num_cpus::get() - 1);

    let num_threads = (num_cpus::get() - 1).max(2);

    let mut engine = Engine::default();
    engine.raster_mode = Rasterizer::HalfSpace;
    engine.init("ff.jpg", MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT);

    let best = Drawing::from_file("ff.json");
    // let best = Drawing::from_file("best.json");
    // let best = Drawing::new_random();

    let (sdl_context, mut canvas, texture_creator) =
        initialize_sdl(engine.w as u32, engine.h as u32);
    let mut texture = texture_creator
        .create_texture_streaming(PixelFormatEnum::ABGR8888, engine.w as u32, engine.h as u32)
        .unwrap();

    canvas.clear();
    canvas.present();

    // channel to send work to worker threads
    let (work_sender, work_receiver) = channel::<EvaluatorPayload>();

    // broadcast channel to send out new best to all workers
    let (best_sender, best_receiver) = broadcast::channel::<Drawing>(1);

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
        &mut engine,
        global_best,
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
    engine: &mut Engine,
    mut global_best: Drawing,
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
            best_sender.send(global_best.clone()).unwrap();
            // TODO: every 10 minutes, save the best to disk

            if last_save_timestamp.elapsed().as_secs() >= 10 {
                global_best.to_file("best.json");
                print!("Saved best to disk\n");
                last_save_timestamp = Instant::now();
            }
        }

        // everything below here is optional (display new best if enough time has passed)
        let elapsed = last_draw_timestamp.elapsed();
        if elapsed.lt(&frametime) {
            continue;
        }

        texture
            .update(None, &update.working_data, engine.w * 4)
            .unwrap();
        canvas.copy(&texture, None, None).unwrap();
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
                panic!("exited");
            }
            _ => {}
        }
    }
}

// TODO: create a helper method to initialize all the sdl stuff
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
