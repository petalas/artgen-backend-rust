use artgen_backend_rust::{
    engine::{Engine, Rasterizer},
    evaluator::{Evaluator, EvaluatorPayload},
    models::drawing::Drawing,
    settings::{MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH, TARGET_FRAMETIME},
};

use sdl2::keyboard::Keycode;
use sdl2::{event::Event, pixels::PixelFormatEnum};
use std::{
    sync::mpsc::{self, channel},
    thread,
    time::{Duration, Instant},
};

const N_THREADS: usize = 10;

fn evaluate(tx: mpsc::Sender<EvaluatorPayload>, mut evaluator: Evaluator) {
    loop {
        let update = evaluator.produce_new_best();
        evaluator.reset(update.best.clone());
        let _ = tx.send(update);
    }
}

fn print_stats(stats: EvaluatorPayload) {
    let t = stats.elapsed;
    let e = stats.evaluations;
    let m = stats.mutations;
    let sec = t as f64 / 1000.0;
    let eval_rate = (e as f64 / sec as f64).round() as usize;
    let mut_rate = (m as f64 / sec as f64).round() as usize;
    println!(
        "Elapsed time (across {} threads): {}, evaluations {} (~{}/s),  mutations {} (~{}/s) --> {}",
        N_THREADS, sec, e, eval_rate, m, mut_rate, stats.best.fitness
    );
}

// #[tokio::main]
fn main() {
    tracing_subscriber::fmt().init();

    let mut engine = Engine::default();
    engine.raster_mode = Rasterizer::HalfSpace;
    engine.init("ff.jpg", MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT);

    // let best = Drawing::from_file("ff.json");
    let best = Drawing::new_random();

    // sdl setup on the main thread
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    let window = video_subsystem
        .window("polygon renderer", engine.w as u32, engine.h as u32)
        .position_centered()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas().build().unwrap();
    let texture_creator = canvas.texture_creator();
    let mut texture = texture_creator
        .create_texture_streaming(PixelFormatEnum::ABGR8888, engine.w as u32, engine.h as u32)
        .unwrap();

    canvas.clear();
    canvas.present();

    // 1. create a channel so workers & UI thread can communicate
    let (tx, rx) = channel::<EvaluatorPayload>();

    // 2. spawn a bunch of worker threads, giving each a sender
    let workers = (0..N_THREADS)
        .map(|_| {
            let tx = tx.clone();
            // TODO: remove refs to engine, calculate ref_image_data, w, h in main
            let evaluator = Evaluator::new(
                engine.ref_image_data.clone(),
                engine.w,
                engine.h,
                best.clone(),
            );
            thread::spawn(move || evaluate(tx, evaluator))
        })
        .collect::<Vec<_>>();

    drop(tx);

    let frametime = Duration::from_millis(TARGET_FRAMETIME);
    let mut event_pump = sdl_context.event_pump().unwrap();
    let mut last_draw_timestamp = Instant::now();
    let mut stats = EvaluatorPayload::default();

    // start receiving messages over the channel
    while let Ok(update) = rx.recv() {
        stats.elapsed += update.elapsed;
        stats.evaluations += update.evaluations;
        stats.mutations += update.mutations;
        stats.best = update.best;

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
        print_stats(stats.clone());
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

    for worker in workers {
        worker.join().expect("worker panicked");
    }
}
