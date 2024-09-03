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
use tokio::sync::broadcast;

fn evaluate(work_sender: mpsc::Sender<EvaluatorPayload>, mut evaluator: Evaluator) {
    loop {
        let update = evaluator.produce_new_best();
        evaluator.reset(update.best.clone());
        let _ = work_sender.send(update);
    }
}

fn print_stats(stats: EvaluatorPayload, real_elapsed: Duration) {
    let t = stats.elapsed;
    if t == 0 || real_elapsed.as_millis() == 0 {
        return;
    }

    let e = stats.evaluations;
    let m = stats.mutations;

    let total_sec = t as f64 / 1000.0;
    let total_min = total_sec / 60.0;
    let total_h = total_min / 60.0;

    let real_ms = real_elapsed.as_millis() as f64;
    let real_sec = real_ms / 1000.0;
    let real_min = real_sec / 60.0;
    let real_h = real_min / 60.0;

    let eval_rate = e as f64 / (real_ms as f64 / 1000.0);
    let mut_rate = m as f64 / (real_ms as f64 / 1000.0);
    let speedup = t as f64 / real_elapsed.as_millis() as f64;

    let time = if real_h > 1.0 {
        format!("{:4.1}h", real_h)
    } else {
        if real_min > 1.0 {
            format!("{:4.1}m", real_min)
        } else {
            format!("{:4.1}s", real_sec)
        }
    };

    let total_time = if total_h > 1.0 {
        format!("{:4.1}h", total_h)
    } else {
        if total_min > 1.0 {
            format!("{:4.1}m", total_min)
        } else {
            format!("{:4.1}s", total_sec)
        }
    };

    println!(
        "Elapsed time: {} | {} total => {:.2}x speedup | evaluations: {:<10} ~{:5.0}/s |  mutations: {:<10} ~{:6.0}/s | best => {:3.4}",
        time, total_time, speedup, e, eval_rate, m, mut_rate, stats.best.fitness
    );
}

// #[tokio::main]
fn main() {
    tracing_subscriber::fmt().init();

    println!("{}", num_cpus::get() - 1);

    let num_threads = (num_cpus::get() - 1).max(2);

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

    let frametime = Duration::from_millis(TARGET_FRAMETIME);
    let mut event_pump = sdl_context.event_pump().unwrap();
    let mut last_draw_timestamp = Instant::now() - frametime;
    let mut stats = EvaluatorPayload::default();
    let mut real_elapsed = Duration::from_millis(0);
    let mut t0 = Instant::now();

    let mut global_best = best.clone();

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
