use artgen_backend_rust::{
    engine::{Engine, Rasterizer},
    models::drawing::Drawing,
    settings::{MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH, TARGET_FRAMETIME},
};

use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use std::time::{Duration, Instant};

// #[tokio::main]
fn main() {
    tracing_subscriber::fmt().init();

    let mut engine = Engine::default();
    engine.raster_mode = Rasterizer::HalfSpace;
    engine.init("ff.jpg", MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT);
    // engine.set_best(Drawing::from_file("ff.json"));

    let sdl_context = sdl2::init().unwrap();
    let t0 = Instant::now();
    let mut event_pump = sdl_context.event_pump().unwrap();
    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                _ => {}
            }
        }

        engine.tick(TARGET_FRAMETIME);
        let t = (Instant::now() - t0).as_millis();
        if t < 1 {
            continue;
        }
        let g = engine.stats.generated;
        let sec = (t as f64 / 1000.0).round();
        let rate = (g as f32 / t as f32 * 1000.0).round() as usize;
        println!(
            "Generated {:?} in {}s (~{}/s) --> {}",
            g, sec, rate, engine.current_best.fitness
        );

        ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));
    }
}
