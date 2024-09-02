use artgen_backend_rust::{
    engine::{Engine, Rasterizer},
    models::drawing::Drawing,
    settings::{MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH, TARGET_FRAMETIME},
};

use sdl2::keyboard::Keycode;
use sdl2::{event::Event, pixels::PixelFormatEnum};
use std::time::Instant;

// #[tokio::main]
fn main() {
    tracing_subscriber::fmt().init();

    let mut engine = Engine::default();
    engine.raster_mode = Rasterizer::HalfSpace;
    engine.init("ff.jpg", MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT);
    engine.set_best(Drawing::from_file("ff.json"));

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

    // let sdl_context = sdl2::init().unwrap();
    let t0 = Instant::now();
    let mut event_pump = sdl_context.event_pump().unwrap();
    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => {
                    *engine.stopped.blocking_write() = true;
                    break 'running;
                }
                _ => {}
            }
        }

        engine.tick(TARGET_FRAMETIME);
        if *engine.should_display.blocking_read() {
            texture
                .update(None, &engine.working_data, engine.w * 4)
                .unwrap();
            canvas.copy(&texture, None, None).unwrap();
            canvas.present();
        }

        let t = (Instant::now() - t0).as_millis();
        if t < 1 {
            continue;
        }
        let g = engine.stats.blocking_read().generated;
        let sec = (t as f64 / 1000.0).round();
        let rate = (g as f32 / t as f32 * 1000.0).round() as usize;
        println!(
            "Generated {:?} in {}s (~{}/s) --> {}",
            g,
            sec,
            rate,
            engine.current_best.blocking_read().fitness
        );
    }
}
