use std::time::Instant;

use artgen_backend_rust::{
    engine::{self, Engine, Rasterizer},
    models::drawing::Drawing,
    settings::{MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH, TARGET_FRAMETIME},
};
use show_image::event;

// mod engine;
// mod models;
// mod settings;
// mod utils;

#[show_image::main]
fn main() {
    let mut engine = Engine::new();
    engine.init("ff.jpg", MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT);
    engine.set_best(Drawing::from_file("ff.json"));

    // engine.test();
    // engine.test2();
    // engine.test3();

    let mut ticks: usize = 0;
    let t0 = Instant::now();
    loop {
        ticks += 1;
        engine.tick(TARGET_FRAMETIME);
        if ticks % TARGET_FRAMETIME == 0 {
            let t = (Instant::now() - t0).as_secs();
            if t < 1 {
                continue;
            }
            let g = engine.stats.generated;
            let rate = (g as f32 / t as f32).round() as usize;
            println!(
                "Generated {:?} in {}s (~{}/s) --> {}",
                g, t, rate, engine.current_best.fitness
            );
        }
    }

    // for event in engine
    //     .window
    //     .event_channel()
    //     .expect("Failed to get event_channel")
    // {
    //     if let event::WindowEvent::KeyboardInput(event) = event {
    //         // println!("{:#?}", event);
    //         if event.input.key_code == Some(event::VirtualKeyCode::Escape)
    //             && event.input.state.is_pressed()
    //         {
    //             break;
    //         }
    //         if event.input.key_code == Some(event::VirtualKeyCode::Space)
    //             && event.input.state.is_pressed()
    //         {
    //             engine.raster_mode = if engine.raster_mode == Rasterizer::Scanline { Rasterizer::HalfSpace } else { Rasterizer::Scanline };
    //             println!("raster mode set to {:?}", engine.raster_mode);
    //             engine.redraw();
    //         }
    //     }
    //     engine.tick(TARGET_FRAMETIME);
    //     println!("{:?}", engine.stats);
    // }
}
