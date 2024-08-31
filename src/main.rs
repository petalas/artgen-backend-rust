

// FIXME: figure out what to do about runtimes, show_image wants to be run from main

use artgen_backend_rust::{engine::{Engine, Rasterizer}, models::drawing::Drawing, settings::{MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH}};
use winit::event_loop::{ControlFlow, EventLoop};

#[tokio::main]
async fn main() {
    // tracing_subscriber::fmt().init();

    let event_loop = EventLoop::new().unwrap();
    // event_loop.set_control_flow(ControlFlow::Poll);

    let mut engine = Engine::default();
    engine.raster_mode = Rasterizer::GPU;
    engine.init("ff.jpg", MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT);
    engine.set_best(Drawing::from_file("ff.json"));

    event_loop.run_app(&mut engine).expect("app to run")

    

    // engine.test().await;

    // let mut ticks: usize = 0;
    // let t0 = Instant::now();
    // loop {
    //     ticks += 1;
    //     engine.tick(TARGET_FRAMETIME).await;
    //     if ticks % TARGET_FRAMETIME == 0 {
    //         let t = (Instant::now() - t0).as_millis();
    //         if t < 1 {
    //             continue;
    //         }
    //         let g = engine.stats.generated;
    //         let rate = (g as f32 / t as f32 * 1000.0).round() as usize;
    //         println!(
    //             "Generated {:?} in {}s (~{}/s) --> {}",
    //             g, t, rate, engine.current_best.fitness
    //         );
    //     }
    // }

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
