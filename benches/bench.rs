use artgen_backend_rust::{
    engine::{
        Engine,
        Rasterizer::{self, *},
    },
    models::{color::RED, drawing::Drawing},
    utils::fill_pixel,
};
use divan::Bencher;

fn main() {
    divan::main();
}

#[divan::bench(args=[HalfSpace, Scanline])]
fn draw(bencher: Bencher, rm: Rasterizer) {
    let w = 384;
    let h = 384;
    let mut buffer = vec![0u8; w * h * 4];
    let d = Drawing::from_file("ff.json");

    bencher.bench_local(move || {
        d.draw(&mut buffer, w, h, rm);
    });
}

#[divan::bench(args=[HalfSpace, Scanline, GPU])]
fn calculate_fitness(bencher: Bencher, rm: Rasterizer) {
    let w = 384;
    let h = 384;

    let mut engine = Engine::default();
    engine.raster_mode = rm;
    engine.init("ff.jpg", w, h);
    let mut d = Drawing::from_file("ff.json");

    bencher.bench_local(move || {
        engine.calculate_fitness(&mut d, false);
    });
}

#[divan::bench]
fn fill_color(bencher: Bencher) {
    let w = 384;
    let h = 384;
    let num_pixels = w * h;
    let mut buffer = vec![255u8; num_pixels * 4];

    bencher.bench_local(move || {
        for i in 0..num_pixels {
            fill_pixel(&mut buffer, i, &RED);
        }
    });
}
