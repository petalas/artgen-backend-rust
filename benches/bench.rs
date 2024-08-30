use artgen_backend_rust::{
    engine::{
        Engine,
        Rasterizer::{self, *},
    },
    models::{
        color::{Color, RED, WHITE},
        drawing::Drawing,
    },
    utils::blend,
};
use divan::Bencher;

fn main() {
    divan::main();
}

#[divan::bench(args=[HalfSpace, Scanline])]
fn draw(bencher: Bencher, rm: Rasterizer) {
    let w = 384;
    let h = 384;
    let mut buffer = vec![WHITE; w * h];
    let d = Drawing::from_file("ff.json");

    bencher.bench_local(move || {
        d.draw(&mut buffer, w, h, rm);
    });
}

#[divan::bench(args=[HalfSpace, Scanline])]
fn calculate_fitness(bencher: Bencher, rm: Rasterizer) {
    let w = 384;
    let h = 384;

    let mut engine = Engine::new();
    engine.init("ff.jpg", w, h);
    engine.raster_mode = rm;
    let mut d = Drawing::from_file("ff.json");

    bencher.bench_local(move || {
        engine.calculate_fitness(&mut d, false);
    });
}

#[divan::bench]
fn fill_color(bencher: Bencher) {
    let w = 384;
    let h = 384;
    let mut buffer = vec![WHITE; w * h];
    let new_buffer = vec![RED; w * h];

    bencher.bench_local(move || {
        for i in 0..buffer.len() {
            blend(&mut buffer[i], &new_buffer[i]);
        }
    });
}
