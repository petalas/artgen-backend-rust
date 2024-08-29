use artgen_backend_rust::{
    engine::Rasterizer::{self, *},
    models::drawing::Drawing,
};
use divan::Bencher;

fn main() {
    divan::main();
}

#[divan::bench(args=[HalfSpace, Scanline], sample_count=100, sample_size=10)]
fn draw(bencher: Bencher, rm: Rasterizer) {
    let w = 384;
    let h = 384;
    let mut buffer = vec![0u8; w * h * 4];
    let d = Drawing::from_file("ff.json");

    bencher.bench_local(move || {
        d.draw(&mut buffer, w, h, rm);
    });
}
