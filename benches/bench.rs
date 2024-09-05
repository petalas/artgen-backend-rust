use artgen_backend_rust::models::color::Color;
use artgen_backend_rust::{
    engine::{
        Engine,
        Rasterizer::{self, *},
    },
    models::{color::RED, drawing::Drawing},
    utils::{blend, blend_simd, fill_pixel},
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
    engine.init("ff.jpg", w, h, w, h);
    let mut d = Drawing::from_file("ff.json");

    bencher.bench_local(move || {
        engine.calculate_fitness(&mut d, false);
    });
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FillImplementation {
    SIMD,
    Naive,
    Slices,
}
use FillImplementation::*;

#[divan::bench(args=[SIMD, Naive, Slices])]
fn color_blend_single_pixel(bencher: Bencher, implementation: FillImplementation) {
    let color = Color {
        r: 142,
        g: 54,
        b: 113,
        a: 65,
    };
    let color_slice: [u8; 4] = color.as_slice();
    let mut vec_buf: Vec<u8> = vec![134, 125, 152, 45];
    let mut slice_buf: [u8; 4] = vec_buf.clone().as_mut_slice().try_into().unwrap();

    match implementation {
        FillImplementation::SIMD => {
            bencher.bench_local(move || {
                blend_simd(&mut slice_buf, &color_slice);
            });
        }
        FillImplementation::Naive => {
            bencher.bench_local(move || {
                fill_pixel(&mut vec_buf, 0, &RED);
            });
        }
        FillImplementation::Slices => {
            bencher.bench_local(move || {
                blend(&mut slice_buf, &color_slice);
            });
        }
    }
}

#[divan::bench(args=[SIMD, Naive, Slices])]
fn color_blend_entire_buffer(bencher: Bencher, implementation: FillImplementation) {
    let color = Color {
        r: 142,
        g: 54,
        b: 113,
        a: 65,
    };
    let color_slice: [u8; 4] = color.as_slice();
    let width = 384;
    let height = 384;
    let buffer_size = width * height * 4; // 4 bytes per pixel (RGBA)
    let mut vec_buf: Vec<u8> = vec![134, 125, 152, 45].repeat(buffer_size / 4);
    let mut slice_buf: &mut [u8] = vec_buf.as_mut_slice();

    match implementation {
        FillImplementation::SIMD => {
            bencher.bench_local(move || {
                for y in 0..height {
                    for x in 0..width {
                        let idx = (y * width + x) * 4;
                        blend_simd(
                            (&mut slice_buf[idx..idx + 4]).try_into().unwrap(),
                            &color_slice,
                        );
                    }
                }
            });
        }
        FillImplementation::Naive => {
            bencher.bench_local(move || {
                for y in 0..height {
                    for x in 0..width {
                        let idx = (y * width + x) * 4;
                        fill_pixel(&mut vec_buf, idx, &color);
                    }
                }
            });
        }
        FillImplementation::Slices => {
            bencher.bench_local(move || {
                for y in 0..height {
                    for x in 0..width {
                        let idx = (y * width + x) * 4;
                        blend(
                            (&mut slice_buf[idx..idx + 4]).try_into().unwrap(),
                            &color_slice,
                        );
                    }
                }
            });
        }
    }
}
