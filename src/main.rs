use models::{color::Color, drawing::Drawing, line::Line, point::Point, polygon::Polygon};

mod models;
mod settings;
mod utils;

fn main() {
    let d: Drawing = Drawing::new_random();
    // println!("{:?}", d);

    let w: usize = 64;
    let h: usize = 64;
    let size = w * h * 4;
    let mut buffer: Vec<u8> = vec![0u8; size];
    d.draw(&mut buffer, w, h);
    // println!("{:?}", buffer);
}
