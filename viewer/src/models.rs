use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Drawing {
    pub polygons: Vec<Polygon>,
    pub is_dirty: bool,
    pub fitness: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Polygon {
    pub points: Vec<Point>,
    pub color: Color,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Drawing {
    pub fn to_svg(&self, width: u32, height: u32) -> String {
        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}">"#,
            width, height
        );
        svg.push('\n');

        // White background
        svg.push_str(&format!(
            r#"  <rect width="{}" height="{}" fill="white"/>"#,
            width, height
        ));
        svg.push('\n');

        for polygon in &self.polygons {
            let points_str: String = polygon
                .points
                .iter()
                .map(|p| format!("{},{}", p.x * width as f32, p.y * height as f32))
                .collect::<Vec<_>>()
                .join(" ");

            let fill = format!(
                "#{:02x}{:02x}{:02x}",
                polygon.color.r, polygon.color.g, polygon.color.b
            );
            let opacity = polygon.color.a as f32 / 255.0;

            svg.push_str(&format!(
                r#"  <polygon points="{}" fill="{}" fill-opacity="{:.4}"/>"#,
                points_str, fill, opacity
            ));
            svg.push('\n');
        }

        svg.push_str("</svg>");
        svg
    }
}
