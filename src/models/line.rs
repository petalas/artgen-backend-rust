use super::point::Point;

#[derive(Debug)]
pub struct Line<'a> {
    pub a: &'a Point,
    pub b: &'a Point,
    pub slope: f64,

    pub max_x: usize,
    pub max_y: usize,
    pub min_x: usize,
    pub min_y: usize,
    pub x_of_min_y: f64,
    pub inverse_slope: f64,
    pub next_x_to_fill: f64,
}

impl<'a> Line<'a> {
    pub fn new(a: &'a Point, b: &'a Point) -> Self {
        let dx = b.x - a.x;
        let slope = if dx == 0.0 { 0.0 } else { (b.y - a.y) / dx };
        let inverse_slope = if dx == 0.0 { 0.0 } else { dx / (b.y - a.y) };
        let next_x_to_fill = if a.y < b.y { a.x } else { b.x };
        let x_of_min_y = if a.y < b.y { a.x } else { a.y };
        Self {
            a,
            b,
            slope,
            max_x: a.x.max(b.x) as usize,
            max_y: a.y.max(b.y) as usize,
            min_x: a.x.min(b.x) as usize,
            min_y: a.y.min(b.y) as usize,
            x_of_min_y,
            inverse_slope,
            next_x_to_fill,
        }
    }

    pub fn max_y(&self) -> usize {
        self.a.y.max(self.b.y) as usize
    }

    pub fn min_y(&self) -> usize {
        self.a.y.min(self.b.y) as usize
    }

    /// returns false if this line segment is entirely above or below the row
    pub fn intersects(&self, row: usize) -> bool {
        let y = row as f64;
        (self.a.y >= y && self.b.y <= y) || (self.a.y <= y && self.b.y >= y)
    }

    /// y = mx + c => x = (y - c) / m               // line equation -> solve for x
    /// m = dy / dx = (b.y - a.y) / (b.x - a.x)     // slope
    pub fn x_intercept(&self, y: usize) -> f64 {
        if self.slope == 0.0 {
            // vertical line
            return self.a.x;
        }
        let y = y as f64;
        let c = self.a.y - (self.slope * self.a.x);
        (y - c) / self.slope // we return early for vertical lines -> no risk of dividing by zero
    }

    pub fn intersection_point(&self, row: usize) -> Option<Point> {
        if !self.intersects(row) {
            return None;
        } else {
            let x = self.x_intercept(row);
            return Some(Point {
                x: x.round(),
                y: row as f64,
            });
        }
    }
}
