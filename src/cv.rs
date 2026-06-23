//! Pure-Rust computer-vision primitives needed to replicate the PaddleOCR
//! detection post-processing and crop extraction without OpenCV.

pub type Pt = (f64, f64);

/// Andrew's monotone chain convex hull. Returns hull points CCW.
pub fn convex_hull(points: &[Pt]) -> Vec<Pt> {
    let mut pts: Vec<Pt> = points.to_vec();
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap()));
    pts.dedup();
    let n = pts.len();
    if n < 3 {
        return pts;
    }
    let cross = |o: Pt, a: Pt, b: Pt| (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0);
    let mut hull: Vec<Pt> = Vec::with_capacity(2 * n);
    // lower
    for &p in &pts {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    // upper
    let lower = hull.len() + 1;
    for &p in pts.iter().rev() {
        while hull.len() >= lower && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();
    hull
}

/// Minimum-area enclosing rectangle via rotating calipers over the convex hull.
/// Returns the 4 corner points and the shorter side length.
pub fn min_area_rect(points: &[Pt]) -> ([Pt; 4], f64) {
    let hull = convex_hull(points);
    if hull.is_empty() {
        return ([(0.0, 0.0); 4], 0.0);
    }
    if hull.len() == 1 {
        let p = hull[0];
        return ([p; 4], 0.0);
    }
    if hull.len() == 2 {
        let (a, b) = (hull[0], hull[1]);
        return ([a, b, b, a], 0.0);
    }
    let n = hull.len();
    let mut best_area = f64::INFINITY;
    let mut best: ([Pt; 4], f64) = ([(0.0, 0.0); 4], 0.0);
    for i in 0..n {
        let p0 = hull[i];
        let p1 = hull[(i + 1) % n];
        let mut ex = p1.0 - p0.0;
        let mut ey = p1.1 - p0.1;
        let len = (ex * ex + ey * ey).sqrt();
        if len < 1e-9 {
            continue;
        }
        ex /= len;
        ey /= len;
        // normal
        let nx = -ey;
        let ny = ex;
        let (mut min_u, mut max_u) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut min_v, mut max_v) = (f64::INFINITY, f64::NEG_INFINITY);
        for &p in &hull {
            let u = p.0 * ex + p.1 * ey;
            let v = p.0 * nx + p.1 * ny;
            min_u = min_u.min(u);
            max_u = max_u.max(u);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        let w = max_u - min_u;
        let h = max_v - min_v;
        let area = w * h;
        if area < best_area {
            best_area = area;
            // corners in (u,v) space -> world space
            let to_world = |u: f64, v: f64| (u * ex + v * nx, u * ey + v * ny);
            let c = [
                to_world(min_u, min_v),
                to_world(max_u, min_v),
                to_world(max_u, max_v),
                to_world(min_u, max_v),
            ];
            best = (c, w.min(h));
        }
    }
    best
}

/// Polygon area (shoelace, absolute).
pub fn poly_area(pts: &[Pt]) -> f64 {
    let n = pts.len();
    let mut a = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        a += pts[i].0 * pts[j].1 - pts[j].0 * pts[i].1;
    }
    a.abs() / 2.0
}

/// Polygon perimeter (closed).
pub fn poly_perimeter(pts: &[Pt]) -> f64 {
    let n = pts.len();
    let mut p = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        let dx = pts[j].0 - pts[i].0;
        let dy = pts[j].1 - pts[i].1;
        p += (dx * dx + dy * dy).sqrt();
    }
    p
}

/// Grow a (near-)rectangular quad outward by `dist` on every side. Equivalent to
/// pyclipper round offset of a rectangle followed by minAreaRect: expand each
/// corner along the bisector away from the centroid by `dist`.
pub fn unclip_rect(box4: &[Pt; 4], dist: f64) -> [Pt; 4] {
    let cx = (box4[0].0 + box4[1].0 + box4[2].0 + box4[3].0) / 4.0;
    let cy = (box4[0].1 + box4[1].1 + box4[2].1 + box4[3].1) / 4.0;
    // Recover orthonormal basis from edges.
    let mut ex = box4[1].0 - box4[0].0;
    let mut ey = box4[1].1 - box4[0].1;
    let l1 = (ex * ex + ey * ey).sqrt();
    if l1 < 1e-9 {
        return *box4;
    }
    ex /= l1;
    ey /= l1;
    let nx = -ey;
    let ny = ex;
    // half extents
    let mut out = [(0.0, 0.0); 4];
    for (i, p) in box4.iter().enumerate() {
        let du = (p.0 - cx) * ex + (p.1 - cy) * ey;
        let dv = (p.0 - cx) * nx + (p.1 - cy) * ny;
        let su = if du >= 0.0 { du + dist } else { du - dist };
        let sv = if dv >= 0.0 { dv + dist } else { dv - dist };
        out[i] = (cx + su * ex + sv * nx, cy + su * ey + sv * ny);
    }
    out
}

/// Mean of `pred` over the filled quad, restricted to its bounding box.
/// Mirrors DBPostProcess.box_score_fast.
pub fn box_score_fast(pred: &[f32], w: usize, h: usize, box4: &[Pt; 4]) -> f32 {
    let xs = [box4[0].0, box4[1].0, box4[2].0, box4[3].0];
    let ys = [box4[0].1, box4[1].1, box4[2].1, box4[3].1];
    let xmin = (xs.iter().cloned().fold(f64::INFINITY, f64::min).floor() as i64).clamp(0, w as i64 - 1) as usize;
    let xmax = (xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max).ceil() as i64).clamp(0, w as i64 - 1) as usize;
    let ymin = (ys.iter().cloned().fold(f64::INFINITY, f64::min).floor() as i64).clamp(0, h as i64 - 1) as usize;
    let ymax = (ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max).ceil() as i64).clamp(0, h as i64 - 1) as usize;
    if xmax < xmin || ymax < ymin {
        return 0.0;
    }
    let bw = xmax - xmin + 1;
    let bh = ymax - ymin + 1;
    // local polygon
    let poly: Vec<Pt> = box4.iter().map(|p| (p.0 - xmin as f64, p.1 - ymin as f64)).collect();
    let mut sum = 0.0f64;
    let mut cnt = 0u64;
    // scanline fill
    for row in 0..bh {
        let yc = row as f64 + 0.5; // pixel center; matches cv2 fillPoly coverage closely
        let mut xints: Vec<f64> = Vec::with_capacity(4);
        let m = poly.len();
        for i in 0..m {
            let (x1, y1) = poly[i];
            let (x2, y2) = poly[(i + 1) % m];
            if (y1 <= yc && y2 > yc) || (y2 <= yc && y1 > yc) {
                let t = (yc - y1) / (y2 - y1);
                xints.push(x1 + t * (x2 - x1));
            }
        }
        if xints.len() < 2 {
            continue;
        }
        xints.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut k = 0;
        while k + 1 < xints.len() {
            let xa = xints[k].max(0.0).round() as i64;
            let xb = xints[k + 1].min(bw as f64 - 1.0).round() as i64;
            for x in xa..=xb {
                if x >= 0 && (x as usize) < bw {
                    let gx = xmin + x as usize;
                    let gy = ymin + row;
                    sum += pred[gy * w + gx] as f64;
                    cnt += 1;
                }
            }
            k += 2;
        }
    }
    if cnt == 0 {
        // fallback: nearest pixel
        return pred[ymin * w + xmin];
    }
    (sum / cnt as f64) as f32
}

/// Solve 3x3 perspective transform H mapping src_pts -> dst_pts (4 correspondences).
/// Returns row-major [9].
fn get_perspective(src: &[Pt; 4], dst: &[Pt; 4]) -> [f64; 9] {
    // Build 8x8 system for [h0..h7], h8=1.
    let mut a = [[0.0f64; 9]; 8]; // augmented 8x(8+1)
    for i in 0..4 {
        let (x, y) = src[i];
        let (u, v) = dst[i];
        a[i * 2] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
        a[i * 2 + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
    }
    // Gaussian elimination
    for col in 0..8 {
        // pivot
        let mut piv = col;
        for r in col + 1..8 {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        a.swap(col, piv);
        let d = a[col][col];
        if d.abs() < 1e-12 {
            continue;
        }
        for c in col..9 {
            a[col][c] /= d;
        }
        for r in 0..8 {
            if r != col {
                let f = a[r][col];
                if f != 0.0 {
                    for c in col..9 {
                        a[r][c] -= f * a[col][c];
                    }
                }
            }
        }
    }
    [
        a[0][8], a[1][8], a[2][8], a[3][8], a[4][8], a[5][8], a[6][8], a[7][8], 1.0,
    ]
}

/// Warp-crop a BGR image given 4 source quad points to a `dst_w x dst_h` BGR crop
/// using inverse bilinear sampling with border replicate. Mirrors
/// get_rotate_crop_image (perspective transform). Bilinear is used instead of
/// cv2's INTER_CUBIC: it is faster and yields identical recognition fidelity
/// here (the residual diff vs paddle is ONNXRuntime-vs-paddle numerics).
pub fn warp_crop(src: &[u8], sw: usize, sh: usize, quad: &[Pt; 4], dst_w: usize, dst_h: usize) -> Vec<u8> {
    let dst_pts: [Pt; 4] = [
        (0.0, 0.0),
        (dst_w as f64, 0.0),
        (dst_w as f64, dst_h as f64),
        (0.0, dst_h as f64),
    ];
    // H maps dst -> src so we can inverse sample.
    let h = get_perspective(&dst_pts, quad);
    let mut out = vec![0u8; dst_w * dst_h * 3];
    for y in 0..dst_h {
        for x in 0..dst_w {
            let fx = x as f64;
            let fy = y as f64;
            let w = h[6] * fx + h[7] * fy + h[8];
            let sx = (h[0] * fx + h[1] * fy + h[2]) / w;
            let sy = (h[3] * fx + h[4] * fy + h[5]) / w;
            let x0 = sx.floor();
            let y0 = sy.floor();
            let ax = sx - x0;
            let ay = sy - y0;
            let x0i = (x0 as i64).clamp(0, sw as i64 - 1) as usize;
            let y0i = (y0 as i64).clamp(0, sh as i64 - 1) as usize;
            let x1i = (x0 as i64 + 1).clamp(0, sw as i64 - 1) as usize;
            let y1i = (y0 as i64 + 1).clamp(0, sh as i64 - 1) as usize;
            let o = (y * dst_w + x) * 3;
            for c in 0..3 {
                let p00 = src[(y0i * sw + x0i) * 3 + c] as f64;
                let p01 = src[(y0i * sw + x1i) * 3 + c] as f64;
                let p10 = src[(y1i * sw + x0i) * 3 + c] as f64;
                let p11 = src[(y1i * sw + x1i) * 3 + c] as f64;
                let top = p00 * (1.0 - ax) + p01 * ax;
                let bot = p10 * (1.0 - ax) + p11 * ax;
                let v = top * (1.0 - ay) + bot * ay;
                out[o + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Rotate a BGR image 90 degrees counter-clockwise (np.rot90).
pub fn rot90_ccw(src: &[u8], w: usize, h: usize) -> (Vec<u8>, usize, usize) {
    let (nw, nh) = (h, w);
    let mut out = vec![0u8; nw * nh * 3];
    for y in 0..h {
        for x in 0..w {
            // np.rot90: out[w-1-x][y] = in[y][x]
            let ny = w - 1 - x;
            let nx = y;
            let si = (y * w + x) * 3;
            let di = (ny * nw + nx) * 3;
            out[di..di + 3].copy_from_slice(&src[si..si + 3]);
        }
    }
    (out, nw, nh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convex_hull_square() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)];
        let hull = convex_hull(&pts);
        assert_eq!(hull.len(), 4); // interior point dropped
    }

    #[test]
    fn min_area_rect_axis_aligned() {
        // a 10x4 axis-aligned rectangle of points
        let mut pts = Vec::new();
        for x in 0..=10 {
            for y in 0..=4 {
                pts.push((x as f64, y as f64));
            }
        }
        let (_box, side) = min_area_rect(&pts);
        assert!((side - 4.0).abs() < 1e-6, "short side should be 4, got {side}");
    }

    #[test]
    fn unclip_grows_rect() {
        let b = [(0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (0.0, 4.0)];
        let g = unclip_rect(&b, 2.0);
        // every corner should move outward (area strictly larger)
        let area0 = poly_area(&b);
        let area1 = poly_area(&g);
        assert!(area1 > area0, "unclip must enlarge: {area0} -> {area1}");
    }

    #[test]
    fn box_score_uniform() {
        // 5x5 prob map all 0.8; a box covering it -> mean ~0.8
        let w = 5;
        let h = 5;
        let pred = vec![0.8f32; w * h];
        let b = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)];
        let s = box_score_fast(&pred, w, h, &b);
        assert!((s - 0.8).abs() < 0.05, "score {s}");
    }
}
