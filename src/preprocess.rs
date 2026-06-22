//! Optional image preprocessing, applied (when enabled) in the optimal order:
//!   1. resize down to an OCR-friendly size (≤ 2100×3000, aspect preserved)
//!   2. denoise (fast Non-Local Means, grayscale)
//!   3. deskew (Canny + Hough angle estimate, then rotate)
//!   4. binarize (Sauvola adaptive threshold)
//!
//! Everything is pure-Rust and parallelized with rayon to stay fast.

use crate::ocr::{resize_bilinear_bgr, ImageBgr};
use image::GrayImage;
use rayon::prelude::*;

/// Default "optimal" OCR canvas — larger images are scaled down to fit.
pub const MAX_W: usize = 2100;
pub const MAX_H: usize = 3000;

#[derive(Clone, Copy, Default)]
pub struct PreOpts {
    pub resize: bool,
    pub denoise: bool,
    pub deskew: bool,
    pub binarize: bool,
}

impl PreOpts {
    fn any_gray(&self) -> bool {
        self.denoise || self.deskew || self.binarize
    }
    pub fn any(&self) -> bool {
        self.resize || self.any_gray()
    }
}

/// Maps a point/box from preprocessed-image coordinates back to the original
/// input image coordinates (so returned bounds line up with the user's image).
#[derive(Clone, Copy)]
pub struct Transform {
    sx: f64, // resized -> original scale (origW / resizedW)
    sy: f64,
    deskew: Option<DeskewInv>,
}

#[derive(Clone, Copy)]
struct DeskewInv {
    angle_applied_deg: f32, // angle passed to rotate_gray (= -detected skew)
    rw: f32,                // dims before rotation (resized)
    rh: f32,
    dw: f32, // dims after rotation (deskewed canvas)
    dh: f32,
}

impl Transform {
    pub fn identity() -> Self {
        Transform { sx: 1.0, sy: 1.0, deskew: None }
    }

    fn map_point(&self, x: f64, y: f64) -> (f64, f64) {
        let (mut px, mut py) = (x, y);
        // 1) undo deskew rotation (deskewed -> resized)
        if let Some(d) = &self.deskew {
            let theta = (d.angle_applied_deg as f64).to_radians();
            let (s, c) = theta.sin_cos();
            let dx = px - d.dw as f64 / 2.0;
            let dy = py - d.dh as f64 / 2.0;
            px = c * dx + s * dy + d.rw as f64 / 2.0;
            py = -s * dx + c * dy + d.rh as f64 / 2.0;
        }
        // 2) undo resize (resized -> original)
        (px * self.sx, py * self.sy)
    }

    /// Map an axis-aligned [left, top, right, bottom] box back to the original
    /// image (bounding box of the mapped corners).
    pub fn map_box(&self, b: [i32; 4]) -> [i32; 4] {
        if self.deskew.is_none() && self.sx == 1.0 && self.sy == 1.0 {
            return b;
        }
        let corners = [
            (b[0] as f64, b[1] as f64),
            (b[2] as f64, b[1] as f64),
            (b[2] as f64, b[3] as f64),
            (b[0] as f64, b[3] as f64),
        ];
        let (mut minx, mut miny) = (f64::INFINITY, f64::INFINITY);
        let (mut maxx, mut maxy) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for (x, y) in corners {
            let (mx, my) = self.map_point(x, y);
            minx = minx.min(mx);
            miny = miny.min(my);
            maxx = maxx.max(mx);
            maxy = maxy.max(my);
        }
        [minx.round() as i32, miny.round() as i32, maxx.round() as i32, maxy.round() as i32]
    }
}

/// Apply the enabled preprocessing steps to a BGR image, in optimal order.
/// Returns the processed image and a transform mapping its coordinates back to
/// the original image (so bounds stay aligned with the user's input).
pub fn preprocess(img: ImageBgr, o: &PreOpts) -> (ImageBgr, Transform) {
    let (orig_w, orig_h) = (img.w, img.h);
    let mut img = img;

    // 1) resize (still BGR / color)
    if o.resize {
        img = resize_if_large(img, MAX_W, MAX_H);
    }
    let sx = orig_w as f64 / img.w as f64;
    let sy = orig_h as f64 / img.h as f64;
    let mut transform = Transform { sx, sy, deskew: None };

    if !o.any_gray() {
        return (img, transform);
    }

    // 2..4 operate on grayscale
    let (w, h) = (img.w, img.h);
    let mut gray = bgr_to_gray(&img.data, w, h);
    let mut gw = w;
    let mut gh = h;

    if o.denoise {
        gray = nlm_denoise(&gray, gw, gh);
    }
    if o.deskew {
        if let Some(angle) = skew_angle(&gray, gw, gh) {
            if angle.abs() > 0.1 {
                // rotate by the negative of the detected skew to undo it
                let applied = -angle;
                let (rg, rw, rh) = rotate_gray(gray, gw, gh, applied);
                transform.deskew = Some(DeskewInv {
                    angle_applied_deg: applied,
                    rw: gw as f32,
                    rh: gh as f32,
                    dw: rw as f32,
                    dh: rh as f32,
                });
                gray = rg;
                gw = rw;
                gh = rh;
            }
        }
    }
    if o.binarize {
        gray = sauvola(&gray, gw, gh);
    }

    (
        ImageBgr {
            w: gw,
            h: gh,
            data: gray_to_bgr(&gray),
        },
        transform,
    )
}

fn resize_if_large(img: ImageBgr, max_w: usize, max_h: usize) -> ImageBgr {
    if img.w <= max_w && img.h <= max_h {
        return img;
    }
    let scale = (max_w as f64 / img.w as f64).min(max_h as f64 / img.h as f64);
    let nw = ((img.w as f64 * scale).round() as usize).max(1);
    let nh = ((img.h as f64 * scale).round() as usize).max(1);
    let data = resize_bilinear_bgr(&img.data, img.w, img.h, nw, nh);
    ImageBgr { w: nw, h: nh, data }
}

fn bgr_to_gray(bgr: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut g = vec![0u8; w * h];
    g.par_iter_mut().enumerate().for_each(|(i, px)| {
        let b = bgr[i * 3] as f32;
        let gr = bgr[i * 3 + 1] as f32;
        let r = bgr[i * 3 + 2] as f32;
        *px = (0.114 * b + 0.587 * gr + 0.299 * r).round().clamp(0.0, 255.0) as u8;
    });
    g
}

fn gray_to_bgr(gray: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; gray.len() * 3];
    out.par_chunks_mut(3).zip(gray.par_iter()).for_each(|(px, &g)| {
        px[0] = g;
        px[1] = g;
        px[2] = g;
    });
    out
}

// ---------------- Sauvola binarization (integral images) ----------------

fn sauvola(gray: &[u8], w: usize, h: usize) -> Vec<u8> {
    const K: f64 = 0.2;
    const R: f64 = 128.0;
    let radius: i64 = 12; // ~25px window

    // integral images of value and value^2 (size (h+1)*(w+1))
    let iw = w + 1;
    let mut sum = vec![0f64; iw * (h + 1)];
    let mut sq = vec![0f64; iw * (h + 1)];
    for y in 0..h {
        let mut row_s = 0f64;
        let mut row_q = 0f64;
        for x in 0..w {
            let v = gray[y * w + x] as f64;
            row_s += v;
            row_q += v * v;
            sum[(y + 1) * iw + (x + 1)] = sum[y * iw + (x + 1)] + row_s;
            sq[(y + 1) * iw + (x + 1)] = sq[y * iw + (x + 1)] + row_q;
        }
    }
    let box_sum = |arr: &[f64], x0: i64, y0: i64, x1: i64, y1: i64| -> f64 {
        let x0 = x0.clamp(0, w as i64) as usize;
        let y0 = y0.clamp(0, h as i64) as usize;
        let x1 = (x1 + 1).clamp(0, w as i64) as usize;
        let y1 = (y1 + 1).clamp(0, h as i64) as usize;
        arr[y1 * iw + x1] - arr[y0 * iw + x1] - arr[y1 * iw + x0] + arr[y0 * iw + x0]
    };

    let mut out = vec![0u8; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let yi = y as i64;
        let y0 = yi - radius;
        let y1 = yi + radius;
        let yc0 = y0.clamp(0, h as i64 - 1);
        let yc1 = y1.clamp(0, h as i64 - 1);
        for x in 0..w {
            let xi = x as i64;
            let x0 = (xi - radius).clamp(0, w as i64 - 1);
            let x1 = (xi + radius).clamp(0, w as i64 - 1);
            let area = ((x1 - x0 + 1) * (yc1 - yc0 + 1)) as f64;
            let s = box_sum(&sum, x0, yc0, x1, yc1);
            let q = box_sum(&sq, x0, yc0, x1, yc1);
            let mean = s / area;
            let var = (q / area - mean * mean).max(0.0);
            let std = var.sqrt();
            let thresh = mean * (1.0 + K * (std / R - 1.0));
            row[x] = if (gray[y * w + x] as f64) > thresh { 255 } else { 0 };
        }
    });
    out
}

// ---------------- Fast Non-Local Means denoising ----------------
// Integral-image patch distances + a precomputed weight LUT (no per-pixel exp).

fn nlm_denoise(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let n = w * h;
    if n == 0 {
        return src.to_vec();
    }
    let tr: i64 = 2; // template radius (5x5)
    let sr: i64 = 3; // search radius (7x7) — plenty for OCR, keeps it fast
    let hh: f64 = 12.0; // filter strength
    let h2 = hh * hh;

    // weight LUT over normalized mean-squared patch distance
    const LUT_N: usize = 4096;
    const MAX_NORM: f64 = 65025.0; // 255^2
    let lut: Vec<f32> = (0..LUT_N)
        .map(|i| {
            let d = i as f64 * MAX_NORM / LUT_N as f64;
            (-d / h2).exp() as f32
        })
        .collect();

    let srcf: Vec<f32> = src.iter().map(|&v| v as f32).collect();
    let mut accum = vec![0f32; n];
    let mut wsum = vec![0f32; n];

    let tru = tr as usize;
    // per-axis valid window sizes (template box, clamped at borders)
    let count_h: Vec<f32> = (0..w)
        .map(|x| ((x + tru).min(w - 1) - x.saturating_sub(tru) + 1) as f32)
        .collect();
    let count_v: Vec<f32> = (0..h)
        .map(|y| ((y + tru).min(h - 1) - y.saturating_sub(tru) + 1) as f32)
        .collect();

    // reusable scratch buffers
    let mut diff = vec![0f32; n];
    let mut hsum = vec![0f32; n]; // horizontal box sum of diff
    let mut psum = vec![0f32; n]; // full template box sum (patch distance, unnormalized)

    for dy in -sr..=sr {
        for dx in -sr..=sr {
            // 1) squared diff vs the (dx,dy) shift (parallel over rows)
            diff.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                let sy = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;
                for x in 0..w {
                    let sx = (x as i64 + dx).clamp(0, w as i64 - 1) as usize;
                    let d = srcf[y * w + x] - srcf[sy * w + sx];
                    row[x] = d * d;
                }
            });
            // 2) horizontal box sum via an alloc-free sliding window (parallel over rows)
            hsum.par_chunks_mut(w).zip(diff.par_chunks(w)).for_each(|(hrow, drow)| {
                let mut s = 0f32;
                for k in 0..=tru.min(w - 1) {
                    s += drow[k];
                }
                hrow[0] = s;
                for x in 1..w {
                    let add = x + tru;
                    if add < w {
                        s += drow[add];
                    }
                    if x > tru {
                        s -= drow[x - tru - 1];
                    }
                    hrow[x] = s;
                }
            });
            // 3) vertical box sum over hsum -> full patch sum (parallel over rows)
            psum.par_chunks_mut(w).enumerate().for_each(|(y, prow)| {
                let lo = y.saturating_sub(tru);
                let hi = (y + tru).min(h - 1);
                prow.copy_from_slice(&hsum[lo * w..(lo + 1) * w]);
                for yy in (lo + 1)..=hi {
                    let hr = &hsum[yy * w..(yy + 1) * w];
                    for x in 0..w {
                        prow[x] += hr[x];
                    }
                }
            });
            // 4) accumulate weighted contributions (parallel over rows)
            accum
                .par_chunks_mut(w)
                .zip(wsum.par_chunks_mut(w))
                .enumerate()
                .for_each(|(y, (arow, wrow))| {
                    let cv = count_v[y];
                    let sy = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;
                    for x in 0..w {
                        let area = count_h[x] * cv;
                        let norm = psum[y * w + x] / area; // mean squared diff
                        let idx = ((norm * (LUT_N as f32 / MAX_NORM as f32)) as usize).min(LUT_N - 1);
                        let weight = lut[idx];
                        let sx = (x as i64 + dx).clamp(0, w as i64 - 1) as usize;
                        arow[x] += weight * srcf[sy * w + sx];
                        wrow[x] += weight;
                    }
                });
        }
    }

    let mut out = vec![0u8; n];
    out.par_iter_mut().enumerate().for_each(|(i, px)| {
        let v = if wsum[i] > 0.0 { accum[i] / wsum[i] } else { srcf[i] };
        *px = v.round().clamp(0.0, 255.0) as u8;
    });
    out
}

// ---------------- Deskew (Canny + Hough) ----------------

/// Estimate the skew angle (degrees, positive = counter-clockwise correction)
/// from the dominant near-axis-aligned Hough lines. Returns None if undetermined.
fn skew_angle(gray: &[u8], w: usize, h: usize) -> Option<f32> {
    use imageproc::hough::{detect_lines, LineDetectionOptions};

    // Work on a downscaled copy for speed (angle is scale-invariant).
    let target = 1000usize;
    let (dw, dh, small) = if w > target {
        let scale = target as f64 / w as f64;
        let nw = (w as f64 * scale).round() as usize;
        let nh = (h as f64 * scale).round() as usize;
        (nw, nh, downscale_gray(gray, w, h, nw, nh))
    } else {
        (w, h, gray.to_vec())
    };

    let gi = GrayImage::from_raw(dw as u32, dh as u32, small)?;
    let blurred = imageproc::filter::gaussian_blur_f32(&gi, 1.0);
    let edges = imageproc::edges::canny(&blurred, 50.0, 150.0);

    let vote_threshold = ((dw.min(dh) as f32) * 0.25).max(60.0) as u32;
    let lines = detect_lines(
        &edges,
        LineDetectionOptions {
            vote_threshold,
            suppression_radius: 9,
        },
    );
    if lines.is_empty() {
        return None;
    }

    // Map each line's normal angle to a deviation from the nearest axis in (-45,45].
    const LIMIT: f32 = 20.0;
    let mut devs: Vec<f32> = Vec::new();
    for l in lines.iter().take(80) {
        let a = l.angle_in_degrees as f32;
        let mut phi = ((a + 45.0) % 90.0) - 45.0;
        if phi <= -45.0 {
            phi += 90.0;
        }
        if phi.abs() <= LIMIT {
            devs.push(phi);
        }
    }
    if devs.is_empty() {
        return None;
    }
    // Histogram-mode (robust to outliers): pick the densest 1° bin, then average
    // the deviations within ±1.5° of it for sub-degree precision.
    let nbins = (2.0 * LIMIT) as usize + 1;
    let mut hist = vec![0u32; nbins];
    for &d in &devs {
        let b = (d + LIMIT).round().clamp(0.0, (nbins - 1) as f32) as usize;
        hist[b] += 1;
    }
    let peak = hist.iter().enumerate().max_by_key(|(_, &c)| c).map(|(i, _)| i as f32 - LIMIT)?;
    let near: Vec<f32> = devs.iter().cloned().filter(|d| (d - peak).abs() <= 1.5).collect();
    let angle = if near.is_empty() {
        peak
    } else {
        near.iter().sum::<f32>() / near.len() as f32
    };
    if std::env::var("OCR_DEBUG").is_ok() {
        eprintln!("[dbg] deskew detected angle = {angle:.2}° (from {} lines)", devs.len());
    }
    Some(angle)
}

fn downscale_gray(src: &[u8], w: usize, h: usize, nw: usize, nh: usize) -> Vec<u8> {
    let mut out = vec![0u8; nw * nh];
    let sx = w as f64 / nw as f64;
    let sy = h as f64 / nh as f64;
    out.par_chunks_mut(nw).enumerate().for_each(|(y, row)| {
        let oy = ((y as f64 + 0.5) * sy).min(h as f64 - 1.0) as usize;
        for x in 0..nw {
            let ox = ((x as f64 + 0.5) * sx).min(w as f64 - 1.0) as usize;
            row[x] = src[oy * w + ox];
        }
    });
    out
}

/// Rotate a grayscale image by `angle_deg` (counter-clockwise), keeping the full
/// content by expanding the canvas; background filled white.
fn rotate_gray(gray: Vec<u8>, w: usize, h: usize, angle_deg: f32) -> (Vec<u8>, usize, usize) {
    let theta = angle_deg.to_radians();
    let (s, c) = theta.sin_cos();
    // expanded canvas that fits the rotated image
    let nw = ((w as f32 * c.abs()) + (h as f32 * s.abs())).ceil() as usize;
    let nh = ((w as f32 * s.abs()) + (h as f32 * c.abs())).ceil() as usize;
    let nw = nw.max(1);
    let nh = nh.max(1);

    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let ncx = nw as f32 / 2.0;
    let ncy = nh as f32 / 2.0;

    let mut out = vec![255u8; nw * nh];
    out.par_chunks_mut(nw).enumerate().for_each(|(oy, row)| {
        let dy = oy as f32 - ncy;
        for ox in 0..nw {
            let dx = ox as f32 - ncx;
            // inverse rotation (by -theta) to sample source
            let srcx = c * dx + s * dy + cx;
            let srcy = -s * dx + c * dy + cy;
            if srcx < 0.0 || srcy < 0.0 || srcx > (w - 1) as f32 || srcy > (h - 1) as f32 {
                continue; // leave white
            }
            // bilinear
            let x0 = srcx.floor() as usize;
            let y0 = srcy.floor() as usize;
            let x1 = (x0 + 1).min(w - 1);
            let y1 = (y0 + 1).min(h - 1);
            let ax = srcx - x0 as f32;
            let ay = srcy - y0 as f32;
            let p00 = gray[y0 * w + x0] as f32;
            let p01 = gray[y0 * w + x1] as f32;
            let p10 = gray[y1 * w + x0] as f32;
            let p11 = gray[y1 * w + x1] as f32;
            let top = p00 * (1.0 - ax) + p01 * ax;
            let bot = p10 * (1.0 - ax) + p11 * ax;
            row[ox] = (top * (1.0 - ay) + bot * ay).round().clamp(0.0, 255.0) as u8;
        }
    });
    (out, nw, nh)
}
