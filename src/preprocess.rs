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

    let dbg = std::env::var("OCR_DEBUG").is_ok();
    // 2..4 operate on grayscale
    let (w, h) = (img.w, img.h);
    let mut gray = bgr_to_gray(&img.data, w, h);
    let mut gw = w;
    let mut gh = h;

    if o.denoise {
        let t = std::time::Instant::now();
        gray = nlm_denoise(&gray, gw, gh);
        if dbg {
            eprintln!("[dbg] denoise ({}x{}): {:.3}s", gw, gh, t.elapsed().as_secs_f64());
        }
    }
    if o.deskew {
        let t = std::time::Instant::now();
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
        if dbg {
            eprintln!("[dbg] deskew: {:.3}s", t.elapsed().as_secs_f64());
        }
    }
    if o.binarize {
        let t = std::time::Instant::now();
        gray = sauvola(&gray, gw, gh);
        if dbg {
            eprintln!("[dbg] binarize ({}x{}): {:.3}s", gw, gh, t.elapsed().as_secs_f64());
        }
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

// ---------------- Sauvola binarization ----------------
// Cache-resident strips + separable running-sum box for the windowed mean and
// mean-of-squares (no global integral image / sequential build). General win.

fn sauvola(gray: &[u8], w: usize, h: usize) -> Vec<u8> {
    const K: f64 = 0.2;
    const R: f64 = 128.0;
    const RAD: usize = 12; // ~25px window
    if w == 0 || h == 0 {
        return gray.to_vec();
    }

    // valid (clamped) window sizes per axis
    let count_h: Vec<f64> = (0..w)
        .map(|x| ((x + RAD).min(w - 1) - x.saturating_sub(RAD) + 1) as f64)
        .collect();
    let count_v: Vec<f64> = (0..h)
        .map(|y| ((y + RAD).min(h - 1) - y.saturating_sub(RAD) + 1) as f64)
        .collect();

    // Cache-resident strips, separable running-sum box (mean + mean-of-squares),
    // parallel over strips — no global integral image, no sequential build.
    let strip_h = 128usize;
    let mut out = vec![0u8; w * h];
    out.par_chunks_mut(strip_h * w).enumerate().for_each(|(si, ostrip)| {
        let y0 = si * strip_h;
        let rows = ostrip.len() / w;
        let y1 = y0 + rows;
        let band_y0 = y0.saturating_sub(RAD);
        let band_y1 = (y1 + RAD).min(h);
        let band_rows = band_y1 - band_y0;
        // horizontal box sums of value and value^2 for the band
        let mut hs = vec![0f64; band_rows * w];
        let mut hq = vec![0f64; band_rows * w];
        for br in 0..band_rows {
            let g = &gray[(band_y0 + br) * w..(band_y0 + br + 1) * w];
            let (hsr, hqr) = (&mut hs[br * w..br * w + w], &mut hq[br * w..br * w + w]);
            let mut s = 0f64;
            let mut q = 0f64;
            for k in 0..=RAD.min(w - 1) {
                let v = g[k] as f64;
                s += v;
                q += v * v;
            }
            hsr[0] = s;
            hqr[0] = q;
            for x in 1..w {
                if x + RAD < w {
                    let v = g[x + RAD] as f64;
                    s += v;
                    q += v * v;
                }
                if x > RAD {
                    let v = g[x - RAD - 1] as f64;
                    s -= v;
                    q -= v * v;
                }
                hsr[x] = s;
                hqr[x] = q;
            }
        }
        // vertical sum over the cached band + Sauvola threshold
        for oy in 0..rows {
            let y = y0 + oy;
            let v_lo = y.saturating_sub(RAD);
            let v_hi = (y + RAD).min(h - 1);
            let cv = count_v[y];
            let orow = &mut ostrip[oy * w..oy * w + w];
            let grow = &gray[y * w..y * w + w];
            for x in 0..w {
                let mut s = 0f64;
                let mut q = 0f64;
                for r in v_lo..=v_hi {
                    let idx = (r - band_y0) * w + x;
                    s += hs[idx];
                    q += hq[idx];
                }
                let area = count_h[x] * cv;
                let mean = s / area;
                let var = (q / area - mean * mean).max(0.0);
                let std = var.sqrt();
                let thresh = mean * (1.0 + K * (std / R - 1.0));
                orow[x] = if (grow[x] as f64) > thresh { 255 } else { 0 };
            }
        }
    });
    out
}

// ---------------- Fast Non-Local Means denoising ----------------
// Separable box patch-distances + a precomputed weight LUT (no per-pixel exp).
// Processed in cache-resident horizontal strips with the search-offset loop
// *inside* each strip: the strip's pixels stay hot in cache across all offsets
// instead of streaming the whole image from RAM 49 times. This is a general
// cache-blocking win (benefits any CPU with a cache hierarchy) and is also
// embarrassingly parallel across strips.

const NLM_TR: usize = 2; // template radius (5x5)
const NLM_SR: i64 = 3; // search radius (7x7)
const NLM_H: f64 = 12.0; // filter strength
const NLM_LUT_N: usize = 4096;
const NLM_MAX_NORM: f64 = 65025.0; // 255^2

fn nlm_denoise(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let n = w * h;
    if n == 0 || w < 1 || h < 1 {
        return src.to_vec();
    }
    let tr = NLM_TR;
    let sr = NLM_SR;
    let h2 = NLM_H * NLM_H;
    let lut: Vec<f32> = (0..NLM_LUT_N)
        .map(|i| (-(i as f64 * NLM_MAX_NORM / NLM_LUT_N as f64) / h2).exp() as f32)
        .collect();
    let lut_scale = NLM_LUT_N as f32 / NLM_MAX_NORM as f32;

    let srcf: Vec<f32> = src.iter().map(|&v| v as f32).collect();
    // per-axis valid template window sizes (clamped at borders)
    let count_h: Vec<f32> = (0..w)
        .map(|x| ((x + tr).min(w - 1) - x.saturating_sub(tr) + 1) as f32)
        .collect();
    let count_v: Vec<f32> = (0..h)
        .map(|y| ((y + tr).min(h - 1) - y.saturating_sub(tr) + 1) as f32)
        .collect();

    let strip_h = 128usize; // keep a strip's working set hot in cache
    let mut out = vec![0u8; n];
    out.par_chunks_mut(strip_h * w).enumerate().for_each(|(si, ostrip)| {
        let y0 = si * strip_h;
        let rows = ostrip.len() / w;
        let y1 = y0 + rows;
        // band of source rows needed for the template box of output rows [y0,y1)
        let band_y0 = y0.saturating_sub(tr);
        let band_y1 = (y1 + tr).min(h);
        let band_rows = band_y1 - band_y0;

        let mut hband = vec![0f32; band_rows * w]; // horizontal box sums (per offset)
        let mut accum = vec![0f32; rows * w];
        let mut wsum = vec![0f32; rows * w];

        for dy in -sr..=sr {
            for dx in -sr..=sr {
                // 1) horizontal box sum of the squared diff, over the band rows
                for br in 0..band_rows {
                    let r = band_y0 + br;
                    let sr_row = (r as i64 + dy).clamp(0, h as i64 - 1) as usize;
                    let base = r * w;
                    let sbase = sr_row * w;
                    let diff = |x: usize| -> f32 {
                        let sx = (x as i64 + dx).clamp(0, w as i64 - 1) as usize;
                        let d = srcf[base + x] - srcf[sbase + sx];
                        d * d
                    };
                    let hb = br * w;
                    let mut s = 0f32;
                    for k in 0..=tr.min(w - 1) {
                        s += diff(k);
                    }
                    hband[hb] = s;
                    for x in 1..w {
                        let add = x + tr;
                        if add < w {
                            s += diff(add);
                        }
                        if x > tr {
                            s -= diff(x - tr - 1);
                        }
                        hband[hb + x] = s;
                    }
                }
                // 2) vertical box sum (over the cached hband) + weighted accumulate
                for oy in 0..rows {
                    let y = y0 + oy;
                    let v_lo = y.saturating_sub(tr);
                    let v_hi = (y + tr).min(h - 1);
                    let cv = count_v[y];
                    let sr_acc = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;
                    let arow = oy * w;
                    let sbase_acc = sr_acc * w;
                    for x in 0..w {
                        let mut ps = 0f32;
                        for r in v_lo..=v_hi {
                            ps += hband[(r - band_y0) * w + x];
                        }
                        let norm = ps / (count_h[x] * cv);
                        let idx = ((norm * lut_scale) as usize).min(NLM_LUT_N - 1);
                        let weight = lut[idx];
                        let sx = (x as i64 + dx).clamp(0, w as i64 - 1) as usize;
                        accum[arow + x] += weight * srcf[sbase_acc + sx];
                        wsum[arow + x] += weight;
                    }
                }
            }
        }
        // write strip output
        for i in 0..rows * w {
            let v = if wsum[i] > 0.0 { accum[i] / wsum[i] } else { srcf[y0 * w + i] };
            ostrip[i] = (v + 0.5).clamp(0.0, 255.0) as u8;
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nlm_preserves_uniform() {
        let (w, h) = (20, 15);
        let src = vec![128u8; w * h];
        let out = nlm_denoise(&src, w, h);
        assert!(out.iter().all(|&v| (v as i32 - 128).abs() <= 1));
    }

    #[test]
    fn nlm_strips_have_no_seams() {
        // taller than one strip (128): every row must be denoised consistently
        let (w, h) = (16, 320);
        let src: Vec<u8> = (0..w * h).map(|i| ((i * 13) % 256) as u8).collect();
        let out = nlm_denoise(&src, w, h);
        assert_eq!(out.len(), w * h);
        // a near-uniform input stays near its value at the strip boundary rows
        let flat = vec![200u8; w * h];
        let outf = nlm_denoise(&flat, w, h);
        for r in [126usize, 127, 128, 129] {
            for x in 0..w {
                assert!((outf[r * w + x] as i32 - 200).abs() <= 1, "seam at row {r}");
            }
        }
    }

    #[test]
    fn sauvola_uniform_is_foreground() {
        // uniform gray -> std 0 -> thresh = mean*(1-K) < mean -> all 255
        let (w, h) = (30, 20);
        let out = sauvola(&vec![100u8; w * h], w, h);
        assert!(out.iter().all(|&v| v == 255));
    }

    #[test]
    fn sauvola_output_is_binary() {
        let (w, h) = (40, 200); // spans multiple strips
        let src: Vec<u8> = (0..w * h).map(|i| ((i * 37) % 256) as u8).collect();
        let out = sauvola(&src, w, h);
        assert!(out.iter().all(|&v| v == 0 || v == 255));
    }
}
