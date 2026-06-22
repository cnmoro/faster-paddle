//! PP-OCRv6 tiny detection + recognition pipeline (CPU, ONNX Runtime).

use crate::cv;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

// ---- detection params (from PP-OCRv6_tiny_det inference.yml) ----
const DET_THRESH: f32 = 0.2;
const DET_BOX_THRESH: f32 = 0.4;
const DET_UNCLIP_RATIO: f64 = 1.4;
const DET_MAX_CANDIDATES: usize = 3000;
const DET_MIN_SIZE: f64 = 3.0;
const LIMIT_SIDE_LEN: i64 = 736;
const MAX_SIDE_LIMIT: i64 = 4000;
const DET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const DET_STD: [f32; 3] = [0.229, 0.224, 0.225];

const REC_H: usize = 48;
const REC_MAX_W: usize = 3200;
pub const DEFAULT_REC_BATCH: usize = 6;

pub struct OcrResult {
    pub text: String,
    pub score: f32,
    /// axis-aligned box [left, top, right, bottom]
    pub box4: [i32; 4],
}

pub struct Engine {
    det: Session,
    rec: Session,
    chars: Vec<String>,
    rec_batch: usize,
}

/// BGR u8 image.
pub struct ImageBgr {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

impl Engine {
    /// Build an engine from in-memory ONNX model bytes (models are embedded in
    /// the library, so no files are needed at runtime).
    pub fn from_memory(
        det_bytes: &[u8],
        rec_bytes: &[u8],
        char_dict: Vec<String>,
        threads: usize,
        rec_batch: usize,
    ) -> ort::Result<Self> {
        let det = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_memory(det_bytes)?;
        let rec = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_memory(rec_bytes)?;
        // CHARS = ["blank"] + dict + [" "]
        let mut chars = Vec::with_capacity(char_dict.len() + 2);
        chars.push("blank".to_string());
        chars.extend(char_dict);
        chars.push(" ".to_string());
        Ok(Engine {
            det,
            rec,
            chars,
            rec_batch: rec_batch.max(1),
        })
    }

    pub fn run(&mut self, img: &ImageBgr) -> ort::Result<Vec<OcrResult>> {
        let dbg = std::env::var("OCR_DEBUG").is_ok();
        let t0 = std::time::Instant::now();
        // ---------- detection ----------
        let (rw, rh) = det_resize_dims(img.w, img.h);
        let resized = resize_bilinear_bgr(&img.data, img.w, img.h, rw, rh);
        // normalize -> NCHW f32
        let mut input = vec![0f32; 3 * rh * rw];
        let plane = rh * rw;
        for y in 0..rh {
            for x in 0..rw {
                let i = (y * rw + x) * 3;
                for c in 0..3 {
                    let v = resized[i + c] as f32 / 255.0;
                    input[c * plane + y * rw + x] = (v - DET_MEAN[c]) / DET_STD[c];
                }
            }
        }
        let tensor = Tensor::from_array(([1usize, 3, rh, rw], input))?;
        let (pred, ph, pw) = {
            let outputs = self.det.run(ort::inputs!["x" => tensor])?;
            let (shape, pred) = outputs["fetch_name_0"].try_extract_tensor::<f32>()?;
            (pred.to_vec(), shape[2] as usize, shape[3] as usize)
        };

        if dbg {
            eprintln!("[dbg] det infer ({}x{}): {:.3}s", rw, rh, t0.elapsed().as_secs_f64());
        }
        let t1 = std::time::Instant::now();
        let boxes = db_postprocess(&pred, pw, ph, img.w, img.h);
        let boxes = sort_boxes(boxes);
        if dbg {
            eprintln!("[dbg] db_postprocess ({} boxes): {:.3}s", boxes.len(), t1.elapsed().as_secs_f64());
        }
        let t2 = std::time::Instant::now();

        // ---------- crops (parallel; independent per box) ----------
        use rayon::prelude::*;
        let cropped: Vec<Option<(ImageBgr, [cv::Pt; 4])>> = boxes
            .par_iter()
            .map(|b| crop_quad(img, b).filter(|c| c.w > 0 && c.h > 0).map(|c| (c, *b)))
            .collect();
        let mut crops: Vec<ImageBgr> = Vec::with_capacity(boxes.len());
        let mut kept_boxes: Vec<[cv::Pt; 4]> = Vec::with_capacity(boxes.len());
        for item in cropped.into_iter().flatten() {
            crops.push(item.0);
            kept_boxes.push(item.1);
        }
        if crops.is_empty() {
            return Ok(vec![]);
        }
        if dbg {
            eprintln!("[dbg] crops ({}): {:.3}s", crops.len(), t2.elapsed().as_secs_f64());
        }
        let t3 = std::time::Instant::now();

        // sort by aspect ratio for batching
        let mut order: Vec<usize> = (0..crops.len()).collect();
        order.sort_by(|&a, &b| {
            let ra = crops[a].w as f64 / crops[a].h as f64;
            let rb = crops[b].w as f64 / crops[b].h as f64;
            ra.partial_cmp(&rb).unwrap()
        });

        let rec_batch = self.rec_batch;
        let mut texts: Vec<(String, f32)> = vec![(String::new(), 0.0); crops.len()];
        let mut s = 0;
        while s < order.len() {
            let chunk: Vec<usize> = order[s..(s + rec_batch).min(order.len())].to_vec();
            let decoded = self.rec_batch(&crops, &chunk)?;
            for (k, &idx) in chunk.iter().enumerate() {
                texts[idx] = decoded[k].clone();
            }
            s += rec_batch;
        }

        if dbg {
            eprintln!("[dbg] rec ({} crops): {:.3}s", crops.len(), t3.elapsed().as_secs_f64());
        }
        // assemble results (score_thresh = 0.0 -> keep all)
        let mut out = Vec::with_capacity(crops.len());
        for i in 0..crops.len() {
            let (t, sc) = texts[i].clone();
            let q = &kept_boxes[i];
            let left = q.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
            let right = q.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
            let top = q.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
            let bottom = q.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
            out.push(OcrResult {
                text: t,
                score: sc,
                box4: [left as i32, top as i32, right as i32, bottom as i32],
            });
        }
        Ok(out)
    }

    fn rec_batch(&mut self, crops: &[ImageBgr], idxs: &[usize]) -> ort::Result<Vec<(String, f32)>> {
        // compute max_wh_ratio across batch
        let mut max_wh = 320.0 / 48.0;
        for &i in idxs {
            let r = crops[i].w as f64 / crops[i].h as f64;
            if r > max_wh {
                max_wh = r;
            }
        }
        let mut img_w = (REC_H as f64 * max_wh) as usize;
        if img_w > REC_MAX_W {
            img_w = REC_MAX_W;
        }
        if img_w < 1 {
            img_w = 1;
        }
        let n = idxs.len();
        let plane = REC_H * img_w;
        let mut data = vec![0f32; n * 3 * plane];
        for (bi, &i) in idxs.iter().enumerate() {
            let c = &crops[i];
            // resized width
            let resized_w = if img_w >= REC_MAX_W && (REC_H as f64 * c.w as f64 / c.h as f64) as usize > REC_MAX_W {
                REC_MAX_W
            } else {
                let ratio = c.w as f64 / c.h as f64;
                let rw = (REC_H as f64 * ratio).ceil() as usize;
                rw.min(img_w).max(1)
            };
            let small = resize_bilinear_bgr(&c.data, c.w, c.h, resized_w, REC_H);
            let base = bi * 3 * plane;
            for y in 0..REC_H {
                for x in 0..resized_w {
                    let si = (y * resized_w + x) * 3;
                    for ch in 0..3 {
                        let v = small[si + ch] as f32 / 255.0;
                        let v = (v - 0.5) / 0.5;
                        data[base + ch * plane + y * img_w + x] = v;
                    }
                }
            }
        }
        let tensor = Tensor::from_array(([n, 3, REC_H, img_w], data))?;
        let (preds, t, cls) = {
            let outputs = self.rec.run(ort::inputs!["x" => tensor])?;
            let (shape, preds) = outputs["fetch_name_0"].try_extract_tensor::<f32>()?;
            (preds.to_vec(), shape[1] as usize, shape[2] as usize)
        };
        let mut res = Vec::with_capacity(n);
        for b in 0..n {
            res.push(self.ctc_decode(&preds[b * t * cls..(b + 1) * t * cls], t, cls));
        }
        Ok(res)
    }

    fn ctc_decode(&self, logits: &[f32], t: usize, cls: usize) -> (String, f32) {
        let mut last = usize::MAX;
        let mut s = String::new();
        let mut sum = 0.0f64;
        let mut cnt = 0u32;
        for ti in 0..t {
            let row = &logits[ti * cls..(ti + 1) * cls];
            let mut best = 0usize;
            let mut bestv = row[0];
            for (j, &v) in row.iter().enumerate() {
                if v > bestv {
                    bestv = v;
                    best = j;
                }
            }
            // remove duplicates + blank
            if best != last {
                if best != 0 {
                    s.push_str(&self.chars[best]);
                    sum += bestv as f64;
                    cnt += 1;
                }
            }
            last = best;
        }
        let score = if cnt > 0 { (sum / cnt as f64) as f32 } else { 0.0 };
        (s, score)
    }
}

fn det_resize_dims(w: usize, h: usize) -> (usize, usize) {
    let (h, w) = (h as i64, w as i64);
    // limit_type = "min"
    let ratio = if w.min(h) < LIMIT_SIDE_LEN {
        LIMIT_SIDE_LEN as f64 / (if h < w { h } else { w }) as f64
    } else {
        1.0
    };
    let mut rh = (h as f64 * ratio) as i64;
    let mut rw = (w as f64 * ratio) as i64;
    if rh.max(rw) > MAX_SIDE_LIMIT {
        let r2 = MAX_SIDE_LIMIT as f64 / rh.max(rw) as f64;
        rh = (rh as f64 * r2) as i64;
        rw = (rw as f64 * r2) as i64;
    }
    rh = (((rh as f64 / 32.0).round() as i64) * 32).max(32);
    rw = (((rw as f64 / 32.0).round() as i64) * 32).max(32);
    (rw as usize, rh as usize)
}

/// cv2 INTER_LINEAR-style bilinear resize for interleaved BGR u8.
pub fn resize_bilinear_bgr(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let mut out = vec![0u8; dw * dh * 3];
    let scale_x = sw as f64 / dw as f64;
    let scale_y = sh as f64 / dh as f64;
    for y in 0..dh {
        let sy = ((y as f64 + 0.5) * scale_y - 0.5).max(0.0);
        let y0 = sy.floor();
        let ay = sy - y0;
        let y0i = (y0 as i64).clamp(0, sh as i64 - 1) as usize;
        let y1i = (y0i + 1).min(sh - 1);
        for x in 0..dw {
            let sx = ((x as f64 + 0.5) * scale_x - 0.5).max(0.0);
            let x0 = sx.floor();
            let ax = sx - x0;
            let x0i = (x0 as i64).clamp(0, sw as i64 - 1) as usize;
            let x1i = (x0i + 1).min(sw - 1);
            let o = (y * dw + x) * 3;
            for c in 0..3 {
                let p00 = src[(y0i * sw + x0i) * 3 + c] as f64;
                let p01 = src[(y0i * sw + x1i) * 3 + c] as f64;
                let p10 = src[(y1i * sw + x0i) * 3 + c] as f64;
                let p11 = src[(y1i * sw + x1i) * 3 + c] as f64;
                let top = p00 * (1.0 - ax) + p01 * ax;
                let bot = p10 * (1.0 - ax) + p11 * ax;
                out[o + c] = (top * (1.0 - ay) + bot * ay).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// DB post-process. Returns quad boxes in source-image coordinates.
fn db_postprocess(pred: &[f32], pw: usize, ph: usize, src_w: usize, src_h: usize) -> Vec<[cv::Pt; 4]> {
    // binary map (8-connected flood fill)
    let mut fg = vec![false; pw * ph];
    for i in 0..pw * ph {
        fg[i] = pred[i] > DET_THRESH;
    }
    let mut visited = vec![false; pw * ph];
    let width_scale = src_w as f64 / pw as f64;
    let height_scale = src_h as f64 / ph as f64;
    let mut boxes = Vec::new();
    let mut stack: Vec<(i32, i32)> = Vec::new();
    let mut comp: Vec<cv::Pt> = Vec::new();
    for sy in 0..ph {
        for sx in 0..pw {
            let idx = sy * pw + sx;
            if !fg[idx] || visited[idx] {
                continue;
            }
            if boxes.len() >= DET_MAX_CANDIDATES {
                break;
            }
            // flood fill component
            comp.clear();
            stack.clear();
            stack.push((sx as i32, sy as i32));
            visited[idx] = true;
            while let Some((cx, cy)) = stack.pop() {
                comp.push((cx as f64, cy as f64));
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = cx + dx;
                        let ny = cy + dy;
                        if nx < 0 || ny < 0 || nx >= pw as i32 || ny >= ph as i32 {
                            continue;
                        }
                        let nidx = ny as usize * pw + nx as usize;
                        if fg[nidx] && !visited[nidx] {
                            visited[nidx] = true;
                            stack.push((nx, ny));
                        }
                    }
                }
            }
            if comp.len() < 4 {
                continue;
            }
            // minAreaRect on component
            let (box1, side1) = cv::min_area_rect(&comp);
            if side1 < DET_MIN_SIZE {
                continue;
            }
            let score = cv::box_score_fast(pred, pw, ph, &box1);
            if DET_BOX_THRESH > score {
                continue;
            }
            // unclip
            let area = cv::poly_area(&box1);
            let perim = cv::poly_perimeter(&box1);
            if perim < 1e-6 {
                continue;
            }
            let dist = area * DET_UNCLIP_RATIO / perim;
            let box2 = cv::unclip_rect(&box1, dist);
            let (box3, side3) = cv::min_area_rect(&box2);
            if side3 < DET_MIN_SIZE + 2.0 {
                continue;
            }
            // scale to source
            let mut scaled = [(0.0, 0.0); 4];
            for i in 0..4 {
                let x = (box3[i].0 * width_scale).round().clamp(0.0, src_w as f64);
                let y = (box3[i].1 * height_scale).round().clamp(0.0, src_h as f64);
                scaled[i] = (x, y);
            }
            boxes.push(scaled);
        }
    }
    boxes
}

/// Replicates SortQuadBoxes: top-to-bottom, left-to-right.
fn sort_boxes(mut boxes: Vec<[cv::Pt; 4]>) -> Vec<[cv::Pt; 4]> {
    boxes.sort_by(|a, b| {
        a[0].1
            .partial_cmp(&b[0].1)
            .unwrap()
            .then(a[0].0.partial_cmp(&b[0].0).unwrap())
    });
    let n = boxes.len();
    for i in 0..n.saturating_sub(1) {
        let mut j = i as i64;
        while j >= 0 {
            let ju = j as usize;
            if (boxes[ju + 1][0].1 - boxes[ju][0].1).abs() < 10.0 && boxes[ju + 1][0].0 < boxes[ju][0].0 {
                boxes.swap(ju, ju + 1);
                j -= 1;
            } else {
                break;
            }
        }
    }
    boxes
}

/// get_minarea_rect_crop + get_rotate_crop_image.
fn crop_quad(img: &ImageBgr, quad: &[cv::Pt; 4]) -> Option<ImageBgr> {
    // get_minarea_rect_crop: minAreaRect of the (already rectangular) quad, then order points.
    let (rect, _side) = cv::min_area_rect(quad);
    // order points like get_minarea_rect_crop: sort by x, pick a/b/c/d
    let mut pts: Vec<cv::Pt> = rect.to_vec();
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (index_a, index_d) = if pts[1].1 > pts[0].1 { (0, 1) } else { (1, 0) };
    let (index_b, index_c) = if pts[3].1 > pts[2].1 { (2, 3) } else { (3, 2) };
    let ordered = [pts[index_a], pts[index_b], pts[index_c], pts[index_d]];
    // crop width/height (get_rotate_crop_image)
    let dist = |p: cv::Pt, q: cv::Pt| ((p.0 - q.0).powi(2) + (p.1 - q.1).powi(2)).sqrt();
    let cw = dist(ordered[0], ordered[1]).max(dist(ordered[2], ordered[3])) as usize;
    let ch = dist(ordered[0], ordered[3]).max(dist(ordered[1], ordered[2])) as usize;
    if cw == 0 || ch == 0 {
        return None;
    }
    let crop = cv::warp_crop(&img.data, img.w, img.h, &ordered, cw, ch);
    let (data, w, h) = if ch as f64 / cw as f64 >= 1.5 {
        cv::rot90_ccw(&crop, cw, ch)
    } else {
        (crop, cw, ch)
    };
    Some(ImageBgr { w, h, data })
}
