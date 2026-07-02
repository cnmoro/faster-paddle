//! PP-OCRv6 tiny detection + recognition pipeline (CPU, ONNX Runtime).

use crate::cv;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

// ---- detection params (from PP-OCRv6 *_det inference.yml) ----
const DET_THRESH: f32 = 0.2;
const DET_UNCLIP_RATIO: f64 = 1.4;
const DET_MAX_CANDIDATES: usize = 3000;
const DET_MIN_SIZE: f64 = 3.0;
const LIMIT_SIDE_LEN: i64 = 736;
const MAX_SIDE_LIMIT: i64 = 4000;
/// Default cap on the detector's longer side. PaddleOCR runs detection at up to
/// 4000px, but the tiny detector locates text just as well at ~1600px (recognition
/// still crops from the full-res image, so text stays sharp) — ~2x faster with
/// negligible quality loss. Raise toward 4000 for microscopic text.
pub const DEFAULT_DET_MAX_SIDE: i64 = 1600;
const DET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const DET_STD: [f32; 3] = [0.229, 0.224, 0.225];

const REC_H: usize = 48;
const REC_MAX_W: usize = 3200;
// Small batches minimise width padding; the rec session pool supplies the
// parallelism, so a small batch is fastest.
pub const DEFAULT_REC_BATCH: usize = 4;
/// Recognition batch budget: a batch grows until `count * max_rec_width` exceeds
/// this, so narrow crops batch together while wide line-crops run nearly alone
/// (avoiding wasted padding compute). Tuned empirically: with one session per
/// physical core, small batches (~1-2 average-width crops) minimise latency.
pub const REC_BATCH_BUDGET: usize = 800;

pub struct OcrResult {
    pub text: String,
    pub score: f32,
    /// axis-aligned box [left, top, right, bottom]
    pub box4: [i32; 4],
}

pub struct Engine {
    det: Session,
    /// Pool of recognition sessions, run concurrently across batches so the many
    /// small rec matmuls keep all cores busy (a single session under-utilizes them).
    rec: Vec<Session>,
    chars: Vec<String>,
    rec_batch: usize,
    box_thresh: f32,
    det_max_side: i64,
}

/// Interleaved RGB u8 image. The PP-OCR networks expect BGR channel *planes*
/// (cv2.imread order); the flip happens at NCHW tensor-fill time (plane `c`
/// reads interleaved channel `2 - c`), so no pixel-swap pass is ever needed.
pub struct ImageRgb {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

impl Engine {
    /// Build an engine from in-memory ONNX model bytes (models are embedded in
    /// the library, so no files are needed at runtime).
    #[allow(clippy::too_many_arguments)]
    pub fn from_memory(
        det_bytes: &[u8],
        rec_bytes: &[u8],
        char_dict: Vec<String>,
        threads: usize,
        det_threads: usize,
        rec_batch: usize,
        box_thresh: f32,
        rec_pool: usize,
        det_max_side: i64,
    ) -> ort::Result<Self> {
        let mempat = std::env::var("OCR_MEMPAT").map(|v| v != "0").unwrap_or(false);
        let prepack = std::env::var("OCR_PREPACK").map(|v| v != "0").unwrap_or(true);
        let det_spin = std::env::var("OCR_DET_SPIN").map(|v| v != "0").unwrap_or(false);
        let build = |bytes: &[u8], t: usize, spin: bool, pw: Option<&ort::session::builder::PrepackedWeights>| -> ort::Result<Session> {
            let mut b = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_memory_pattern(mempat)?
                .with_intra_op_spinning(spin)?
                .with_intra_threads(t.max(1))?;
            if let Some(pw) = pw {
                b = b.with_prepacked_weights(pw)?;
            }
            b.commit_from_memory(bytes)
        };
        // det spinning off: its (many) pool threads would otherwise keep
        // spin-waiting after det.run() returns, stealing CPU from the rec
        // workers that start right after.
        let det = build(det_bytes, det_threads.max(1), det_spin, None)?;
        // Pool of rec sessions; split the threads across them so concurrent
        // batches together saturate the cores. All pool sessions share one
        // prepacked-weights container so the packed weight buffers exist once,
        // not `pool` times (less memory, better cache reuse).
        let pool = rec_pool.clamp(1, threads.max(1));
        let per = (threads / pool).max(1);
        let shared = ort::session::builder::PrepackedWeights::new();
        let mut rec = Vec::with_capacity(pool);
        for _ in 0..pool {
            rec.push(build(rec_bytes, per, true, prepack.then_some(&shared))?);
        }
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
            box_thresh,
            det_max_side: if det_max_side <= 0 { MAX_SIDE_LIMIT } else { det_max_side },
        })
    }

    pub fn run(&mut self, img: &ImageRgb) -> ort::Result<Vec<OcrResult>> {
        let dbg = std::env::var("OCR_DEBUG").is_ok();
        let t0 = std::time::Instant::now();
        // ---------- detection ----------
        use rayon::prelude::*;
        let (rw, rh) = det_resize_dims(img.w, img.h, self.det_max_side);
        let tpr = std::time::Instant::now();
        let resized = resize_bilinear_rgb(&img.data, img.w, img.h, rw, rh);
        if dbg {
            eprintln!("[dbg]   det resize: {:.3}s", tpr.elapsed().as_secs_f64());
        }
        let tpn = std::time::Instant::now();
        // normalize -> NCHW f32 (parallel over the 3 channel planes; precomputed
        // scale/bias avoids per-pixel divisions)
        let plane = rh * rw;
        let mut input = vec![0f32; 3 * plane];
        let alpha = [
            1.0 / (255.0 * DET_STD[0]),
            1.0 / (255.0 * DET_STD[1]),
            1.0 / (255.0 * DET_STD[2]),
        ];
        let beta = [
            -DET_MEAN[0] / DET_STD[0],
            -DET_MEAN[1] / DET_STD[1],
            -DET_MEAN[2] / DET_STD[2],
        ];
        input
            .par_chunks_mut(plane)
            .enumerate()
            .for_each(|(c, ch)| {
                // plane c is B,G,R -> interleaved RGB channel 2-c
                let sc = 2 - c;
                for px in 0..plane {
                    ch[px] = resized[px * 3 + sc] as f32 * alpha[c] + beta[c];
                }
            });
        if dbg {
            eprintln!("[dbg]   det normalize: {:.3}s", tpn.elapsed().as_secs_f64());
        }
        let tinf = std::time::Instant::now();
        let tensor = Tensor::from_array(([1usize, 3, rh, rw], input))?;
        let box_thresh = self.box_thresh;
        // post-process directly on the borrowed output tensor (the prob map is
        // several MB; no need to copy it out)
        let t1;
        let boxes = {
            let outputs = self.det.run(ort::inputs!["x" => tensor])?;
            let (shape, pred) = outputs["fetch_name_0"].try_extract_tensor::<f32>()?;
            let (ph, pw) = (shape[2] as usize, shape[3] as usize);
            if dbg {
                eprintln!("[dbg]   det ORT infer: {:.3}s", tinf.elapsed().as_secs_f64());
                eprintln!("[dbg] det total ({}x{}): {:.3}s", rw, rh, t0.elapsed().as_secs_f64());
            }
            t1 = std::time::Instant::now();
            db_postprocess(pred, pw, ph, img.w, img.h, box_thresh)
        };
        let boxes = sort_boxes(boxes);
        if dbg {
            eprintln!("[dbg] db_postprocess ({} boxes): {:.3}s", boxes.len(), t1.elapsed().as_secs_f64());
        }
        let t2 = std::time::Instant::now();

        // ---------- crops (parallel; independent per box) ----------
        let cropped: Vec<Option<(ImageRgb, [cv::Pt; 4])>> = boxes
            .par_iter()
            .map(|b| crop_quad(img, b).filter(|c| c.w > 0 && c.h > 0).map(|c| (c, *b)))
            .collect();
        let mut crops: Vec<ImageRgb> = Vec::with_capacity(boxes.len());
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

        // sort by rec-input width so a batch groups similar-width crops
        let rec_w = |c: &ImageRgb| -> usize {
            ((REC_H as f64 * c.w as f64 / c.h as f64).ceil() as usize).clamp(1, REC_MAX_W)
        };
        let rec_widths: Vec<usize> = crops.iter().map(rec_w).collect();
        let mut order: Vec<usize> = (0..crops.len()).collect();
        order.sort_by_key(|&i| rec_widths[i]);

        // Pixel-budget batching: grow a batch until `count * max_width` exceeds a
        // budget, so narrow crops batch many together (amortising per-call cost)
        // while wide line-crops run in tiny batches (no wasted padding compute).
        let budget = std::env::var("REC_BUDGET").ok().and_then(|s| s.parse().ok()).unwrap_or(REC_BATCH_BUDGET);
        let max_count = self.rec_batch.max(1) * 16; // safety cap only
        let batches = plan_rec_batches(&order, &rec_widths, budget, max_count);
        let mut texts: Vec<(String, f32)> = vec![(String::new(), 0.0); crops.len()];

        let chars = &self.chars;
        let crops_ref = &crops;
        let batches_ref = &batches;
        let pool = self.rec.len();
        // Each pool session handles a round-robin slice of the batches.
        let partial: Vec<Vec<(usize, Vec<(String, f32)>)>> = self
            .rec
            .par_iter_mut()
            .enumerate()
            .map(|(si, sess)| {
                let mut local = Vec::new();
                for (bi, batch) in batches_ref.iter().enumerate() {
                    if bi % pool == si {
                        let decoded = rec_batch_run(sess, crops_ref, batch, chars).expect("rec batch");
                        local.push((bi, decoded));
                    }
                }
                local
            })
            .collect();
        for shard in partial {
            for (bi, decoded) in shard {
                for (k, &idx) in batches[bi].iter().enumerate() {
                    texts[idx] = decoded[k].clone();
                }
            }
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

}

/// Run one recognition batch on a given session and CTC-decode it.
fn rec_batch_run(
    sess: &mut Session,
    crops: &[ImageRgb],
    idxs: &[usize],
    chars: &[String],
) -> ort::Result<Vec<(String, f32)>> {
    let dbg = std::env::var("OCR_DEBUG").is_ok();
    let tp = std::time::Instant::now();
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
        let small = resize_bilinear_rgb(&c.data, c.w, c.h, resized_w, REC_H);
        let base = bi * 3 * plane;
        for y in 0..REC_H {
            for x in 0..resized_w {
                let si = (y * resized_w + x) * 3;
                for ch in 0..3 {
                    // plane ch is B,G,R -> interleaved RGB channel 2-ch
                    let v = small[si + 2 - ch] as f32 / 255.0;
                    let v = (v - 0.5) / 0.5;
                    data[base + ch * plane + y * img_w + x] = v;
                }
            }
        }
    }
    let prep_s = tp.elapsed().as_secs_f64();
    let ti = std::time::Instant::now();
    let tensor = Tensor::from_array(([n, 3, REC_H, img_w], data))?;
    // decode straight from the borrowed output tensor (logits are several MB
    // per batch; no need to copy them out)
    let outputs = sess.run(ort::inputs!["x" => tensor])?;
    let (shape, preds) = outputs["fetch_name_0"].try_extract_tensor::<f32>()?;
    let (t, cls) = (shape[1] as usize, shape[2] as usize);
    let infer_s = ti.elapsed().as_secs_f64();
    let tc = std::time::Instant::now();
    let mut res = Vec::with_capacity(n);
    for b in 0..n {
        res.push(ctc_decode(chars, &preds[b * t * cls..(b + 1) * t * cls], t, cls));
    }
    if dbg {
        eprintln!(
            "[dbg]     rec batch n={n} w={img_w}: prep {prep_s:.3}s infer {infer_s:.3}s ctc {:.3}s",
            tc.elapsed().as_secs_f64()
        );
    }
    Ok(res)
}

fn ctc_decode(chars: &[String], logits: &[f32], t: usize, cls: usize) -> (String, f32) {
    let mut last = usize::MAX;
    let mut s = String::new();
    let mut sum = 0.0f64;
    let mut cnt = 0u32;
    for ti in 0..t {
        let row = &logits[ti * cls..(ti + 1) * cls];
        // two-pass argmax: a lane-wise max reduction (vectorizes; the naive
        // index-tracking loop does not), then locate the first max
        let mut lanes = [f32::NEG_INFINITY; 8];
        let mut chunks = row.chunks_exact(8);
        for ch in &mut chunks {
            for (l, &v) in lanes.iter_mut().zip(ch) {
                *l = l.max(v);
            }
        }
        let mut bestv = lanes.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        for &v in chunks.remainder() {
            bestv = bestv.max(v);
        }
        let best = row.iter().position(|&v| v >= bestv).unwrap_or(0);
        // remove duplicates + blank
        if best != last {
            if best != 0 {
                s.push_str(&chars[best]);
                sum += bestv as f64;
                cnt += 1;
            }
        }
        last = best;
    }
    let score = if cnt > 0 { (sum / cnt as f64) as f32 } else { 0.0 };
    (s, score)
}

/// Group `order` (crop indices, pre-sorted by rec width ascending) into batches
/// so that `batch_len * max_rec_width_in_batch <= budget` (a padding-compute
/// budget), with a hard `max_count` per batch. Narrow crops batch many together;
/// wide line-crops end up nearly alone.
fn plan_rec_batches(order: &[usize], rec_widths: &[usize], budget: usize, max_count: usize) -> Vec<Vec<usize>> {
    let mut batches: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut cur_max = 0usize;
    for &idx in order {
        let w = rec_widths[idx];
        let new_max = cur_max.max(w);
        if !cur.is_empty() && ((cur.len() + 1) * new_max > budget || cur.len() >= max_count) {
            batches.push(std::mem::take(&mut cur));
            cur_max = 0;
        }
        cur_max = cur_max.max(w);
        cur.push(idx);
    }
    if !cur.is_empty() {
        batches.push(cur);
    }
    batches
}

fn det_resize_dims(w: usize, h: usize, max_side: i64) -> (usize, usize) {
    let (h, w) = (h as i64, w as i64);
    // limit_type = "min"
    let ratio = if w.min(h) < LIMIT_SIDE_LEN {
        LIMIT_SIDE_LEN as f64 / (if h < w { h } else { w }) as f64
    } else {
        1.0
    };
    let mut rh = (h as f64 * ratio) as i64;
    let mut rw = (w as f64 * ratio) as i64;
    if rh.max(rw) > max_side {
        let r2 = max_side as f64 / rh.max(rw) as f64;
        rh = (rh as f64 * r2) as i64;
        rw = (rw as f64 * r2) as i64;
    }
    rh = (((rh as f64 / 32.0).round() as i64) * 32).max(32);
    rw = (((rw as f64 / 32.0).round() as i64) * 32).max(32);
    (rw as usize, rh as usize)
}

/// cv2 INTER_LINEAR-style bilinear resize for interleaved RGB u8.
/// Parallel over output rows for large outputs, with precomputed per-column x
/// weights so the inner loop is cheap (f32 math). Small outputs (recognition
/// line-crops) run sequentially: they are resized *inside* the parallel rec
/// workers, where nested rayon splitting only adds scheduling overhead.
pub fn resize_bilinear_rgb(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    use rayon::prelude::*;
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let scale_x = sw as f32 / dw as f32;
    let scale_y = sh as f32 / dh as f32;
    // precompute x sampling once (reused for every row)
    let xmap: Vec<(usize, usize, f32)> = (0..dw)
        .map(|x| {
            let sx = ((x as f32 + 0.5) * scale_x - 0.5).max(0.0);
            let x0 = sx.floor();
            let ax = sx - x0;
            let x0i = (x0 as i64).clamp(0, sw as i64 - 1) as usize;
            let x1i = (x0i + 1).min(sw - 1);
            (x0i, x1i, ax)
        })
        .collect();

    let row_op = |y: usize, row: &mut [u8]| {
        let sy = ((y as f32 + 0.5) * scale_y - 0.5).max(0.0);
        let y0 = sy.floor();
        let ay = sy - y0;
        let y0i = (y0 as i64).clamp(0, sh as i64 - 1) as usize;
        let y1i = (y0i + 1).min(sh - 1);
        let r0 = y0i * sw * 3;
        let r1 = y1i * sw * 3;
        for (x, &(x0i, x1i, ax)) in xmap.iter().enumerate() {
            let i00 = r0 + x0i * 3;
            let i01 = r0 + x1i * 3;
            let i10 = r1 + x0i * 3;
            let i11 = r1 + x1i * 3;
            let o = x * 3;
            for c in 0..3 {
                let top = src[i00 + c] as f32 * (1.0 - ax) + src[i01 + c] as f32 * ax;
                let bot = src[i10 + c] as f32 * (1.0 - ax) + src[i11 + c] as f32 * ax;
                row[o + c] = (top * (1.0 - ay) + bot * ay + 0.5) as u8;
            }
        }
    };

    let mut out = vec![0u8; dw * dh * 3];
    if dw * dh >= 256 * 1024 {
        out.par_chunks_mut(dw * 3).enumerate().for_each(|(y, row)| row_op(y, row));
    } else {
        for (y, row) in out.chunks_exact_mut(dw * 3).enumerate() {
            row_op(y, row);
        }
    }
    out
}

/// DB post-process. Returns quad boxes in source-image coordinates.
fn db_postprocess(pred: &[f32], pw: usize, ph: usize, src_w: usize, src_h: usize, box_thresh: f32) -> Vec<[cv::Pt; 4]> {
    use rayon::prelude::*;
    // binary map (parallel threshold)
    let mut fg = vec![false; pw * ph];
    fg.par_iter_mut().enumerate().for_each(|(i, v)| *v = pred[i] > DET_THRESH);

    // collect connected components (sequential flood fill; cheap pointer chasing)
    let mut visited = vec![false; pw * ph];
    let mut components: Vec<Vec<cv::Pt>> = Vec::new();
    let mut stack: Vec<(i32, i32)> = Vec::new();
    'outer: for sy in 0..ph {
        for sx in 0..pw {
            let idx = sy * pw + sx;
            if !fg[idx] || visited[idx] {
                continue;
            }
            if components.len() >= DET_MAX_CANDIDATES {
                break 'outer;
            }
            let mut comp: Vec<cv::Pt> = Vec::new();
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
            if comp.len() >= 4 {
                components.push(comp);
            }
        }
    }

    let width_scale = src_w as f64 / pw as f64;
    let height_scale = src_h as f64 / ph as f64;
    // process components in parallel: minAreaRect -> score -> unclip -> scale
    components
        .par_iter()
        .filter_map(|comp| {
            let (box1, side1) = cv::min_area_rect(comp);
            if side1 < DET_MIN_SIZE {
                return None;
            }
            let score = cv::box_score_fast(pred, pw, ph, &box1);
            if box_thresh > score {
                return None;
            }
            let area = cv::poly_area(&box1);
            let perim = cv::poly_perimeter(&box1);
            if perim < 1e-6 {
                return None;
            }
            let dist = area * DET_UNCLIP_RATIO / perim;
            let box2 = cv::unclip_rect(&box1, dist);
            let (box3, side3) = cv::min_area_rect(&box2);
            if side3 < DET_MIN_SIZE + 2.0 {
                return None;
            }
            let mut scaled = [(0.0, 0.0); 4];
            for i in 0..4 {
                let x = (box3[i].0 * width_scale).round().clamp(0.0, src_w as f64);
                let y = (box3[i].1 * height_scale).round().clamp(0.0, src_h as f64);
                scaled[i] = (x, y);
            }
            Some(scaled)
        })
        .collect()
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
fn crop_quad(img: &ImageRgb, quad: &[cv::Pt; 4]) -> Option<ImageRgb> {
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
    // Fast path: an axis-aligned rectangle on integer coordinates (the common
    // case for clean scans — detected boxes are rounded to integers). The
    // perspective transform then degenerates to an integer translation and the
    // bilinear warp to an exact pixel copy, so copy rows directly.
    let axis_aligned = ordered[0].1 == ordered[1].1
        && ordered[1].0 == ordered[2].0
        && ordered[2].1 == ordered[3].1
        && ordered[3].0 == ordered[0].0
        && ordered.iter().all(|p| p.0.fract() == 0.0 && p.1.fract() == 0.0)
        && (ordered[1].0 - ordered[0].0) as usize == cw
        && (ordered[3].1 - ordered[0].1) as usize == ch
        && ordered[0].0 >= 0.0
        && ordered[0].1 >= 0.0
        && (ordered[0].0 as usize + cw) <= img.w
        && (ordered[0].1 as usize + ch) <= img.h;
    let crop = if axis_aligned {
        let (x0, y0) = (ordered[0].0 as usize, ordered[0].1 as usize);
        let mut out = vec![0u8; cw * ch * 3];
        for (r, row) in out.chunks_exact_mut(cw * 3).enumerate() {
            let s = ((y0 + r) * img.w + x0) * 3;
            row.copy_from_slice(&img.data[s..s + cw * 3]);
        }
        out
    } else {
        cv::warp_crop(&img.data, img.w, img.h, &ordered, cw, ch)
    };
    let (data, w, h) = if ch as f64 / cw as f64 >= 1.5 {
        cv::rot90_ccw(&crop, cw, ch)
    } else {
        (crop, cw, ch)
    };
    Some(ImageRgb { w, h, data })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_identity() {
        let src: Vec<u8> = (0..(4 * 3 * 3)).map(|i| (i % 256) as u8).collect();
        let out = resize_bilinear_rgb(&src, 4, 3, 4, 3);
        assert_eq!(out, src);
    }

    #[test]
    fn resize_solid_color_preserved() {
        // a solid color image resized stays the same color everywhere
        let (w, h) = (7, 5);
        let mut src = vec![0u8; w * h * 3];
        for px in src.chunks_mut(3) {
            px[0] = 30;
            px[1] = 100;
            px[2] = 200;
        }
        let out = resize_bilinear_rgb(&src, w, h, 13, 9);
        for px in out.chunks(3) {
            assert_eq!((px[0], px[1], px[2]), (30, 100, 200));
        }
    }

    #[test]
    fn resize_downscale_dims() {
        let src = vec![128u8; 100 * 80 * 3];
        let out = resize_bilinear_rgb(&src, 100, 80, 50, 40);
        assert_eq!(out.len(), 50 * 40 * 3);
    }

    #[test]
    fn batcher_narrow_crops_group_wide_run_alone() {
        // widths sorted ascending: three narrow (100) then two wide (2000)
        let widths = vec![100usize, 100, 100, 2000, 2000];
        let order: Vec<usize> = (0..widths.len()).collect();
        let b = plan_rec_batches(&order, &widths, 2400, 64);
        // narrow: 2400/100 = up to 24 -> all 3 in one batch; wide: 2400/2000 -> 1 each
        assert_eq!(b[0], vec![0, 1, 2]);
        assert_eq!(b[1], vec![3]);
        assert_eq!(b[2], vec![4]);
    }

    #[test]
    fn batcher_covers_all_indices_once() {
        let widths: Vec<usize> = (0..37).map(|i| 50 + (i * 91) % 1500).collect();
        let mut order: Vec<usize> = (0..widths.len()).collect();
        order.sort_by_key(|&i| widths[i]);
        let b = plan_rec_batches(&order, &widths, 2400, 8);
        let mut seen: Vec<usize> = b.iter().flatten().copied().collect();
        seen.sort();
        assert_eq!(seen, (0..37).collect::<Vec<_>>());
        assert!(b.iter().all(|batch| batch.len() <= 8));
    }

    #[test]
    fn det_resize_caps_long_side() {
        // large image: long side capped near max_side, rounded to a multiple of 32
        let (w, h) = det_resize_dims(2480, 3508, 1600);
        assert!(w.max(h) <= 1600 && w.max(h) >= 1568, "{w}x{h}");
        assert!(w % 32 == 0 && h % 32 == 0);
        // an image already within [736, max]: only /32 rounding, no large rescale
        let (w2, h2) = det_resize_dims(900, 800, 1600);
        assert!((w2 as i64 - 900).abs() <= 32 && (h2 as i64 - 800).abs() <= 32);
        assert!(w2 % 32 == 0 && h2 % 32 == 0);
    }

    #[test]
    fn axis_aligned_crop_matches_warp() {
        // deterministic pseudo-random image
        let (w, h) = (64, 40);
        let data: Vec<u8> = (0..w * h * 3).map(|i| ((i * 31 + 7) % 256) as u8).collect();
        let img = ImageRgb { w, h, data };
        // axis-aligned integer quad (any corner order; crop_quad re-orders)
        let quad = [(5.0, 8.0), (37.0, 8.0), (37.0, 20.0), (5.0, 20.0)];
        let fast = crop_quad(&img, &quad).unwrap();
        // reference: force the generic warp on the same ordered quad
        let warped = cv::warp_crop(&img.data, w, h, &quad, 32, 12);
        assert_eq!(fast.w, 32);
        assert_eq!(fast.h, 12);
        assert_eq!(fast.data, warped, "fast path must equal the perspective warp");
    }

    #[test]
    fn ctc_argmax_first_max_wins_ties() {
        // two equal maxima per row: index of the FIRST must win (matches the
        // strict `>` scan it replaced)
        let chars = vec!["blank".to_string(), "A".to_string(), "B".to_string(), "C".to_string()];
        let rows = [
            [0.1f32, 0.8, 0.8, 0.1], // tie A/B -> A
            [0.1, 0.1, 0.9, 0.9],    // tie B/C -> B
        ];
        let logits: Vec<f32> = rows.iter().flatten().copied().collect();
        let (text, _) = ctc_decode(&chars, &logits, 2, 4);
        assert_eq!(text, "AB");
    }

    #[test]
    fn ctc_decode_removes_repeats_and_blank() {
        // chars: index0=blank, 1='A', 2='B'
        let chars = vec!["blank".to_string(), "A".to_string(), "B".to_string()];
        let cls = 3;
        // timesteps: A A blank B  -> "AB"
        let rows = [
            [0.1f32, 0.8, 0.1], // A
            [0.1, 0.8, 0.1],    // A (dup)
            [0.9, 0.05, 0.05],  // blank
            [0.1, 0.1, 0.8],    // B
        ];
        let logits: Vec<f32> = rows.iter().flatten().copied().collect();
        let (text, score) = ctc_decode(&chars, &logits, 4, cls);
        assert_eq!(text, "AB");
        assert!(score > 0.7);
    }
}
