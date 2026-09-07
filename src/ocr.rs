//! PP-OCRv6 tiny detection + recognition pipeline (CPU, ONNX Runtime).

use crate::cv;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

// ---- detection params (from PP-OCRv6 *_det inference.yml) ----
const DET_THRESH: f32 = 0.2;
const DET_UNCLIP_RATIO: f64 = 1.4;
const DET_MAX_CANDIDATES: usize = 3000;
const DET_MIN_SIZE: f64 = 3.0;
const LIMIT_SIDE_LEN: i64 = 736;
const MAX_SIDE_LIMIT: i64 = 4000;
/// Detector long-side cap. Lowering it saves compute but can miss small text;
/// recognition still uses full-resolution source crops.
pub const DEFAULT_DET_MAX_SIDE: i64 = 1600;
const DET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const DET_STD: [f32; 3] = [0.229, 0.224, 0.225];

const REC_H: usize = 48;
const REC_MAX_W: usize = 3200;
// Singleton crops minimize measured CPU latency; the worker pool supplies
// parallelism. The public override is a real cap, not a multiplier.
pub const DEFAULT_REC_BATCH: usize = 1;
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

struct RecWorker {
    session: Session,
    input: Vec<f32>,
}

pub struct Engine {
    det: Session,
    /// Pool of recognition sessions, run concurrently across batches so the many
    /// small rec matmuls keep all cores busy (a single session under-utilizes them).
    rec: Vec<RecWorker>,
    wide_rec: Vec<RecWorker>,
    wide_threads: usize,
    approximate_gelu: bool,
    det_spin: bool,
    rec_spin: bool,
    pub workers: Arc<rayon::ThreadPool>,
    rec_min_width: usize,
    rec_budget: usize,
    det_min_side: i64,
    det_input: Vec<f32>,
    det_threads: usize,
    rec_threads: usize,
    threads: usize,
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
        let approximate_gelu = std::env::var("OCR_APPROX_GELU")
            .map(|v| v != "0")
            .unwrap_or(false);
        let mempat = std::env::var("OCR_MEMPAT")
            .map(|v| v != "0")
            .unwrap_or(false);
        let prepack = std::env::var("OCR_PREPACK")
            .map(|v| v != "0")
            .unwrap_or(true);
        let rec_spin = std::env::var("OCR_REC_SPIN")
            .map(|v| v != "0")
            .unwrap_or(true);
        let det_spin = std::env::var("OCR_DET_SPIN")
            .map(|v| v != "0")
            .unwrap_or(true);
        let build = |bytes: &[u8],
                     t: usize,
                     spin: bool,
                     pw: Option<&ort::session::builder::PrepackedWeights>|
         -> ort::Result<Session> {
            let mut b = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_memory_pattern(mempat)?
                .with_intra_op_spinning(spin)?
                .with_config_entry("session.force_spinning_stop", "1")?
                .with_intra_threads(t.max(1))?;
            if approximate_gelu {
                b = b.with_approximate_gelu()?;
            }
            if let Some(pw) = pw {
                b = b.with_prepacked_weights(pw)?;
            }
            b.commit_from_memory(bytes)
        };
        // Spin only while Run is active. force_spinning_stop prevents idle
        // detector/recognizer threads stealing CPU from subsequent phases.
        let det = build(det_bytes, det_threads.max(1), det_spin, None)?;
        // Pool of rec sessions; split the threads across them so concurrent
        // batches together saturate the cores. All pool sessions share one
        // prepacked-weights container so the packed weight buffers exist once,
        // not `pool` times (less memory, better cache reuse).
        let pool = rec_pool.clamp(1, threads.max(1));
        let per = (threads / pool).max(1);
        let shared = ort::session::builder::PrepackedWeights::new();
        let workers = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(
                    crate::env_usize("RAYON_NUM_THREADS")
                        .unwrap_or(threads)
                        .min(threads)
                        .max(1),
                )
                .build()
                .map_err(|e| ort::Error::new(e.to_string()))?,
        );
        let build_workers = |count: usize, intra: usize| -> ort::Result<Vec<RecWorker>> {
            use rayon::prelude::*;
            let mut result = Vec::with_capacity(count);
            // Bound simultaneous graph optimization and transient model memory.
            for start in (0..count).step_by(4) {
                let group: ort::Result<Vec<_>> = workers.install(|| {
                    (start..(start + 4).min(count))
                        .into_par_iter()
                        .map(|_| {
                            Ok(RecWorker {
                                session: build(
                                    rec_bytes,
                                    intra,
                                    rec_spin,
                                    prepack.then_some(&shared),
                                )?,
                                input: Vec::new(),
                            })
                        })
                        .collect()
                });
                result.extend(group?);
            }
            Ok(result)
        };
        let rec = build_workers(pool, per)?;
        // Sparse wide jobs use a small intra-op pool; dense jobs use the normal
        // crop pool. These pools are never active at the same time.
        let wide_threads = threads.min(4).max(1);
        let wide_rec = build_workers(
            if wide_threads > per {
                (threads / wide_threads).clamp(1, 4)
            } else {
                0
            },
            wide_threads,
        )?;
        // CHARS = ["blank"] + dict + [" "]
        let mut chars = Vec::with_capacity(char_dict.len() + 2);
        chars.push("blank".to_string());
        chars.extend(char_dict);
        chars.push(" ".to_string());
        Ok(Engine {
            det,
            rec,
            wide_rec,
            wide_threads,
            workers,
            approximate_gelu,
            det_spin,
            rec_spin,
            rec_min_width: crate::env_usize("OCR_REC_MIN_WIDTH")
                .unwrap_or(320)
                .clamp(64, REC_MAX_W),
            rec_budget: crate::env_usize("REC_BUDGET").unwrap_or(REC_BATCH_BUDGET),
            det_min_side: std::env::var("OCR_DET_MIN_SIDE")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&v| v >= 0)
                .unwrap_or(LIMIT_SIDE_LEN),
            det_input: Vec::new(),
            det_threads,
            rec_threads: per,
            threads,
            chars,
            rec_batch: rec_batch.max(1),
            box_thresh,
            det_max_side: if det_max_side <= 0 {
                MAX_SIDE_LIMIT
            } else {
                det_max_side
            },
        })
    }

    pub fn set_input_limits(&mut self, det_min: Option<i64>, rec_min: Option<usize>) {
        if let Some(v) = det_min {
            self.det_min_side = v;
        }
        if let Some(v) = rec_min {
            self.rec_min_width = v;
        }
    }

    pub fn settings(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("threads", self.threads),
            ("approximate_gelu", usize::from(self.approximate_gelu)),
            ("det_spin", usize::from(self.det_spin)),
            ("rec_spin", usize::from(self.rec_spin)),
            ("det_threads", self.det_threads),
            ("rec_workers", self.rec.len()),
            ("rec_threads", self.rec_threads),
            ("wide_rec_threads", self.wide_threads),
            ("wide_rec_workers", self.wide_rec.len()),
            ("rayon_threads", self.workers.current_num_threads()),
            ("rec_batch", self.rec_batch),
            ("rec_budget", self.rec_budget),
            ("rec_min_width", self.rec_min_width),
            ("det_min_side", self.det_min_side as usize),
            ("det_max_side", self.det_max_side as usize),
        ]
    }

    pub fn run(&mut self, img: &ImageRgb) -> ort::Result<Vec<OcrResult>> {
        let workers = Arc::clone(&self.workers);
        workers.install(|| {
            let (crops, boxes) = self.detect_crops(img)?;
            self.recognize(&crops, &boxes)
        })
    }

    pub fn run_many(&mut self, images: &[ImageRgb]) -> ort::Result<Vec<Vec<OcrResult>>> {
        let workers = Arc::clone(&self.workers);
        workers.install(|| {
            let mut crops = Vec::new();
            let mut boxes = Vec::new();
            let mut counts = Vec::with_capacity(images.len());
            for img in images {
                let (cs, bs) = self.detect_crops(img)?;
                counts.push(cs.len());
                crops.extend(cs);
                boxes.extend(bs);
            }
            let mut recognized = self.recognize(&crops, &boxes)?.into_iter();
            Ok(counts
                .into_iter()
                .map(|n| recognized.by_ref().take(n).collect())
                .collect())
        })
    }

    fn detect_crops(&mut self, img: &ImageRgb) -> ort::Result<(Vec<ImageRgb>, Vec<[cv::Pt; 4]>)> {
        let dbg = std::env::var("OCR_DEBUG").is_ok();
        let t0 = std::time::Instant::now();
        // ---------- detection ----------
        use rayon::prelude::*;
        let (rw, rh) = det_resize_dims_with_min(img.w, img.h, self.det_max_side, self.det_min_side);
        let tpn = std::time::Instant::now();
        self.det_input.resize(3 * rw * rh, 0.0);
        let input = &mut self.det_input;
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
        resize_to_tensor(
            img,
            rw,
            rh,
            rw,
            input,
            |v, c| v as f32 * alpha[c] + beta[c],
            true,
        );
        if dbg {
            eprintln!(
                "[dbg]   det resize+normalize: {:.3}s",
                tpn.elapsed().as_secs_f64()
            );
        }
        let tinf = std::time::Instant::now();
        let tensor = TensorRef::from_array_view(([1usize, 3, rh, rw], input.as_slice()))?;
        let box_thresh = self.box_thresh;
        // post-process directly on the borrowed output tensor (the prob map is
        // several MB; no need to copy it out)
        let t1;
        let boxes = {
            let outputs = self.det.run(ort::inputs!["x" => tensor])?;
            let (shape, pred) = outputs["fetch_name_0"].try_extract_tensor::<f32>()?;
            let (ph, pw) = (shape[2] as usize, shape[3] as usize);
            if dbg {
                eprintln!(
                    "[dbg]   det ORT infer: {:.3}s",
                    tinf.elapsed().as_secs_f64()
                );
                eprintln!(
                    "[dbg] det total ({}x{}): {:.3}s",
                    rw,
                    rh,
                    t0.elapsed().as_secs_f64()
                );
            }
            t1 = std::time::Instant::now();
            db_postprocess(pred, pw, ph, img.w, img.h, box_thresh)
        };
        let boxes = sort_boxes(boxes);
        if dbg {
            eprintln!(
                "[dbg] db_postprocess ({} boxes): {:.3}s",
                boxes.len(),
                t1.elapsed().as_secs_f64()
            );
        }
        let t2 = std::time::Instant::now();

        // ---------- crops (parallel; independent per box) ----------
        let cropped: Vec<Option<(ImageRgb, [cv::Pt; 4])>> = boxes
            .par_iter()
            .map(|b| {
                crop_quad(img, b)
                    .filter(|c| c.w > 0 && c.h > 0)
                    .map(|c| (c, *b))
            })
            .collect();
        let mut crops: Vec<ImageRgb> = Vec::with_capacity(boxes.len());
        let mut kept_boxes: Vec<[cv::Pt; 4]> = Vec::with_capacity(boxes.len());
        for item in cropped.into_iter().flatten() {
            crops.push(item.0);
            kept_boxes.push(item.1);
        }
        if dbg {
            eprintln!(
                "[dbg] crops ({}): {:.3}s",
                crops.len(),
                t2.elapsed().as_secs_f64()
            );
        }
        Ok((crops, kept_boxes))
    }

    fn recognize(
        &mut self,
        crops: &[ImageRgb],
        kept_boxes: &[[cv::Pt; 4]],
    ) -> ort::Result<Vec<OcrResult>> {
        use rayon::prelude::*;
        if crops.is_empty() {
            return Ok(Vec::new());
        }
        let dbg = std::env::var("OCR_DEBUG").is_ok();
        let t3 = std::time::Instant::now();
        let min_width = self.rec_min_width;
        let rec_widths: Vec<usize> = crops
            .iter()
            .map(|c| rec_width(c.w, c.h, min_width))
            .collect();
        let mut order: Vec<usize> = (0..crops.len()).collect();
        order.sort_by_key(|&i| rec_widths[i]);
        let batches = plan_rec_batches(&order, &rec_widths, self.rec_budget, self.rec_batch);
        let mut jobs: Vec<usize> = (0..batches.len()).collect();
        // Start expensive jobs first; workers dynamically claim the next job.
        jobs.sort_by_key(|&i| {
            std::cmp::Reverse(batches[i].len() * rec_widths[*batches[i].last().unwrap()])
        });
        let next = AtomicUsize::new(0);
        let chars = &self.chars;
        let wide_threshold = if chars.len() <= 6906 { 2048 } else { 1024 };
        let use_wide = jobs.len() <= self.wide_rec.len()
            && batches
                .iter()
                .all(|batch| rec_widths[*batch.last().unwrap()] >= wide_threshold);
        let workers = if use_wide {
            &mut self.wide_rec
        } else {
            &mut self.rec
        };
        let partial: ort::Result<Vec<_>> = workers
            .par_iter_mut()
            .take(jobs.len())
            .map(|worker| {
                let mut local = Vec::new();
                loop {
                    let job = next.fetch_add(1, Ordering::Relaxed);
                    if job >= jobs.len() {
                        break;
                    }
                    let bi = jobs[job];
                    let decoded = rec_batch_run(worker, crops, &batches[bi], chars, min_width)?;
                    local.push((bi, decoded));
                }
                Ok(local)
            })
            .collect();
        let mut texts = vec![(String::new(), 0.0); crops.len()];
        for shard in partial? {
            for (bi, decoded) in shard {
                for (&idx, result) in batches[bi].iter().zip(decoded) {
                    texts[idx] = result;
                }
            }
        }

        if dbg {
            eprintln!(
                "[dbg] rec ({} crops): {:.3}s",
                crops.len(),
                t3.elapsed().as_secs_f64()
            );
        }
        // assemble results (score_thresh = 0.0 -> keep all)
        let mut out = Vec::with_capacity(crops.len());
        for ((t, sc), q) in texts.into_iter().zip(kept_boxes) {
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
    worker: &mut RecWorker,
    crops: &[ImageRgb],
    idxs: &[usize],
    chars: &[String],
    min_width: usize,
) -> ort::Result<Vec<(String, f32)>> {
    let dbg = std::env::var("OCR_DEBUG").is_ok();
    let tp = std::time::Instant::now();
    let img_w = idxs
        .iter()
        .map(|&i| rec_width(crops[i].w, crops[i].h, min_width))
        .max()
        .unwrap_or(min_width);
    let n = idxs.len();
    let plane = REC_H * img_w;
    worker.input.resize(n * 3 * plane, 0.0);
    let data = &mut worker.input;
    for (bi, &i) in idxs.iter().enumerate() {
        let c = &crops[i];
        // resized width
        let resized_w = if img_w >= REC_MAX_W
            && (REC_H as f64 * c.w as f64 / c.h as f64) as usize > REC_MAX_W
        {
            REC_MAX_W
        } else {
            let ratio = c.w as f64 / c.h as f64;
            let rw = (REC_H as f64 * ratio).ceil() as usize;
            rw.min(img_w).max(1)
        };
        let base = bi * 3 * plane;
        resize_to_tensor(
            c,
            resized_w,
            REC_H,
            img_w,
            &mut data[base..base + 3 * plane],
            |v, _| (v as f32 / 255.0 - 0.5) / 0.5,
            false,
        );
    }
    let prep_s = tp.elapsed().as_secs_f64();
    let ti = std::time::Instant::now();
    let tensor = TensorRef::from_array_view(([n, 3, REC_H, img_w], data.as_slice()))?;
    // decode straight from the borrowed output tensor (logits are several MB
    // per batch; no need to copy them out)
    let outputs = worker.session.run(ort::inputs!["x" => tensor])?;
    let (shape, preds) = outputs["fetch_name_0"].try_extract_tensor::<f32>()?;
    let (t, cls) = (shape[1] as usize, shape[2] as usize);
    let infer_s = ti.elapsed().as_secs_f64();
    let tc = std::time::Instant::now();
    let mut res = Vec::with_capacity(n);
    for b in 0..n {
        res.push(ctc_decode(
            chars,
            &preds[b * t * cls..(b + 1) * t * cls],
            t,
            cls,
        ));
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
    let score = if cnt > 0 {
        (sum / cnt as f64) as f32
    } else {
        0.0
    };
    (s, score)
}

/// Group `order` (crop indices, pre-sorted by rec width ascending) into batches
/// so that `batch_len * max_rec_width_in_batch <= budget` (a padding-compute
/// budget), with a hard `max_count` per batch. Narrow crops batch many together;
/// wide line-crops end up nearly alone.
fn plan_rec_batches(
    order: &[usize],
    rec_widths: &[usize],
    budget: usize,
    max_count: usize,
) -> Vec<Vec<usize>> {
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

// Preserve the runner's historical floor rounding; planning and allocation must agree.
fn rec_width(w: usize, h: usize, min_width: usize) -> usize {
    ((REC_H as f64 * w as f64 / h as f64) as usize).clamp(min_width, REC_MAX_W)
}

#[cfg(test)]
fn det_resize_dims(w: usize, h: usize, max_side: i64) -> (usize, usize) {
    det_resize_dims_with_min(w, h, max_side, LIMIT_SIDE_LEN)
}

fn det_resize_dims_with_min(w: usize, h: usize, max_side: i64, min_side: i64) -> (usize, usize) {
    let (h, w) = (h as i64, w as i64);
    // A zero minimum disables the minimum-side upscale.

    let ratio = if w.min(h) < min_side {
        min_side as f64 / (if h < w { h } else { w }) as f64
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
    let cap = (max_side / 32).max(1) * 32;
    rh = (((rh as f64 / 32.0).round() as i64) * 32).clamp(32, cap);
    rw = (((rw as f64 / 32.0).round() as i64) * 32).clamp(32, cap);
    (rw as usize, rh as usize)
}

/// Fuse byte-exact bilinear resize, BGR planar conversion and normalization.
/// Padding is reset on every use so reused buffers never leak a previous crop.
fn resize_to_tensor(
    src: &ImageRgb,
    dw: usize,
    dh: usize,
    stride: usize,
    out: &mut [f32],
    normalize: impl Fn(u8, usize) -> f32 + Sync,
    parallel: bool,
) {
    use rayon::prelude::*;
    let sx = src.w as f32 / dw as f32;
    let sy = src.h as f32 / dh as f32;
    let xmap: Vec<_> = (0..dw)
        .map(|x| {
            let v = ((x as f32 + 0.5) * sx - 0.5).max(0.0);
            let lo = (v.floor() as usize).min(src.w - 1);
            (lo, (lo + 1).min(src.w - 1), v - v.floor())
        })
        .collect();
    let row = |i: usize, dst: &mut [f32]| {
        let (c, y) = (i / dh, i % dh);
        let v = ((y as f32 + 0.5) * sy - 0.5).max(0.0);
        let y0 = (v.floor() as usize).min(src.h - 1);
        let y1 = (y0 + 1).min(src.h - 1);
        let ay = v - v.floor();
        let (r0, r1) = (y0 * src.w * 3, y1 * src.w * 3);
        let ch = 2 - c;
        for (x, &(x0, x1, ax)) in xmap.iter().enumerate() {
            let top = src.data[r0 + x0 * 3 + ch] as f32 * (1.0 - ax)
                + src.data[r0 + x1 * 3 + ch] as f32 * ax;
            let bot = src.data[r1 + x0 * 3 + ch] as f32 * (1.0 - ax)
                + src.data[r1 + x1 * 3 + ch] as f32 * ax;
            let byte = (top * (1.0 - ay) + bot * ay + 0.5) as u8;
            dst[x] = normalize(byte, c);
        }
        dst[dw..].fill(0.0);
    };
    if parallel && dw * dh >= 256 * 1024 {
        out.par_chunks_mut(stride)
            .enumerate()
            .for_each(|(i, dst)| row(i, dst));
    } else {
        for (i, dst) in out.chunks_exact_mut(stride).enumerate() {
            row(i, dst);
        }
    }
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
        out.par_chunks_mut(dw * 3)
            .enumerate()
            .for_each(|(y, row)| row_op(y, row));
    } else {
        for (y, row) in out.chunks_exact_mut(dw * 3).enumerate() {
            row_op(y, row);
        }
    }
    out
}

/// Scanline flood fill with 8-connectivity. Horizontal run endpoints have the
/// same convex hull as all foreground pixels, without storing/sorting interiors.
fn component_extrema(fg: &mut [u8], w: usize, h: usize, limit: usize) -> Vec<Vec<cv::Pt>> {
    let mut components = Vec::new();
    let mut stack = Vec::new();
    for seed in 0..fg.len() {
        if fg[seed] == 0 {
            continue;
        }
        if components.len() >= limit {
            break;
        }
        stack.push(seed);
        let mut points = Vec::new();
        let mut count = 0;
        while let Some(at) = stack.pop() {
            if fg[at] == 0 {
                continue;
            }
            let (y, x) = (at / w, at % w);
            let row = y * w;
            let mut left = x;
            let mut right = x;
            while left > 0 && fg[row + left - 1] != 0 {
                left -= 1;
            }
            while right + 1 < w && fg[row + right + 1] != 0 {
                right += 1;
            }
            fg[row + left..=row + right].fill(0);
            count += right - left + 1;
            points.push((left as f64, y as f64));
            if right != left {
                points.push((right as f64, y as f64));
            }
            for ny in [y.checked_sub(1), (y + 1 < h).then_some(y + 1)]
                .into_iter()
                .flatten()
            {
                let mut nx = left.saturating_sub(1);
                let end = (right + 1).min(w - 1);
                while nx <= end {
                    if fg[ny * w + nx] != 0 {
                        stack.push(ny * w + nx);
                        // One seed per neighboring run, claimed when popped.
                        while nx <= end && fg[ny * w + nx] != 0 {
                            nx += 1;
                        }
                    } else {
                        nx += 1;
                    }
                }
            }
        }
        if count >= 4 {
            components.push(points);
        }
    }
    components
}

/// DB post-process. Returns quad boxes in source-image coordinates.
fn db_postprocess(
    pred: &[f32],
    pw: usize,
    ph: usize,
    src_w: usize,
    src_h: usize,
    box_thresh: f32,
) -> Vec<[cv::Pt; 4]> {
    use rayon::prelude::*;
    let mut fg: Vec<u8> = pred.par_iter().map(|&v| u8::from(v > DET_THRESH)).collect();
    let components = component_extrema(&mut fg, pw, ph, DET_MAX_CANDIDATES);

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
            if (boxes[ju + 1][0].1 - boxes[ju][0].1).abs() < 10.0
                && boxes[ju + 1][0].0 < boxes[ju][0].0
            {
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
    let mut pts = rect;
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
        && ordered
            .iter()
            .all(|p| p.0.fract() == 0.0 && p.1.fract() == 0.0)
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
    fn scanline_components_preserve_pixel_hulls() {
        // Exhaust every 3x3 map, including diagonal-only 8-connected regions,
        // then larger deterministic maps with holes, bridges and edge runs.
        for sample in 0..560u64 {
            let (w, h) = if sample < 512 { (3, 3) } else { (31, 19) };
            let mut state = sample + 1;
            let pixels: Vec<u8> = (0..w * h)
                .map(|i| {
                    if sample < 512 {
                        ((sample >> i) & 1) as u8
                    } else {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((state >> 32) % 3 != 0) as u8
                    }
                })
                .collect();
            let mut visited = vec![false; pixels.len()];
            let mut expected = Vec::new();
            for seed in 0..pixels.len() {
                if visited[seed] || pixels[seed] == 0 {
                    continue;
                }
                let mut pending = vec![seed];
                visited[seed] = true;
                let mut points = Vec::new();
                while let Some(at) = pending.pop() {
                    let (x, y) = (at % w, at / w);
                    points.push((x as f64, y as f64));
                    for ny in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                        for nx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                            let ni = ny * w + nx;
                            if !visited[ni] && pixels[ni] != 0 {
                                visited[ni] = true;
                                pending.push(ni);
                            }
                        }
                    }
                }
                if points.len() >= 4 {
                    expected.push(cv::convex_hull(&points));
                }
            }
            let actual: Vec<_> = component_extrema(&mut pixels.clone(), w, h, usize::MAX)
                .iter()
                .map(|p| cv::convex_hull(p))
                .collect();
            assert_eq!(actual, expected, "map {sample}");
            let limited = component_extrema(&mut pixels.clone(), w, h, 1);
            assert_eq!(limited.len(), expected.len().min(1));
        }
    }

    #[test]
    fn recognition_budget_accounts_for_padding_and_honors_cap() {
        let widths: Vec<_> = [(10, 48), (20, 48), (500, 48)]
            .into_iter()
            .map(|(w, h)| rec_width(w, h, 320))
            .collect();
        assert_eq!(widths, [320, 320, 500]);
        assert_eq!(
            plan_rec_batches(&[0, 1, 2], &widths, 800, 4),
            vec![vec![0, 1], vec![2]]
        );
        assert_eq!(
            plan_rec_batches(&[0, 1, 2], &widths, 800, 1),
            vec![vec![0], vec![1], vec![2]]
        );
        assert_eq!(rec_width(100000, 1, 64), REC_MAX_W);
        assert_eq!(rec_width(1, 10000, 64), 64);
    }

    #[test]
    fn detection_minimum_side_can_be_disabled() {
        assert_eq!(det_resize_dims_with_min(804, 505, 1600, 736), (1184, 736));
        assert_eq!(det_resize_dims_with_min(804, 505, 1600, 0), (800, 512));
        for cap in [32, 33, 63, 1000, 1599] {
            let (w, h) = det_resize_dims_with_min(2480, 3508, cap, 736);
            assert!(w.max(h) <= cap as usize);
            assert!(w % 32 == 0 && h % 32 == 0);
        }
    }

    #[test]
    fn fused_tensor_matches_resize_and_clears_old_padding() {
        let img = ImageRgb {
            w: 29,
            h: 13,
            data: (0..29 * 13 * 3)
                .map(|i| ((i * 37 + 7) % 256) as u8)
                .collect(),
        };
        for (dw, dh) in [(29, 13), (7, 9), (63, 48), (1, 1), (800, 736)] {
            let rgb = resize_bilinear_rgb(&img.data, img.w, img.h, dw, dh);
            let stride = dw + 11;
            let mut actual = vec![123.0; 3 * stride * dh];
            let normalize = |v: u8, _: usize| (v as f32 / 255.0 - 0.5) / 0.5;
            resize_to_tensor(&img, dw, dh, stride, &mut actual, normalize, true);
            for c in 0..3 {
                for y in 0..dh {
                    for x in 0..stride {
                        let expected = if x < dw {
                            normalize(rgb[(y * dw + x) * 3 + 2 - c], c)
                        } else {
                            0.0
                        };
                        assert_eq!(actual[(c * dh + y) * stride + x], expected);
                    }
                }
            }
        }
    }

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
        assert_eq!(
            fast.data, warped,
            "fast path must equal the perspective warp"
        );
    }

    #[test]
    fn ctc_argmax_first_max_wins_ties() {
        // two equal maxima per row: index of the FIRST must win (matches the
        // strict `>` scan it replaced)
        let chars = vec![
            "blank".to_string(),
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
        ];
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
