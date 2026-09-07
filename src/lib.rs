//! Python bindings for FasterPaddle — a fast, CPU-only OCR engine specialized
//! for PaddleOCR's lightweight PP-OCRv6 *tiny* detection + recognition models.
//!
//! The ONNX models and character dictionary are embedded in the compiled
//! extension, so the wheel is fully self-contained — no model files or network
//! access are needed at runtime.

mod cv;
mod hardware;
mod layout;
mod ocr;
mod preprocess;

// Older-glibc compatibility shims. Prebuilt components (ONNX Runtime, Rust std)
// reference a few symbols from newer glibc; we provide them ourselves so the
// references resolve at link time and the wheels run on older glibc:
//   - __isoc23_strto{l,ll,ull} (glibc 2.38) — ONNX Runtime's strtol* redirects.
//   - __libc_single_threaded   (glibc 2.32) — Rust std's atomics fast-path.
#[cfg(target_os = "linux")]
mod glibc_compat {
    use std::os::raw::{c_char, c_int, c_long, c_longlong, c_ulonglong};
    extern "C" {
        fn strtol(s: *const c_char, e: *mut *mut c_char, b: c_int) -> c_long;
        fn strtoll(s: *const c_char, e: *mut *mut c_char, b: c_int) -> c_longlong;
        fn strtoull(s: *const c_char, e: *mut *mut c_char, b: c_int) -> c_ulonglong;
    }
    #[no_mangle]
    pub unsafe extern "C" fn __isoc23_strtol(
        s: *const c_char,
        e: *mut *mut c_char,
        b: c_int,
    ) -> c_long {
        strtol(s, e, b)
    }
    #[no_mangle]
    pub unsafe extern "C" fn __isoc23_strtoll(
        s: *const c_char,
        e: *mut *mut c_char,
        b: c_int,
    ) -> c_longlong {
        strtoll(s, e, b)
    }
    #[no_mangle]
    pub unsafe extern "C" fn __isoc23_strtoull(
        s: *const c_char,
        e: *mut *mut c_char,
        b: c_int,
    ) -> c_ulonglong {
        strtoull(s, e, b)
    }

    // glibc 2.32 introduced `__libc_single_threaded` (a byte that Rust's std reads
    // to skip atomics in single-threaded processes). The prebuilt std references
    // it, so on glibc < 2.32 the extension fails with
    // `undefined symbol: __libc_single_threaded` (hit on aarch64 wheels, whose
    // other symbols top out at glibc 2.28 — e.g. Debian 11 / Ubuntu 20.04 arm64).
    // Provide it as 0 ("not single-threaded" — always-safe: std just keeps using
    // atomics), so the wheel loads on glibc 2.28+. It must live in writable .data
    // (glibc's real one is a writable global), hence `static mut`: in an
    // executable context glibc may write through this symbol, which would fault
    // on a read-only static.
    #[no_mangle]
    pub static mut __libc_single_threaded: u8 = 0;
}

use base64::Engine as _;
use ocr::{Engine, ImageRgb};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::{PyBytes, PyDict, PyTuple};
use std::borrow::Cow;
use std::sync::{Mutex, OnceLock};

// ---- embedded models (tiny + small) ----
// The medium models (~138 MB) exceed PyPI's size limit, so they are downloaded
// on demand and cached locally (see `medium_model_bytes`).
const TINY_DET: &[u8] = include_bytes!("../models/tiny/det.onnx");
const TINY_REC: &[u8] = include_bytes!("../models/tiny/rec.onnx");
const TINY_DICT: &str = include_str!("../models/tiny/char_dict.json");
const SMALL_DET: &[u8] = include_bytes!("../models/small/det.onnx");
const SMALL_REC: &[u8] = include_bytes!("../models/small/rec.onnx");
// small and medium share the same (larger) character dictionary
const BIG_DICT: &str = include_str!("../models/small/char_dict.json");

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn parse_dict(json: &str) -> Vec<String> {
    serde_json::from_str(json).expect("embedded char_dict.json is valid")
}

/// Resolved model assets for a given size: det bytes, rec bytes, char dict, and
/// the detection box-score threshold (tiny=0.40, small/medium=0.45).
struct ModelAssets {
    det: Cow<'static, [u8]>,
    rec: Cow<'static, [u8]>,
    dict: Vec<String>,
    box_thresh: f32,
}

fn resolve_model(size: &str) -> PyResult<ModelAssets> {
    match size {
        "tiny" => Ok(ModelAssets {
            det: Cow::Borrowed(TINY_DET),
            rec: Cow::Borrowed(TINY_REC),
            dict: parse_dict(TINY_DICT),
            box_thresh: 0.40,
        }),
        "small" => Ok(ModelAssets {
            det: Cow::Borrowed(SMALL_DET),
            rec: Cow::Borrowed(SMALL_REC),
            dict: parse_dict(BIG_DICT),
            box_thresh: 0.45,
        }),
        "medium" => {
            let (det, rec) = medium_model_bytes()?;
            Ok(ModelAssets {
                det: Cow::Owned(det),
                rec: Cow::Owned(rec),
                dict: parse_dict(BIG_DICT),
                box_thresh: 0.45,
            })
        }
        other => Err(PyValueError::new_err(format!(
            "unknown model_size {other:?}; expected 'tiny', 'small', or 'medium'"
        ))),
    }
}

/// Download (once, then cache) and return the medium det + rec ONNX bytes.
fn medium_model_bytes() -> PyResult<(Vec<u8>, Vec<u8>)> {
    let cache = dirs::cache_dir()
        .ok_or_else(|| {
            PyRuntimeError::new_err("cannot determine a cache directory for medium models")
        })?
        .join("faster_paddle")
        .join(format!("v{VERSION}"))
        .join("medium");
    let det = fetch_cached(&cache, "det.onnx", "ppocrv6_medium_det.onnx")?;
    let rec = fetch_cached(&cache, "rec.onnx", "ppocrv6_medium_rec.onnx")?;
    Ok((det, rec))
}

fn fetch_cached(cache_dir: &std::path::Path, filename: &str, asset: &str) -> PyResult<Vec<u8>> {
    let path = cache_dir.join(filename);
    if let Ok(bytes) = std::fs::read(&path) {
        if bytes.len() > 1024 {
            return Ok(bytes);
        }
    }
    std::fs::create_dir_all(cache_dir)
        .map_err(|e| PyRuntimeError::new_err(format!("cannot create cache dir: {e}")))?;
    let url =
        format!("https://github.com/cnmoro/faster-paddle/releases/download/v{VERSION}/{asset}");
    let resp = ureq::get(&url).call().map_err(|e| {
        PyRuntimeError::new_err(format!("failed to download medium model from {url}: {e}"))
    })?;
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut resp.into_reader(), &mut bytes)
        .map_err(|e| PyRuntimeError::new_err(format!("failed reading medium model: {e}")))?;
    if bytes.len() <= 1024 {
        return Err(PyRuntimeError::new_err(format!(
            "downloaded medium model from {url} looks invalid ({} bytes)",
            bytes.len()
        )));
    }
    // atomic-ish write via temp file
    let tmp = cache_dir.join(format!("{filename}.tmp"));
    std::fs::write(&tmp, &bytes)
        .map_err(|e| PyRuntimeError::new_err(format!("cannot write cache: {e}")))?;
    let _ = std::fs::rename(&tmp, &path);
    Ok(bytes)
}

/// Decode encoded image bytes (jpeg/png/...) into an RGB buffer. The network
/// wants BGR planes (cv2.imread order), but rather than swapping every pixel
/// here, the channel index is flipped where the NCHW tensors are filled — so a
/// JPEG/PNG that decodes straight to RGB8 needs no extra pass at all.
fn decode_rgb(bytes: &[u8]) -> Result<ImageRgb, String> {
    let img = image::load_from_memory(bytes).map_err(|e| e.to_string())?;
    let rgb = img.into_rgb8(); // move (no copy) when already RGB8
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    Ok(ImageRgb {
        w,
        h,
        data: rgb.into_raw(),
    })
}

/// Encode an RGB image to PNG bytes. If the image is grayscale (all channels
/// equal, e.g. after denoise/deskew/binarize) it is written as a smaller 8-bit
/// grayscale PNG; otherwise as RGB.
fn encode_png(img: ImageRgb) -> Result<Vec<u8>, String> {
    let n = img.w * img.h;
    let is_gray = (0..n).all(|i| {
        img.data[i * 3] == img.data[i * 3 + 1] && img.data[i * 3 + 1] == img.data[i * 3 + 2]
    });
    let mut out = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut out);
    if is_gray {
        let gray: Vec<u8> = (0..n).map(|i| img.data[i * 3]).collect();
        let buf = image::GrayImage::from_raw(img.w as u32, img.h as u32, gray)
            .ok_or("failed to build grayscale image")?;
        image::DynamicImage::ImageLuma8(buf)
            .write_to(&mut cursor, image::ImageFormat::Png)
            .map_err(|e| e.to_string())?;
    } else {
        let buf = image::RgbImage::from_raw(img.w as u32, img.h as u32, img.data)
            .ok_or("failed to build RGB image")?;
        image::DynamicImage::ImageRgb8(buf)
            .write_to(&mut cursor, image::ImageFormat::Png)
            .map_err(|e| e.to_string())?;
    }
    Ok(out)
}

/// Run the (enabled) preprocessing steps and return the prepared image as PNG
/// bytes. Returns the original bytes unchanged when no option is enabled.
fn prepare_bytes(image: &[u8], opts: preprocess::PreOpts) -> Result<Vec<u8>, String> {
    if !opts.any() {
        return Ok(image.to_vec());
    }
    let img = decode_rgb(image)?;
    let (processed, _transform) = preprocess::preprocess(img, &opts);
    encode_png(processed)
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()?
        .parse::<usize>()
        .ok()
        .filter(|&n| n > 0)
}

fn new_engine(
    model_size: &str,
    threads: Option<usize>,
    rec_batch: Option<usize>,
    det_max_side: Option<i64>,
    det_min_side: Option<i64>,
    rec_min_width: Option<usize>,
) -> PyResult<Engine> {
    if threads == Some(0) || rec_batch == Some(0) {
        return Err(PyValueError::new_err(
            "threads and rec_batch must be positive",
        ));
    }
    if det_max_side.is_some_and(|v| v > 0 && v < 32) || det_min_side.is_some_and(|v| v < 0) {
        return Err(PyValueError::new_err(
            "det_max_side must be at least 32 (or <=0 for 4000); det_min_side must be nonnegative",
        ));
    }
    if rec_min_width.is_some_and(|v| !(64..=3200).contains(&v)) {
        return Err(PyValueError::new_err(
            "rec_min_width must be between 64 and 3200",
        ));
    }
    let t = threads
        .or_else(|| env_usize("OCR_THREADS"))
        .unwrap_or_else(hardware::physical_cores)
        .max(1);
    let det_t = env_usize("OCR_DET_THREADS").unwrap_or(t.min(8)).clamp(1, t);
    let rb = rec_batch.unwrap_or(ocr::DEFAULT_REC_BATCH);
    let det_max = det_max_side.unwrap_or(ocr::DEFAULT_DET_MAX_SIDE);
    let pool = env_usize("REC_POOL")
        .unwrap_or(t.min(if model_size == "medium" { 8 } else { 32 }))
        .clamp(1, t);
    let m = resolve_model(model_size)?;
    let mut engine = Engine::from_memory(
        &m.det,
        &m.rec,
        m.dict,
        t,
        det_t,
        rb,
        m.box_thresh,
        pool,
        det_max,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("failed to init OCR engine: {e}")))?;
    // Short-label validation supports smaller tensors for tiny/small. Medium
    // retains the reference floor until it has equivalent coverage.
    let width = rec_min_width
        .or_else(|| env_usize("OCR_REC_MIN_WIDTH"))
        .unwrap_or(match model_size {
            "tiny" => 64,
            "small" => 96,
            _ => 320,
        })
        .clamp(64, 3200);
    engine.set_input_limits(det_min_side, Some(width));
    Ok(engine)
}

/// Plain-Rust OCR output (GIL-free), assembled into a Python dict afterwards.
type RawResult = (
    String,
    String,
    Vec<(usize, [i32; 2], [i32; 2], String, f32)>,
);

fn run_ocr(
    engine: &Mutex<Engine>,
    bytes: &[u8],
    opts: preprocess::PreOpts,
) -> Result<RawResult, String> {
    let dbg = std::env::var("OCR_DEBUG").is_ok();
    let t0 = std::time::Instant::now();
    let img = decode_rgb(bytes)?;
    if dbg {
        eprintln!(
            "[dbg] decode ({}x{}): {:.3}s",
            img.w,
            img.h,
            t0.elapsed().as_secs_f64()
        );
    }
    // Preprocessing may resize/rotate the image; `transform` maps detected boxes
    // back to the ORIGINAL image coordinates so returned bounds stay aligned.
    let workers = engine.lock().map_err(|e| e.to_string())?.workers.clone();
    let (img, transform) = workers.install(|| preprocess::preprocess(img, &opts));
    let mut eng = engine.lock().map_err(|e| e.to_string())?;
    let res = eng.run(&img).map_err(|e| e.to_string())?;
    drop(eng);
    Ok(finish_ocr(res, transform))
}

fn finish_ocr(res: Vec<ocr::OcrResult>, transform: preprocess::Transform) -> RawResult {
    let dbg = std::env::var("OCR_DEBUG").is_ok();
    let tl = std::time::Instant::now();
    // Text/layout run in the (straightened, scaled) preprocessed space.
    let (text, bounds) = layout::extract_text_and_bounds(&res);
    let structured = layout::structured_text(&res);
    if dbg {
        eprintln!("[dbg] layout: {:.3}s", tl.elapsed().as_secs_f64());
    }
    // Bounds are mapped back to original-image coordinates for the caller.
    let items = bounds
        .into_iter()
        .map(|(i, b)| {
            let mapped = transform.map_box([
                b.top_left[0],
                b.top_left[1],
                b.bottom_right[0],
                b.bottom_right[1],
            ]);
            (
                i,
                [mapped[0], mapped[1]],
                [mapped[2], mapped[3]],
                b.text,
                b.confidence,
            )
        })
        .collect();
    (text, structured, items)
}

/// Decode bounded windows in parallel, then share recognition work across pages.
fn run_ocr_batch(
    engine: &Mutex<Engine>,
    images: &[PyBackedBytes],
    opts: preprocess::PreOpts,
    batch_size: usize,
) -> Result<Vec<RawResult>, String> {
    use rayon::prelude::*;
    let workers = engine.lock().map_err(|e| e.to_string())?.workers.clone();
    let mut output = Vec::with_capacity(images.len());
    for chunk in images.chunks(batch_size) {
        let prepared: Result<Vec<_>, String> = workers.install(|| {
            chunk
                .par_iter()
                .enumerate()
                .map(|(i, bytes)| {
                    let img = decode_rgb(bytes)
                        .map_err(|e| format!("image {}: {e}", output.len() + i))?;
                    Ok(preprocess::preprocess(img, &opts))
                })
                .collect()
        });
        let (imgs, transforms): (Vec<_>, Vec<_>) = prepared?.into_iter().unzip();
        let results = engine
            .lock()
            .map_err(|e| e.to_string())?
            .run_many(&imgs)
            .map_err(|e| e.to_string())?;
        output.extend(workers.install(|| {
            results
                .into_par_iter()
                .zip(transforms)
                .map(|(res, tr)| finish_ocr(res, tr))
                .collect::<Vec<_>>()
        }));
    }
    Ok(output)
}

fn build_dict<'py>(py: Python<'py>, raw: RawResult) -> PyResult<Bound<'py, PyDict>> {
    let (text, structured, items) = raw;
    let out = PyDict::new(py);
    out.set_item("text", text)?;
    out.set_item("structured_text", structured)?;
    let bounds = PyDict::new(py);
    for (i, tl, br, t, conf) in items {
        let entry = PyDict::new(py);
        entry.set_item("topLeftCoord", PyTuple::new(py, [tl[0], tl[1]])?)?;
        entry.set_item("bottomRightCoord", PyTuple::new(py, [br[0], br[1]])?)?;
        entry.set_item("text", t)?;
        entry.set_item("confidence", conf)?;
        bounds.set_item(i, entry)?;
    }
    out.set_item("bounds", bounds)?;
    Ok(out)
}

/// A reusable OCR engine holding the loaded ONNX sessions. Construct once and
/// reuse across images. Thread-safe: calls are serialized internally and the
/// GIL is released during inference.
#[pyclass]
struct OcrEngine {
    inner: Mutex<Engine>,
}

#[pymethods]
impl OcrEngine {
    /// Create an engine.
    ///
    /// Args:
    ///     model_size: "tiny" (default, bundled), "small" (bundled), or "medium"
    ///         (downloaded once and cached on first use).
    ///     threads: total CPU budget; defaults to available physical cores,
    ///         respecting affinity and container limits.
    ///     rec_batch: actual recognition batch cap (default 1).
    ///     det_max_side: detector longer-side cap (default 1600).
    ///     det_min_side: detector minimum short side (default 736; 0 disables
    ///         minimum-side upscaling, with multiple-of-32 rounding retained).
    ///     rec_min_width: padding floor (tiny=64, small=96, medium=320).
    ///         Use 320 for reference padding. Changes can affect recognition.
    #[new]
    #[pyo3(signature = (model_size="tiny", threads=None, rec_batch=None, det_max_side=None, *, det_min_side=None, rec_min_width=None))]
    fn new(
        py: Python<'_>,
        model_size: &str,
        threads: Option<usize>,
        rec_batch: Option<usize>,
        det_max_side: Option<i64>,
        det_min_side: Option<i64>,
        rec_min_width: Option<usize>,
    ) -> PyResult<Self> {
        // Model construction and downloads run without holding the GIL.
        let size = model_size.to_string();
        let engine = py.allow_threads(|| {
            new_engine(
                &size,
                threads,
                rec_batch,
                det_max_side,
                det_min_side,
                rec_min_width,
            )
        })?;
        Ok(Self {
            inner: Mutex::new(engine),
        })
    }

    /// OCR several encoded images, preserving order. A bounded window shares
    /// recognition workers across pages; detection retains each page's shape.
    #[pyo3(signature = (images, *, batch_size=4, resize=false, denoise=false, deskew=false, binarize=false))]
    fn ocr_batch<'py>(
        &self,
        py: Python<'py>,
        images: Vec<PyBackedBytes>,
        batch_size: usize,
        resize: bool,
        denoise: bool,
        deskew: bool,
        binarize: bool,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be positive"));
        }
        let opts = preprocess::PreOpts {
            resize,
            denoise,
            deskew,
            binarize,
        };
        let raw = py
            .allow_threads(|| run_ocr_batch(&self.inner, &images, opts, batch_size))
            .map_err(PyRuntimeError::new_err)?;
        raw.into_iter().map(|r| build_dict(py, r)).collect()
    }

    /// Resolved CPU and input-shape settings for diagnostics and reproducibility.
    #[getter]
    fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let settings = py
            .allow_threads(|| {
                self.inner
                    .lock()
                    .map(|e| e.settings())
                    .map_err(|e| e.to_string())
            })
            .map_err(PyRuntimeError::new_err)?;
        let out = PyDict::new(py);
        for (key, value) in settings {
            out.set_item(key, value)?;
        }
        Ok(out)
    }

    /// Run OCR on raw encoded image bytes (jpeg/png/webp/bmp/tiff/gif).
    ///
    /// Optional preprocessing (applied in this order): ``resize`` (down to
    /// ≤2100×3000), ``denoise`` (fast NLM), ``deskew``, ``binarize`` (Sauvola).
    /// Returns ``{"text": str, "structured_text": str, "bounds": {idx: {...}}}``.
    #[pyo3(signature = (image, resize=false, denoise=false, deskew=false, binarize=false))]
    fn ocr<'py>(
        &self,
        py: Python<'py>,
        image: &[u8],
        resize: bool,
        denoise: bool,
        deskew: bool,
        binarize: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let opts = preprocess::PreOpts {
            resize,
            denoise,
            deskew,
            binarize,
        };
        let raw = py
            .allow_threads(|| run_ocr(&self.inner, image, opts))
            .map_err(PyRuntimeError::new_err)?;
        build_dict(py, raw)
    }

    /// Run OCR on a base64-encoded image string (same payload as the original
    /// paddle-ocr-api ``image_base64`` field). See ``ocr`` for the options.
    #[pyo3(signature = (image_base64, resize=false, denoise=false, deskew=false, binarize=false))]
    fn ocr_base64<'py>(
        &self,
        py: Python<'py>,
        image_base64: &str,
        resize: bool,
        denoise: bool,
        deskew: bool,
        binarize: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let bytes = py
            .allow_threads(|| {
                base64::engine::general_purpose::STANDARD.decode(image_base64.as_bytes())
            })
            .map_err(|e| PyValueError::new_err(format!("invalid base64: {e}")))?;
        let opts = preprocess::PreOpts {
            resize,
            denoise,
            deskew,
            binarize,
        };
        let raw = py
            .allow_threads(|| run_ocr(&self.inner, &bytes, opts))
            .map_err(PyRuntimeError::new_err)?;
        build_dict(py, raw)
    }

    /// Apply the enabled preprocessing steps (resize, denoise, deskew, binarize)
    /// in one pass and return the prepared image as PNG bytes (this does not run
    /// OCR). If every option is ``False`` the original bytes are returned
    /// unchanged.
    #[pyo3(signature = (image, resize=false, denoise=false, deskew=false, binarize=false))]
    fn prepare<'py>(
        &self,
        py: Python<'py>,
        image: &[u8],
        resize: bool,
        denoise: bool,
        deskew: bool,
        binarize: bool,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let opts = preprocess::PreOpts {
            resize,
            denoise,
            deskew,
            binarize,
        };
        let out = py
            .allow_threads(|| {
                let workers = self
                    .inner
                    .lock()
                    .map_err(|e| e.to_string())?
                    .workers
                    .clone();
                workers.install(|| prepare_bytes(image, opts))
            })
            .map_err(PyRuntimeError::new_err)?;
        Ok(PyBytes::new(py, &out))
    }
}

// ---- module-level convenience using a lazily-built default engine ----
static DEFAULT_ENGINE: OnceLock<Mutex<Engine>> = OnceLock::new();
static DEFAULT_INIT: Mutex<()> = Mutex::new(());

fn default_engine() -> PyResult<&'static Mutex<Engine>> {
    if let Some(e) = DEFAULT_ENGINE.get() {
        return Ok(e);
    }
    let _init = DEFAULT_INIT
        .lock()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    if let Some(e) = DEFAULT_ENGINE.get() {
        return Ok(e);
    }
    let eng = new_engine("tiny", None, None, None, None, None)?;
    Ok(DEFAULT_ENGINE.get_or_init(|| Mutex::new(eng)))
}

/// OCR multiple images with the shared default engine.
#[pyfunction]
#[pyo3(name = "ocr_batch", signature = (images, *, batch_size=4, resize=false, denoise=false, deskew=false, binarize=false))]
fn py_ocr_batch<'py>(
    py: Python<'py>,
    images: Vec<PyBackedBytes>,
    batch_size: usize,
    resize: bool,
    denoise: bool,
    deskew: bool,
    binarize: bool,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    if batch_size == 0 {
        return Err(PyValueError::new_err("batch_size must be positive"));
    }
    if images.is_empty() {
        return Ok(Vec::new());
    }
    let opts = preprocess::PreOpts {
        resize,
        denoise,
        deskew,
        binarize,
    };
    let engine = py.allow_threads(default_engine)?;
    let raw = py
        .allow_threads(|| run_ocr_batch(engine, &images, opts, batch_size))
        .map_err(PyRuntimeError::new_err)?;
    raw.into_iter().map(|r| build_dict(py, r)).collect()
}

/// OCR raw encoded image bytes using a shared default engine.
#[pyfunction]
#[pyo3(name = "ocr", signature = (image, resize=false, denoise=false, deskew=false, binarize=false))]
fn py_ocr<'py>(
    py: Python<'py>,
    image: &[u8],
    resize: bool,
    denoise: bool,
    deskew: bool,
    binarize: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = preprocess::PreOpts {
        resize,
        denoise,
        deskew,
        binarize,
    };
    let engine = py.allow_threads(default_engine)?;
    let raw = py
        .allow_threads(|| run_ocr(engine, image, opts))
        .map_err(PyRuntimeError::new_err)?;
    build_dict(py, raw)
}

/// OCR a base64-encoded image using a shared default engine.
#[pyfunction]
#[pyo3(name = "ocr_base64", signature = (image_base64, resize=false, denoise=false, deskew=false, binarize=false))]
fn py_ocr_base64<'py>(
    py: Python<'py>,
    image_base64: &str,
    resize: bool,
    denoise: bool,
    deskew: bool,
    binarize: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let bytes = py
        .allow_threads(|| base64::engine::general_purpose::STANDARD.decode(image_base64.as_bytes()))
        .map_err(|e| PyValueError::new_err(format!("invalid base64: {e}")))?;
    let opts = preprocess::PreOpts {
        resize,
        denoise,
        deskew,
        binarize,
    };
    let engine = py.allow_threads(default_engine)?;
    let raw = py
        .allow_threads(|| run_ocr(engine, &bytes, opts))
        .map_err(PyRuntimeError::new_err)?;
    build_dict(py, raw)
}

/// Apply the enabled preprocessing steps and return the prepared image as PNG
/// bytes (no OCR). Returns the original bytes unchanged when no option is set.
#[pyfunction]
#[pyo3(name = "prepare", signature = (image, resize=false, denoise=false, deskew=false, binarize=false))]
fn py_prepare<'py>(
    py: Python<'py>,
    image: &[u8],
    resize: bool,
    denoise: bool,
    deskew: bool,
    binarize: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let opts = preprocess::PreOpts {
        resize,
        denoise,
        deskew,
        binarize,
    };
    let out = py
        .allow_threads(|| prepare_bytes(image, opts))
        .map_err(PyRuntimeError::new_err)?;
    Ok(PyBytes::new(py, &out))
}

#[pymodule]
fn faster_paddle(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__runtime_build__", ort::info())?;
    m.add_class::<OcrEngine>()?;
    m.add_function(wrap_pyfunction!(py_ocr, m)?)?;
    m.add_function(wrap_pyfunction!(py_ocr_batch, m)?)?;
    m.add_function(wrap_pyfunction!(py_ocr_base64, m)?)?;
    m.add_function(wrap_pyfunction!(py_prepare, m)?)?;
    Ok(())
}
