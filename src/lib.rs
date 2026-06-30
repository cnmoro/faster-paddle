//! Python bindings for FasterPaddle — a fast, CPU-only OCR engine specialized
//! for PaddleOCR's lightweight PP-OCRv6 *tiny* detection + recognition models.
//!
//! The ONNX models and character dictionary are embedded in the compiled
//! extension, so the wheel is fully self-contained — no model files or network
//! access are needed at runtime.

mod cv;
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
    pub unsafe extern "C" fn __isoc23_strtol(s: *const c_char, e: *mut *mut c_char, b: c_int) -> c_long {
        strtol(s, e, b)
    }
    #[no_mangle]
    pub unsafe extern "C" fn __isoc23_strtoll(s: *const c_char, e: *mut *mut c_char, b: c_int) -> c_longlong {
        strtoll(s, e, b)
    }
    #[no_mangle]
    pub unsafe extern "C" fn __isoc23_strtoull(s: *const c_char, e: *mut *mut c_char, b: c_int) -> c_ulonglong {
        strtoull(s, e, b)
    }

    // glibc 2.32 introduced `__libc_single_threaded` (a byte that Rust's std reads
    // to skip atomics in single-threaded processes). The prebuilt std references
    // it, so on glibc < 2.32 the extension fails with
    // `undefined symbol: __libc_single_threaded` (hit on aarch64 wheels, whose
    // other symbols top out at glibc 2.28 — e.g. Debian 11 / Ubuntu 20.04 arm64).
    // Provide it as 0 ("not single-threaded" — always-safe: std just keeps using
    // atomics), so the wheel loads on glibc 2.28+.
    #[no_mangle]
    pub static __libc_single_threaded: u8 = 0;
}

use base64::Engine as _;
use ocr::{Engine, ImageBgr};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
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
        .ok_or_else(|| PyRuntimeError::new_err("cannot determine a cache directory for medium models"))?
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
    let url = format!(
        "https://github.com/cnmoro/faster-paddle/releases/download/v{VERSION}/{asset}"
    );
    let resp = ureq::get(&url)
        .call()
        .map_err(|e| PyRuntimeError::new_err(format!("failed to download medium model from {url}: {e}")))?;
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
    std::fs::write(&tmp, &bytes).map_err(|e| PyRuntimeError::new_err(format!("cannot write cache: {e}")))?;
    let _ = std::fs::rename(&tmp, &path);
    Ok(bytes)
}

/// Number of physical CPU cores (best for compute-bound inference; SMT threads
/// tend to slow it down). Falls back to logical parallelism off Linux.
fn physical_cores() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(txt) = std::fs::read_to_string("/proc/cpuinfo") {
            let mut seen = std::collections::HashSet::new();
            let (mut phys, mut core) = (String::new(), String::new());
            for line in txt.lines() {
                if let Some(v) = line.strip_prefix("physical id") {
                    phys = v.split(':').nth(1).unwrap_or("").trim().to_string();
                } else if let Some(v) = line.strip_prefix("core id") {
                    core = v.split(':').nth(1).unwrap_or("").trim().to_string();
                } else if line.trim().is_empty() {
                    if !phys.is_empty() && !core.is_empty() {
                        seen.insert((phys.clone(), core.clone()));
                    }
                    phys.clear();
                    core.clear();
                }
            }
            if !seen.is_empty() {
                return seen.len();
            }
        }
    }
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8)
}

/// Decode encoded image bytes (jpeg/png/...) into a BGR buffer matching
/// cv2.imread channel order (PaddleOCR feeds the network BGR).
fn decode_bgr(bytes: &[u8]) -> Result<ImageBgr, String> {
    let img = image::load_from_memory(bytes).map_err(|e| e.to_string())?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let raw = rgb.into_raw();
    let mut data = vec![0u8; w * h * 3];
    for i in 0..w * h {
        data[i * 3] = raw[i * 3 + 2]; // B
        data[i * 3 + 1] = raw[i * 3 + 1]; // G
        data[i * 3 + 2] = raw[i * 3]; // R
    }
    Ok(ImageBgr { w, h, data })
}

/// Encode a BGR image to PNG bytes. If the image is grayscale (all channels
/// equal, e.g. after denoise/deskew/binarize) it is written as a smaller 8-bit
/// grayscale PNG; otherwise as RGB.
fn encode_png(img: &ImageBgr) -> Result<Vec<u8>, String> {
    let n = img.w * img.h;
    let is_gray = (0..n).all(|i| img.data[i * 3] == img.data[i * 3 + 1] && img.data[i * 3 + 1] == img.data[i * 3 + 2]);
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
        let mut rgb = vec![0u8; n * 3];
        for i in 0..n {
            rgb[i * 3] = img.data[i * 3 + 2]; // R
            rgb[i * 3 + 1] = img.data[i * 3 + 1]; // G
            rgb[i * 3 + 2] = img.data[i * 3]; // B
        }
        let buf = image::RgbImage::from_raw(img.w as u32, img.h as u32, rgb)
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
    let img = decode_bgr(image)?;
    let (processed, _transform) = preprocess::preprocess(img, &opts);
    encode_png(&processed)
}

fn new_engine(model_size: &str, threads: Option<usize>, rec_batch: Option<usize>) -> PyResult<Engine> {
    let t = threads.unwrap_or_else(physical_cores).max(1);
    let rb = rec_batch.unwrap_or(ocr::DEFAULT_REC_BATCH);
    // Recognition session-pool size: run several rec sessions concurrently so the
    // many small rec matmuls keep the cores busy. The general heuristic is ~2 ORT
    // threads per session (small matmuls don't scale past that); this scales with
    // the core count. Capped at 8 to bound memory (each session holds a copy of
    // the rec weights). Override with REC_POOL.
    let pool = std::env::var("REC_POOL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| (t / 2).clamp(1, 8));
    let m = resolve_model(model_size)?;
    Engine::from_memory(&m.det, &m.rec, m.dict, t, rb, m.box_thresh, pool)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to init OCR engine: {e}")))
}

/// Plain-Rust OCR output (GIL-free), assembled into a Python dict afterwards.
type RawResult = (String, String, Vec<(usize, [i32; 2], [i32; 2], String, f32)>);

fn run_ocr(engine: &Mutex<Engine>, bytes: &[u8], opts: preprocess::PreOpts) -> Result<RawResult, String> {
    let img = decode_bgr(bytes)?;
    // Preprocessing may resize/rotate the image; `transform` maps detected boxes
    // back to the ORIGINAL image coordinates so returned bounds stay aligned.
    let (img, transform) = if opts.any() {
        preprocess::preprocess(img, &opts)
    } else {
        (img, preprocess::Transform::identity())
    };
    let mut eng = engine.lock().unwrap();
    let res = eng.run(&img).map_err(|e| e.to_string())?;
    // Text/layout run in the (straightened, scaled) preprocessed space.
    let (text, bounds) = layout::extract_text_and_bounds(&res);
    let structured = layout::structured_text(&res);
    // Bounds are mapped back to original-image coordinates for the caller.
    let items = bounds
        .into_iter()
        .map(|(i, b)| {
            let mapped = transform.map_box([b.top_left[0], b.top_left[1], b.bottom_right[0], b.bottom_right[1]]);
            (i, [mapped[0], mapped[1]], [mapped[2], mapped[3]], b.text, b.confidence)
        })
        .collect();
    Ok((text, structured, items))
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
    ///     threads: ONNX Runtime intra-op threads. Defaults to physical cores.
    ///     rec_batch: recognition batch size (default 6).
    #[new]
    #[pyo3(signature = (model_size="tiny", threads=None, rec_batch=None))]
    fn new(py: Python<'_>, model_size: &str, threads: Option<usize>, rec_batch: Option<usize>) -> PyResult<Self> {
        // medium may download large files; release the GIL during construction.
        let size = model_size.to_string();
        let engine = py.allow_threads(|| new_engine(&size, threads, rec_batch))?;
        Ok(Self {
            inner: Mutex::new(engine),
        })
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
        let opts = preprocess::PreOpts { resize, denoise, deskew, binarize };
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
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(image_base64.as_bytes())
            .map_err(|e| PyValueError::new_err(format!("invalid base64: {e}")))?;
        let opts = preprocess::PreOpts { resize, denoise, deskew, binarize };
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
        let opts = preprocess::PreOpts { resize, denoise, deskew, binarize };
        let out = py
            .allow_threads(|| prepare_bytes(image, opts))
            .map_err(PyRuntimeError::new_err)?;
        Ok(PyBytes::new(py, &out))
    }
}

// ---- module-level convenience using a lazily-built default engine ----
static DEFAULT_ENGINE: OnceLock<Mutex<Engine>> = OnceLock::new();

fn default_engine() -> PyResult<&'static Mutex<Engine>> {
    if let Some(e) = DEFAULT_ENGINE.get() {
        return Ok(e);
    }
    let eng = new_engine("tiny", None, None)?;
    Ok(DEFAULT_ENGINE.get_or_init(|| Mutex::new(eng)))
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
    let opts = preprocess::PreOpts { resize, denoise, deskew, binarize };
    let engine = default_engine()?;
    let raw = py.allow_threads(|| run_ocr(engine, image, opts)).map_err(PyRuntimeError::new_err)?;
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
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(image_base64.as_bytes())
        .map_err(|e| PyValueError::new_err(format!("invalid base64: {e}")))?;
    let opts = preprocess::PreOpts { resize, denoise, deskew, binarize };
    let engine = default_engine()?;
    let raw = py.allow_threads(|| run_ocr(engine, &bytes, opts)).map_err(PyRuntimeError::new_err)?;
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
    let opts = preprocess::PreOpts { resize, denoise, deskew, binarize };
    let out = py
        .allow_threads(|| prepare_bytes(image, opts))
        .map_err(PyRuntimeError::new_err)?;
    Ok(PyBytes::new(py, &out))
}

#[pymodule]
fn faster_paddle(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<OcrEngine>()?;
    m.add_function(wrap_pyfunction!(py_ocr, m)?)?;
    m.add_function(wrap_pyfunction!(py_ocr_base64, m)?)?;
    m.add_function(wrap_pyfunction!(py_prepare, m)?)?;
    Ok(())
}
