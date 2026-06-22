//! Python bindings for FasterPaddle — a fast, CPU-only OCR engine specialized
//! for PaddleOCR's lightweight PP-OCRv6 *tiny* detection + recognition models.
//!
//! The ONNX models and character dictionary are embedded in the compiled
//! extension, so the wheel is fully self-contained — no model files or network
//! access are needed at runtime.

mod cv;
mod layout;
mod ocr;

use base64::Engine as _;
use ocr::{Engine, ImageBgr};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::sync::{Mutex, OnceLock};

// ---- embedded assets ----
const DET_ONNX: &[u8] = include_bytes!("../models/det.onnx");
const REC_ONNX: &[u8] = include_bytes!("../models/rec.onnx");
const CHAR_DICT_JSON: &str = include_str!("../models/char_dict.json");

fn load_char_dict() -> Vec<String> {
    serde_json::from_str(CHAR_DICT_JSON).expect("embedded char_dict.json is valid")
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

fn new_engine(threads: Option<usize>, rec_batch: Option<usize>) -> PyResult<Engine> {
    let t = threads.unwrap_or_else(physical_cores).max(1);
    let rb = rec_batch.unwrap_or(ocr::DEFAULT_REC_BATCH);
    Engine::from_memory(DET_ONNX, REC_ONNX, load_char_dict(), t, rb)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to init OCR engine: {e}")))
}

/// Plain-Rust OCR output (GIL-free), assembled into a Python dict afterwards.
type RawResult = (String, Vec<(usize, [i32; 2], [i32; 2], String, f32)>);

fn run_ocr(engine: &Mutex<Engine>, bytes: &[u8]) -> Result<RawResult, String> {
    let img = decode_bgr(bytes)?;
    let mut eng = engine.lock().unwrap();
    let res = eng.run(&img).map_err(|e| e.to_string())?;
    let (text, bounds) = layout::extract_text_and_bounds(&res);
    let items = bounds
        .into_iter()
        .map(|(i, b)| (i, b.top_left, b.bottom_right, b.text, b.confidence))
        .collect();
    Ok((text, items))
}

fn build_dict<'py>(py: Python<'py>, raw: RawResult) -> PyResult<Bound<'py, PyDict>> {
    let (text, items) = raw;
    let out = PyDict::new(py);
    out.set_item("text", text)?;
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
    ///     threads: ONNX Runtime intra-op threads. Defaults to physical cores.
    ///     rec_batch: recognition batch size (default 6).
    #[new]
    #[pyo3(signature = (threads=None, rec_batch=None))]
    fn new(threads: Option<usize>, rec_batch: Option<usize>) -> PyResult<Self> {
        Ok(Self {
            inner: Mutex::new(new_engine(threads, rec_batch)?),
        })
    }

    /// Run OCR on raw encoded image bytes (jpeg/png/webp/bmp/tiff/gif).
    /// Returns ``{"text": str, "bounds": {idx: {...}}}``.
    fn ocr<'py>(&self, py: Python<'py>, image: &[u8]) -> PyResult<Bound<'py, PyDict>> {
        let raw = py
            .allow_threads(|| run_ocr(&self.inner, image))
            .map_err(PyRuntimeError::new_err)?;
        build_dict(py, raw)
    }

    /// Run OCR on a base64-encoded image string (same payload as the original
    /// paddle-ocr-api ``image_base64`` field).
    fn ocr_base64<'py>(&self, py: Python<'py>, image_base64: &str) -> PyResult<Bound<'py, PyDict>> {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(image_base64.as_bytes())
            .map_err(|e| PyValueError::new_err(format!("invalid base64: {e}")))?;
        let raw = py
            .allow_threads(|| run_ocr(&self.inner, &bytes))
            .map_err(PyRuntimeError::new_err)?;
        build_dict(py, raw)
    }
}

// ---- module-level convenience using a lazily-built default engine ----
static DEFAULT_ENGINE: OnceLock<Mutex<Engine>> = OnceLock::new();

fn default_engine() -> PyResult<&'static Mutex<Engine>> {
    if let Some(e) = DEFAULT_ENGINE.get() {
        return Ok(e);
    }
    let eng = new_engine(None, None)?;
    Ok(DEFAULT_ENGINE.get_or_init(|| Mutex::new(eng)))
}

/// OCR raw encoded image bytes using a shared default engine.
#[pyfunction]
#[pyo3(name = "ocr")]
fn py_ocr<'py>(py: Python<'py>, image: &[u8]) -> PyResult<Bound<'py, PyDict>> {
    let engine = default_engine()?;
    let raw = py.allow_threads(|| run_ocr(engine, image)).map_err(PyRuntimeError::new_err)?;
    build_dict(py, raw)
}

/// OCR a base64-encoded image using a shared default engine.
#[pyfunction]
#[pyo3(name = "ocr_base64")]
fn py_ocr_base64<'py>(py: Python<'py>, image_base64: &str) -> PyResult<Bound<'py, PyDict>> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(image_base64.as_bytes())
        .map_err(|e| PyValueError::new_err(format!("invalid base64: {e}")))?;
    let engine = default_engine()?;
    let raw = py.allow_threads(|| run_ocr(engine, &bytes)).map_err(PyRuntimeError::new_err)?;
    build_dict(py, raw)
}

#[pymodule]
fn faster_paddle(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<OcrEngine>()?;
    m.add_function(wrap_pyfunction!(py_ocr, m)?)?;
    m.add_function(wrap_pyfunction!(py_ocr_base64, m)?)?;
    Ok(())
}
