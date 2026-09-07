# faster-paddle

**Fast, CPU-only OCR in Rust with Python bindings** — a self-contained
reimplementation of PaddleOCR's PP-OCRv6 detection + recognition pipeline
powered by [ONNX Runtime](https://onnxruntime.ai/).

- ⚡ CPU latency optimizations: fused image transforms, compact DB components,
  dynamic recognition scheduling, and model-specific input widths.
- 🗂️ **Multi-image OCR** with `ocr_batch`: ordered results and shared recognition
  work across pages. [See batch usage](#multiple-images).
- 📦 **Self-contained** — the tiny + small ONNX models are bundled inside the
  wheel. No `paddlepaddle`, no model downloads for tiny/small.
- 🎚️ **Three model sizes**: `tiny` (default, fastest), `small`, and `medium`
  (higher accuracy; downloaded once on first use and cached).
- 🦀 Pure-Rust pre/post-processing (detection DB decode, `minAreaRect`,
  perspective crop, CTC decode, reading-order text reconstruction). No OpenCV.
- 🖥️ Prebuilt wheels for **Linux, Windows, macOS** (x86-64 + arm64).

Defaults automatically respect available physical cores, Linux CPU affinity,
and container CPU limits. Reuse an engine to amortize model/session loading.
See [performance results and tradeoffs](PERFORMANCE_CHANGES.md) and the
[benchmark harnesses](benchmarks/) for reproducible measurements.

---

## Install

```bash
pip install faster-paddle
```

## Usage

```python
import faster_paddle

# One-shot, using a shared default engine (lazily initialized):
with open("document.jpg", "rb") as f:
    result = faster_paddle.ocr(f.read())

print(result["text"])              # reading-order reconstructed text
for idx, b in result["bounds"].items():
    print(idx, b["text"], b["confidence"], b["topLeftCoord"], b["bottomRightCoord"])
```

Reuse an explicit engine (recommended for servers — load the models once):

```python
from faster_paddle import OcrEngine

# model_size: "tiny" (default), "small", or "medium"
engine = OcrEngine(model_size="tiny", threads=None, det_max_side=1600)

result = engine.ocr(image_bytes)                 # raw jpeg/png/webp/bmp/tiff/gif bytes
result = engine.ocr_base64(b64_string)           # base64-encoded image
```

### Multiple images

Available in **v1.0.2 and later**. Pass a list of encoded image bytes and get
one result dictionary per image, in the same order:

```python
from pathlib import Path
from faster_paddle import OcrEngine

paths = [Path("page1.png"), Path("page2.jpg"), Path("page3.png")]
images = [path.read_bytes() for path in paths]

engine = OcrEngine(model_size="small")
results = engine.ocr_batch(images, batch_size=4)
for path, result in zip(paths, results):
    print(path.name, result["text"])
```

The shared default tiny engine also supports batches:

```python
import faster_paddle

results = faster_paddle.ocr_batch(images, batch_size=4, resize=True)
```

`ocr_batch(images, *, batch_size=4, resize=False, denoise=False, deskew=False,
binarize=False)` processes bounded windows of **images**. It decodes/preprocesses
in parallel and schedules recognition crops across pages. Each page keeps its
own detector resolution and coordinate transform. `batch_size` controls the
number of decoded pages resident in a window; it is independent of `rec_batch`.
Input bytes are borrowed when possible. An empty list returns `[]`; invalid
images raise an error with their zero-based input index.

Batching helps fill recognition workers on sparse pages and can improve
throughput. It is not guaranteed to accelerate dense pages, and the caller waits
for the complete result list. Use `ocr` for minimum time to the first page's
result. Minor floating-point confidence differences can occur between the
single-line and multi-page execution paths.

### Optional preprocessing

`ocr`, `ocr_base64`, and `ocr_batch` take four optional flags (all default `False`), applied —
when enabled — in the optimal order, all in fast parallel Rust:

```python
result = engine.ocr(
    image_bytes,
    resize=True,     # 1. downscale source to ≤ 2100×3000 (aspect preserved) if larger
    denoise=True,    # 2. fast Non-Local-Means denoise (grayscale)
    deskew=True,     # 3. detect skew (Canny + Hough) and rotate to straighten
    binarize=True,   # 4. Sauvola adaptive thresholding (clean black/white)
)
```

Order rationale: resize first (everything downstream is then faster), denoise
before angle detection and thresholding, deskew on the cleaned image, binarize
last to produce the final B/W. `resize` can reduce source/crop work, but may add an extra resampling pass when
detection already hits its size cap; benchmark it on your input. Any of `denoise`/`deskew`/`binarize` converts the
image to grayscale.

Returned `bounds` are always in the **original image's coordinate space** — even
when `resize` or `deskew` changes the working image, the boxes are mapped back so
they line up with your input.

### Preprocess only (no OCR)

`prepare` runs the same preprocessing in one pass and returns the prepared image
as **PNG bytes** (grayscale once any of denoise/deskew/binarize is on, else
color). If every option is `False` the original bytes are returned unchanged.

```python
prepared = engine.prepare(image_bytes, resize=True, denoise=True, deskew=True, binarize=False)
# or module-level:  faster_paddle.prepare(image_bytes, resize=True, ...)

with open("prepared.png", "wb") as f:
    f.write(prepared)
# you can also feed it straight back in:
result = engine.ocr(prepared)
```

### Model sizes

| size     | bundled | det+rec | notes |
|----------|---------|---------|-------|
| `tiny`   | ✅ yes  | ~6 MB   | default, fastest, lightweight |
| `small`  | ✅ yes  | ~31 MB  | better accuracy |
| `medium` | ⬇️ on demand | ~138 MB | best accuracy; downloaded once from the GitHub release and cached under your user cache dir |

`tiny` and `small` are embedded in the wheel (offline). `medium` exceeds PyPI's
file-size limit, so the first `OcrEngine(model_size="medium")` downloads it once
(needs network that time only) and caches it for subsequent runs.

### Result shape

```python
{
  "text": "full reconstructed text...",
  "structured_text": "layout-preserving text (see below)",
  "bounds": {
     0: {
        "topLeftCoord":     (x1, y1),
        "bottomRightCoord": (x2, y2),
        "text":             "line text",
        "confidence":       0.97,
     },
     1: { ... },
  }
}
```

`text` and `bounds` match the JSON contract of the original `paddle-ocr-api`
service, so it is a drop-in replacement.

### `structured_text`

A spatial reconstruction that reads **left-to-right, top-to-bottom** while
preserving the visual layout: vertical whitespace gaps split the page into
columns/panes (each read fully before the next), and within each one the rows are
laid out as a monospace grid, so indentation (tree nesting) and aligned
sub-columns (key/value tables) are kept. Single-glyph UI icon noise is dropped.

Use **`structured_text`** for screenshots, forms, table/tree UIs, and code —
anything where spatial structure carries meaning. Use **`text`** for dense
multi-column prose: there the absolute pixel spacing of `structured_text`
produces very wide lines, so the column-merging `text` reconstruction reads
better. Both are always returned, so you can pick per use case.

Example `structured_text` for a two-pane file-tree + settings UI:

```
Project
 src (14)
   main.rs
   parser.rs
   utils.rs
 tests
 docs

Setting                                            Value
        max_connections                            128
        request_timeout_seconds                    30
        cache_size_mb                              512
```

## API

| | |
|---|---|
| `faster_paddle.ocr(image, resize=False, denoise=False, deskew=False, binarize=False) -> dict` | OCR encoded image bytes (shared default engine). |
| `faster_paddle.ocr_batch(images, *, batch_size=4, resize=False, denoise=False, deskew=False, binarize=False) -> list[dict]` | OCR multiple encoded images; results preserve input order. |
| `faster_paddle.ocr_base64(image_base64, resize=False, denoise=False, deskew=False, binarize=False) -> dict` | OCR a base64 image string. |
| `OcrEngine(model_size="tiny", threads=None, rec_batch=None, det_max_side=None, *, det_min_side=None, rec_min_width=None)` | Construct a reusable engine. |
| `OcrEngine.ocr(image, resize=False, denoise=False, deskew=False, binarize=False) -> dict` | OCR encoded image bytes. |
| `OcrEngine.ocr_batch(images, *, batch_size=4, resize=False, denoise=False, deskew=False, binarize=False) -> list[dict]` | OCR multiple encoded images; results preserve input order. |
| `OcrEngine.ocr_base64(image_base64, resize=False, denoise=False, deskew=False, binarize=False) -> dict` | OCR a base64 image string. |
| `faster_paddle.prepare(image, resize=False, denoise=False, deskew=False, binarize=False) -> bytes` | Preprocess only; returns PNG bytes (no OCR). |
| `OcrEngine.prepare(image, resize=False, denoise=False, deskew=False, binarize=False) -> bytes` | Preprocess only; returns PNG bytes (no OCR). |

- `resize`/`denoise`/`deskew`/`binarize`: optional preprocessing (see above).
- `model_size`: `"tiny"` (default), `"small"`, or `"medium"`.
- `threads`: total CPU budget; defaults to available physical cores, limited by
  affinity and OS/container quotas. An explicit positive value overrides discovery.
- `batch_size`: maximum decoded images per batch window (default **4**, must be positive).
- `rec_batch`: **actual maximum** crops per recognition tensor (default **1**).
  Independent crops share the worker pool. Larger caps are available for tuning.
- `det_max_side`: detector long-side limit (default **1600**); recognition crops
  still come from the original source. Lower values can miss small text.
- `det_min_side`: minimum detector short side (default **736**). Set **0** to
  disable minimum-side upscaling. Dimensions are still rounded to multiples of 32.
- `rec_min_width`: recognition padding floor, automatically **64 for tiny**, **96
  for small**, and **320 for medium**. Set **320** for the reference padding
  behavior. Shorter padding changes context and can change text/confidence;
  validate your languages and documents when migrating.
- `engine.config`: dictionary of resolved thread counts, pool size and input
  limits, for diagnostics and reproducible benchmarks.

Calls release the GIL. Inference on a shared engine is serialized; decode and
layout can overlap. Independent engines have independent CPU budgets, so set
`threads` explicitly when running several engines concurrently.

### Parallelism

- The detector uses up to **8** threads within the CPU budget.
- Recognition uses up to one worker per available physical core, capped at 32
  (8 for medium) to limit model memory. Jobs are scheduled dynamically, largest
  estimated jobs first. Sparse long-line jobs use a separate pool of up to four sessions with up to
  four threads each, within the same CPU budget. Session construction is parallel
  in bounded groups to reduce startup latency.
- An engine-local Rayon pool handles image and geometry work within the same
  CPU budget. ONNX workers spin during inference to reduce wakeup latency, and stop
  spinning immediately when their inference call finishes.
- Input tensor buffers are reused. Recognition budgets include actual padding.

Advanced overrides (read at engine construction): `OCR_THREADS`,
`OCR_DET_THREADS`, `REC_POOL`, `REC_BUDGET`, `RAYON_NUM_THREADS`, `OCR_REC_MIN_WIDTH`,
`OCR_DET_MIN_SIDE`, `OCR_MEMPAT`, `OCR_PREPACK`, `OCR_DET_SPIN`, `OCR_REC_SPIN`,
`OCR_APPROX_GELU`. Approximate GELU stays off: its measured end-to-end benefit
was inconsistent and it changed a detection.
Constructor arguments take precedence over their corresponding environment
values; detector/worker/Rayon counts are capped by the total CPU budget.
The CPU-count defaults are measured heuristics, not online autotuning; benchmark
other CPU architectures with `benchmarks/cpu_latency.py` before overriding them.

---

## How it works

The pipeline faithfully mirrors PaddleOCR's lightweight path:

1. **Detection** — resize (min-side 736, cap the longer side at `det_max_side`
   = 1600 by default vs PaddleOCR's 4000, round to ×32), normalize (BGR mean/std),
   run the DB detector. Lower detector resolution reduces work but can miss small text; recognition
   still crops from the full-res image.
2. **DB post-process** — threshold 0.2, connected components, `minAreaRect`,
   box score ≥ 0.4, `unclip` ratio 1.4, rescale to source coordinates.
3. **Sort** boxes top-to-bottom / left-to-right; **crop** each via perspective warp.
4. **Recognition** — resize each crop to H=48, normalize, batch, run the CTC
   recognizer (6,906 output classes for tiny, 18,710 for small), greedy CTC decode.
5. **Reconstruct** reading-order text with dynamic column/line detection.

Detection matches PaddlePaddle at **96 % IoU>0.5** with **0.93 character-level
similarity** on the recognized text; the residual difference is ONNX-Runtime vs
PaddlePaddle floating-point numerics, not the algorithm.

The bundled tiny and small PP-OCRv6 models were exported with `paddle2onnx`.

## Building from source

```bash
pip install maturin
maturin develop --release      # build + install into the current environment
# or
maturin build --release        # produce a wheel in target/wheels/
```

Requires a Rust toolchain. ONNX Runtime is fetched automatically by the `ort`
crate at build time and linked into the extension.

## Tests

```bash
cargo test --release                 # Rust unit tests (geometry, resize, CTC)
maturin develop --release            # then the Python integration tests:
python -m pytest tests -q
```

The tests cover geometry equivalence, fused resize/padding, recognition caps,
known text, both small models, batch ordering and transforms, buffer reuse,
concurrent calls, error recovery, CPU affinity/budgets, and preprocessing. CI
runs Rust and Python tests before publishing. Install `pytest` and `pillow`
for the Python suite. Performance comparisons live under `benchmarks/`; the
small bundled test corpus is not a comprehensive OCR accuracy benchmark.

## License

MIT
