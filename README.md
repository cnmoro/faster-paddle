# faster-paddle

**Fast, CPU-only OCR in Rust with Python bindings** — a self-contained
reimplementation of PaddleOCR's PP-OCRv6 detection + recognition pipeline
powered by [ONNX Runtime](https://onnxruntime.ai/).

- ⚡ **~9× faster** than `paddleocr` on CPU for the same models and output
  (parallel detection pre/post-processing + a concurrent recognition session pool).
- 📦 **Self-contained** — the tiny + small ONNX models are bundled inside the
  wheel. No `paddlepaddle`, no model downloads for tiny/small.
- 🎚️ **Three model sizes**: `tiny` (default, fastest), `small`, and `medium`
  (higher accuracy; downloaded once on first use and cached).
- 🦀 Pure-Rust pre/post-processing (detection DB decode, `minAreaRect`,
  perspective crop, CTC decode, reading-order text reconstruction). No OpenCV.
- 🖥️ Prebuilt wheels for **Linux, Windows, macOS** (x86-64 + arm64).

```
paddleocr (PaddlePaddle, CPU)        22.7 s / image
faster-paddle (Rust + ONNXRuntime)    2.5 s / image     →  ~9× faster
```
*(test image 3157×4464, AMD Ryzen 7 5800X3D; both after warm-up, same weights.)*

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
engine = OcrEngine(model_size="tiny", threads=None, rec_batch=6)

result = engine.ocr(image_bytes)                 # raw jpeg/png/webp/bmp/tiff/gif bytes
result = engine.ocr_base64(b64_string)           # base64-encoded image
```

### Optional preprocessing

`ocr` / `ocr_base64` take four optional flags (all default `False`), applied —
when enabled — in the optimal order, all in fast parallel Rust:

```python
result = engine.ocr(
    image_bytes,
    resize=True,     # 1. downscale to ≤ 2100×3000 (aspect preserved) if larger
    denoise=True,    # 2. fast Non-Local-Means denoise (grayscale)
    deskew=True,     # 3. detect skew (Canny + Hough) and rotate to straighten
    binarize=True,   # 4. Sauvola adaptive thresholding (clean black/white)
)
```

Order rationale: resize first (everything downstream is then faster), denoise
before angle detection and thresholding, deskew on the cleaned image, binarize
last to produce the final B/W. Enabling `resize` typically makes OCR *faster*
overall (less detector work). Any of `denoise`/`deskew`/`binarize` converts the
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

Example `structured_text` for a two-pane database UI:

```
PNS
 Collections (11)
   System
   CAGED
   IPCMAPS_MUNICIPIO
 Functions
 Users

Key                                                Value
        OUTRAS_DESPESAS_POTENCIAL_DE_CONSUMO_EM... 7332964
        TOTAL_DO_CONSUMO_URBANO_E_RURAL            613855113
        CD_MUNI_IBGE                               1100015
```

## API

| | |
|---|---|
| `faster_paddle.ocr(image, resize=False, denoise=False, deskew=False, binarize=False) -> dict` | OCR encoded image bytes (shared default engine). |
| `faster_paddle.ocr_base64(image_base64, resize=False, denoise=False, deskew=False, binarize=False) -> dict` | OCR a base64 image string. |
| `OcrEngine(model_size="tiny", threads=None, rec_batch=None)` | Construct a reusable engine. |
| `OcrEngine.ocr(image, resize=False, denoise=False, deskew=False, binarize=False) -> dict` | OCR encoded image bytes. |
| `OcrEngine.ocr_base64(image_base64, resize=False, denoise=False, deskew=False, binarize=False) -> dict` | OCR a base64 image string. |
| `faster_paddle.prepare(image, resize=False, denoise=False, deskew=False, binarize=False) -> bytes` | Preprocess only; returns PNG bytes (no OCR). |
| `OcrEngine.prepare(image, resize=False, denoise=False, deskew=False, binarize=False) -> bytes` | Preprocess only; returns PNG bytes (no OCR). |

- `resize`/`denoise`/`deskew`/`binarize`: optional preprocessing (see above).
- `model_size`: `"tiny"` (default), `"small"`, or `"medium"`.
- `threads`: ONNX Runtime intra-op threads. Defaults to the number of **physical**
  CPU cores (SMT/logical threads tend to slow compute-bound inference down).
- `rec_batch`: recognition batch size (default 6).

Calls are thread-safe (serialized internally) and release the GIL during
inference.

---

## How it works

The pipeline faithfully mirrors PaddleOCR's lightweight path:

1. **Detection** — resize (min-side 736, clamp max-side 4000, round to ×32),
   normalize (BGR mean/std), run the DB detector.
2. **DB post-process** — threshold 0.2, connected components, `minAreaRect`,
   box score ≥ 0.4, `unclip` ratio 1.4, rescale to source coordinates.
3. **Sort** boxes top-to-bottom / left-to-right; **crop** each via perspective warp.
4. **Recognition** — resize each crop to H=48, normalize, batch, run the CTC
   recognizer (`[N, T, 6906]`), greedy CTC decode.
5. **Reconstruct** reading-order text with dynamic column/line detection.

Detection matches PaddlePaddle at **96 % IoU>0.5** with **0.93 character-level
similarity** on the recognized text; the residual difference is ONNX-Runtime vs
PaddlePaddle floating-point numerics, not the algorithm.

The bundled models are `PP-OCRv6_tiny_det` and `PP-OCRv6_tiny_rec` exported with
`paddle2onnx`.

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
python faster_paddle/tests/test_integration.py
```

The integration tests check the result shape, known-text detection, that the
recognition session pool is deterministic, that bounds map back to original
coordinates after `resize`, that all preprocessing options run, and a speed
regression guard.

## License

MIT
