"""Integration tests for faster_paddle.

Self-contained — uses the bundled fixture image under tests/fixtures/.

    pip install pytest pillow && pytest faster_paddle/tests -q
or directly:
    python faster_paddle/tests/test_integration.py
"""
import io
import os
import time

import faster_paddle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "document.png")  # two-pane UI screenshot


def _small():
    with open(FIXTURE, "rb") as f:
        return f.read()


def _big():
    """An image larger than the 2100x3000 OCR canvas, so resize=True triggers."""
    im = Image.open(FIXTURE).convert("RGB").resize((2400, 3200))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def test_result_shape():
    r = faster_paddle.OcrEngine().ocr(_small())
    assert set(r.keys()) == {"text", "structured_text", "bounds"}
    assert len(r["bounds"]) > 50
    b = next(iter(r["bounds"].values()))
    assert set(b.keys()) == {"topLeftCoord", "bottomRightCoord", "text", "confidence"}


def test_known_text_detected():
    r = faster_paddle.OcrEngine().ocr(_small())
    norm = "".join(ch for ch in r["text"].upper() if ch.isalnum())
    assert "INFRAESTRUTURA" in norm
    assert "IPCMAPSMUNICIPIO" in norm


def test_rec_pool_is_deterministic():
    """Output must be identical regardless of the rec session-pool size."""
    data = _big()

    def run(pool):
        os.environ["REC_POOL"] = str(pool)
        r = faster_paddle.OcrEngine().ocr(data)
        return [(b["topLeftCoord"], b["bottomRightCoord"], b["text"]) for b in r["bounds"].values()]

    try:
        a, b = run(1), run(4)
    finally:
        os.environ.pop("REC_POOL", None)
    assert a == b, "rec pool changed the result (must be deterministic)"


def test_bounds_in_original_coords_after_resize():
    """With resize=True the engine works on a smaller image, but bounds must be
    mapped back to the ORIGINAL image coordinate space (not the resized one)."""
    data = _big()
    eng = faster_paddle.OcrEngine()

    def max_xy(**kw):
        r = eng.ocr(data, **kw)
        xs = [c for b in r["bounds"].values() for c in (b["topLeftCoord"][0], b["bottomRightCoord"][0])]
        ys = [c for b in r["bounds"].values() for c in (b["topLeftCoord"][1], b["bottomRightCoord"][1])]
        return max(xs), max(ys)

    base_x, base_y = max_xy()                 # no resize -> original coords
    res_x, res_y = max_xy(resize=True)        # resize -> must still be original coords
    # if bounds were left in the resized (<=2100x3000) space they'd be much smaller
    assert res_x > 0.8 * base_x, f"x looks resized: {res_x} vs {base_x}"
    assert res_y > 0.8 * base_y, f"y looks resized: {res_y} vs {base_y}"


def test_preprocessing_options_run():
    data = _small()
    eng = faster_paddle.OcrEngine()
    for kw in (
        {"resize": True},
        {"denoise": True},
        {"deskew": True},
        {"binarize": True},
        {"resize": True, "denoise": True, "deskew": True, "binarize": True},
    ):
        r = eng.ocr(data, **kw)
        assert len(r["bounds"]) > 30, f"{kw} produced too few boxes"


def test_prepare_returns_image_bytes():
    data = _small()
    eng = faster_paddle.OcrEngine()

    # all options off -> original bytes returned unchanged
    assert eng.prepare(data) == data
    assert faster_paddle.prepare(data) == data

    # resize -> a decodable (color) PNG
    p = eng.prepare(data, resize=True)
    assert isinstance(p, bytes)
    Image.open(io.BytesIO(p)).verify()

    # binarize -> grayscale PNG with only black/white pixels
    pb = eng.prepare(data, binarize=True)
    im = Image.open(io.BytesIO(pb))
    assert im.mode == "L"
    assert set(im.convert("L").get_flattened_data()) <= {0, 255}

    # deskew expands the canvas of a rotated input
    rot = Image.open(FIXTURE).convert("RGB").rotate(8, expand=True, fillcolor=(255, 255, 255))
    buf = io.BytesIO(); rot.save(buf, format="PNG")
    pd = Image.open(io.BytesIO(eng.prepare(buf.getvalue(), deskew=True)))
    assert pd.size[0] >= rot.size[0] and pd.size[1] >= rot.size[1]

    # a prepared image still OCRs
    r = eng.ocr(pb)
    norm = "".join(c for c in r["text"].upper() if c.isalnum())
    assert "INFRAESTRUTURA" in norm


def test_speed_regression_guard():
    data = _big()
    eng = faster_paddle.OcrEngine()
    eng.ocr(data)  # warm up
    times = []
    for _ in range(3):
        t = time.time()
        eng.ocr(data)
        times.append(time.time() - t)
    median = sorted(times)[len(times) // 2]
    assert median < 6.0, f"OCR too slow: {median:.2f}s"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        t = time.time()
        fn()
        print(f"  PASS {fn.__name__} ({time.time()-t:.2f}s)")
    print(f"All {len(fns)} integration tests passed.")
