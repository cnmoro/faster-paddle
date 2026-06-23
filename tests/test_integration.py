"""Integration tests for faster_paddle.

Run from the repo root (so the test images resolve):
    pip install pytest && pytest faster_paddle/tests -q
or directly:
    python faster_paddle/tests/test_integration.py
"""
import io
import os
import time

import faster_paddle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # repo root (FasterPaddle/)
BIG = os.path.join(ROOT, "paddle-ocr-api", "test.jpg")   # 3157x4464 newspaper
UI = os.path.join(ROOT, "test2.png")                     # two-pane UI screenshot


def _img(path):
    with open(path, "rb") as f:
        return f.read()


def test_result_shape():
    r = faster_paddle.OcrEngine().ocr(_img(UI))
    assert set(r.keys()) == {"text", "structured_text", "bounds"}
    assert len(r["bounds"]) > 50
    b = next(iter(r["bounds"].values()))
    assert set(b.keys()) == {"topLeftCoord", "bottomRightCoord", "text", "confidence"}


def test_known_text_detected():
    r = faster_paddle.OcrEngine().ocr(_img(UI))
    # normalize: the recognizer may read '_' as ' ' (or drop it), so compare on
    # an alphanumeric-only, uppercased form.
    norm = "".join(ch for ch in r["text"].upper() if ch.isalnum())
    assert "INFRAESTRUTURA" in norm
    assert "IPCMAPSMUNICIPIO" in norm


def test_rec_pool_is_deterministic():
    """Output must be identical regardless of the rec session-pool size."""
    data = _img(BIG)

    def run(pool):
        os.environ["REC_POOL"] = str(pool)
        eng = faster_paddle.OcrEngine()
        r = eng.ocr(data)
        return [
            (b["topLeftCoord"], b["bottomRightCoord"], b["text"]) for b in r["bounds"].values()
        ]

    try:
        a = run(1)
        b = run(4)
    finally:
        os.environ.pop("REC_POOL", None)
    assert a == b, "rec pool changed the result (must be deterministic)"


def test_bounds_in_original_coords_after_resize():
    """With resize=True the engine works on a smaller image, but bounds must be
    mapped back to the ORIGINAL image coordinate space."""
    from PIL import Image

    im = Image.open(BIG)
    W, H = im.size  # 3157 x 4464, larger than the 2100x3000 OCR canvas

    eng = faster_paddle.OcrEngine()
    r = eng.ocr(_img(BIG), resize=True)
    xs = [c for b in r["bounds"].values() for c in (b["topLeftCoord"][0], b["bottomRightCoord"][0])]
    ys = [c for b in r["bounds"].values() for c in (b["topLeftCoord"][1], b["bottomRightCoord"][1])]
    # if bounds were left in the resized (<=2100x3000) space this would fail
    assert max(xs) > 2200, f"max x {max(xs)} looks like resized space, not original"
    assert max(ys) > 3100, f"max y {max(ys)} looks like resized space, not original"
    assert max(xs) <= W + 2 and max(ys) <= H + 2


def test_preprocessing_options_run():
    data = _img(UI)
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


def test_speed_regression_guard():
    """Sanity guard: the big newspaper image should OCR well under 6s after the
    optimization round (was ~22.7s in PaddlePaddle, ~4s before this round)."""
    data = _img(BIG)
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
