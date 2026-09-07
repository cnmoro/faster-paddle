"""Synthetic labeled CPU/quality probe; requires Pillow and a built extension.

Run each variant in a fresh process using PYTHONPATH to select the extension.
The font can be supplied with --font. This is not a multilingual accuracy set.
"""
import argparse
import collections
import io
import json
import re
import statistics
import time
from pathlib import Path

import faster_paddle
from PIL import Image, ImageDraw, ImageFont


def encoded(im):
    out = io.BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()


def corpus(font_path):
    yield "screenshot", Path("tests/fixtures/document.png").read_bytes(), ""
    labels = ["A", "B", "C", "X", "Y", "Z", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
              "RF", "RE", "OK", "ID", "Yes", "No", "SIM", "NAO", "CEP", "ZIP", "ABC", "123", "USD", "EUR", "R$", "42"]
    im = Image.new("RGB", (720, 820), "white")
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(font_path, 32)
    for i, label in enumerate(labels):
        draw.text((30 + 180 * (i % 4), 20 + 100 * (i // 4)), label, fill="black", font=font)
    yield "labels", encoded(im), " ".join(labels)
    cases = [
        ("short", (640, 180), 32, ["Yes   No   OK   ID   42", "A   B   C   1   2   3"]),
        ("receipt", (720, 900), 28, ["MERCEARIA CENTRAL", "Rua das Flores 123", "Data 06/09/2026", "Cafe 12.50", "Leite 8.90", "Pao 6.00", "TOTAL 27.40", "Obrigado pela visita"]),
        ("sparse", (1000, 700), 38, ["Invoice 12345", "Total USD 129.95"]),
        ("paragraph", (1200, 900), 24, [f"Line {i:02d}: The quick brown fox jumps over the lazy dog." for i in range(15)]),
        ("small-text", (1200, 700), 14, [f"Item {i:02d}   Configuration enabled   version 1.0.1" for i in range(18)]),
        ("long-line", (1800, 260), 24, ["Reference ABC-123: " + "the quick brown fox " * 6]),
        ("scan", (2400, 3200), 48, [f"Document line {i:02d}   Project report September 2026" for i in range(26)]),
    ]
    for name, size, height, lines in cases:
        im = Image.new("RGB", size, "white")
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype(font_path, height)
        for i, line in enumerate(lines):
            draw.text((24, 20 + i * int(height * 1.8)), line, font=font, fill="black")
        yield name, encoded(im), " ".join(lines)
        if name == "receipt":
            yield "rotated", encoded(im.rotate(7, expand=True, fillcolor="white")), " ".join(lines)
    yield "blank", encoded(Image.new("RGB", (600, 400), "white")), ""


def token_recall(expected, recognized):
    tokens = lambda t: collections.Counter(re.findall(r"\w+", t.casefold()))
    expected = tokens(expected)
    return sum((expected & tokens(recognized)).values()) / sum(expected.values()) if expected else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--font", default="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    p.add_argument("--rec-min-width", type=int)
    p.add_argument("--det-min-side", type=int)
    p.add_argument("--repeat", type=int, default=3)
    args = p.parse_args()
    data = list(corpus(args.font))
    for model in ("tiny", "small"):
        kw = {k: v for k, v in vars(args).items() if k in ("rec_min_width", "det_min_side") and v is not None}
        t = time.perf_counter()
        engine = faster_paddle.OcrEngine(model_size=model, **kw)
        init_ms = (time.perf_counter() - t) * 1000
        for name, image, expected in data:
            engine.ocr(image)
            times = []
            for _ in range(args.repeat):
                t = time.perf_counter()
                result = engine.ocr(image)
                times.append((time.perf_counter() - t) * 1000)
            print(json.dumps(dict(model=model, case=name, init_ms=init_ms,
                package_version=faster_paddle.__version__,
                runtime_build=getattr(faster_paddle, "__runtime_build__", None),
                config=getattr(engine, "config", kw), median_ms=statistics.median(times), times_ms=times,
                token_recall=token_recall(expected, result["text"]), result=result)), flush=True)
        del engine


if __name__ == "__main__":
    main()
