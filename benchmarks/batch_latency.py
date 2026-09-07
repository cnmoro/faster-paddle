"""Compare sequential OCR with bounded multi-image processing on one CPU engine."""
import argparse
import json
import statistics
import time

import faster_paddle
from validate_cpu import corpus


def signature(r):
    return [(v["topLeftCoord"], v["bottomRightCoord"], v["text"]) for v in r["bounds"].values()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--font", default="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--images", type=int, default=4)
    args = p.parse_args()
    selected = [(name, image) for name, image, _ in corpus(args.font)
                if name in ("screenshot", "sparse", "long-line")]
    for size in ("tiny", "small"):
        e = faster_paddle.OcrEngine(model_size=size)
        for case, image in selected:
            inputs = [image] * args.images
            e.ocr_batch(inputs)
            sequential, batched = [], []
            equal = True
            for i in range(args.repeat):
                # Alternate order to reduce systematic cache/thermal bias.
                results = {}
                for mode in ("single", "batch") if i % 2 == 0 else ("batch", "single"):
                    start = time.perf_counter()
                    out = [e.ocr(x) for x in inputs] if mode == "single" else e.ocr_batch(inputs)
                    (sequential if mode == "single" else batched).append((time.perf_counter() - start) * 1000)
                    results[mode] = list(map(signature, out))
                equal &= results["single"] == results["batch"]
            print(json.dumps(dict(model=size, case=case, images=args.images,
                config=e.config, sequential_ms=sequential, batch_ms=batched,
                sequential_median_ms=statistics.median(sequential), batch_median_ms=statistics.median(batched),
                speedup=statistics.median(sequential)/statistics.median(batched), text_boxes_match=equal)), flush=True)
        del e


if __name__ == "__main__":
    main()
