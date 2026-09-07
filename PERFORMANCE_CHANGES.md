The CPU latency fixes from [the original review](PERFORMANCE_REVIEW.md) are
implemented. The default remains CPU-only FP32, with the original model weights
and character dictionaries. No GPU execution provider or driver configuration
was introduced.

For subsequent runtime, kernel, and memory-bandwidth investigation, see
[the optimization follow-up](PERFORMANCE_NEXT.md).

**Resolved defaults**

| Setting | Automatic behavior |
|---|---|
| Total CPU budget | Available physical cores, bounded by process affinity and OS/container CPU quota |
| Detector threads | Up to 8 within that budget |
| Normal recognition pool | One worker per core up to 32; medium capped at 8; intra-op threads split within budget |
| Sparse wide recognition | Up to four sessions with up to four threads each, used instead of the normal pool for a few long lines |
| Image processing | Engine-local Rayon pool within the same CPU budget |
| Recognition batch cap | 1 crop; a supplied `rec_batch` is now an actual maximum |
| Recognition minimum width | tiny: 64; small: 96; medium: 320 |
| Detector size limits | Minimum short side 736, maximum long side 1600, rounded to multiples of 32 |
| ONNX spinning | Enabled during inference, forcibly stopped when each inference call returns |
| Approximate GELU | Disabled |

`engine.config` exposes the resolved settings. `threads` overrides detection of
the total budget. Environment overrides and the two new keyword arguments
`det_min_side` and `rec_min_width` are described in [README.md](README.md).
These defaults combine hardware discovery with measured workload heuristics;
they do not run a hidden calibration benchmark on the first request. Different
CPU architectures and document distributions can have different optima.

**Implementation**

- Recognition planning and tensor allocation use the same effective width,
  including padding. The old sixteen-fold multiplier on `rec_batch` is gone.
- Workers dynamically claim jobs, starting with the largest estimated jobs.
  A separate bounded pool handles sparse long-line workloads; both pools never
  run concurrently, keeping active inference within the CPU budget.
- Detection and recognition tensors are reused. Fused resize/channel-conversion/
  normalization preserves the reference u8 resize rounding and clears padding
  on every reuse. Outputs remain borrowed from ONNX Runtime.
- Connected components use 8-connected scanline runs, retaining run endpoints
  rather than every foreground pixel. Their convex hulls are equivalent to the
  reference pixel flood fill. Larger hulls use rotating calipers; small hulls
  retain the cheaper exhaustive loop. Polygon scoring uses stack storage.
- Recognition failures propagate as errors instead of panicking inside Rayon.
  Layout runs after releasing the inference mutex. Shared-engine initialization
  is serialized without holding the GIL, avoiding redundant concurrent builds.
- Model sessions are initialized in parallel groups of at most four, sharing
  prepacked weights. This reduces startup cost while bounding concurrent graph
  optimization. PNG preparation consumes its RGB buffer instead of cloning it;
  base64 decoding runs without holding the GIL.
- CPU detection intersects Linux topology with allowed CPUs, including the last
  `/proc/cpuinfo` block, and respects available parallelism/quota. Other platforms
  use native physical-core discovery with the OS budget as an upper bound.
- Optional minimum detector size and recognition padding are explicit. A
  non-multiple-of-32 detector cap cannot be exceeded by rounding.
- Wheels no longer include a second copy of ONNX model files already embedded
  in the extension. Source distributions explicitly include the build script
  and compilation assets, excluding incidental test caches.
- The resolved ort rc.12 version is pinned; the unused direct ndarray version
  was removed. CI now runs Rust and Python tests before publishing.

**Multiple images**

```python
from faster_paddle import OcrEngine

engine = OcrEngine(model_size="small")
results = engine.ocr_batch(images, batch_size=4)
```

The return value is a list in input order. Each result has the same shape as
`ocr`. Input byte buffers are borrowed when possible. Each bounded window
decodes/preprocesses pages in parallel, retains their individual detector
dimensions and transforms, then shares recognition work across pages. No
padding of differently shaped detector images is needed. Both the engine and
module-level `ocr_batch` accept the existing preprocessing flags.

`batch_size` limits resident decoded pages; `rec_batch` limits crops per neural
input. They are independent. Batch mode is useful for throughput on sparse
pages, but it waits for the entire list and is not guaranteed to beat sequential
calls on dense pages. `ocr` remains the choice for the earliest individual-page
response. Invalid input reports its zero-based image index without poisoning
the engine; an empty list returns an empty list.

**Measurements collected before the execution environment changed**

Host: Ryzen 9 9950X3D, 16 physical / 32 logical CPUs, Linux under a Microsoft
hypervisor, release build, ONNX Runtime 1.24.2 CPU. The screenshot is the bundled
804×505 fixture. The retained original baseline contains ten warm samples per
model; the revised spinning comparison also contained ten per setting.

| Screenshot | Original audit median | Revised settings median |
|---|---:|---:|
| tiny | 186.4 ms | 68.9 ms |
| small | 497.0 ms | 263.6 ms |

These are separate local probes, not a paired universal speedup claim. The
revised probe used the settings now selected by default: 8 detector threads,
16 single-thread recognition workers, model-specific padding, and spinning
stopped on return. No additional full-CPU runs were made after the user's
report of GPU/system disruption. Validation afterward was restricted to two
CPU cores and low scheduling priority.

Earlier completed multi-image probes, with spinning disabled, showed four
long-line small-model pages taking approximately 304 ms sequentially versus
253 ms through `ocr_batch` (about 20% greater throughput). The text and box
signatures matched. Sparse-page and dense-page gains varied; these numbers
should not be treated as a guarantee for the final spinning defaults.

Synthetic validation covered short labels, receipts, rotated text, sparse
pages, paragraphs, small text, long lines, large scans, and blank input. The
selected padding defaults retained the expected tokens on the labeled cases.
This is a small Latin-script synthetic corpus, not a multilingual accuracy
certification. Padding changes can affect text/confidence, notably UI glyphs;
set `rec_min_width=320` for the reference padding behavior.

The initial audit's raw JSONL remains under `benchmarks/results/`. Temporary
implementation probes were no longer available after the execution environment
changed; the table above summarizes their recorded outputs. Reproducible
harnesses are provided rather than reconstructing missing raw samples:

```bash
python benchmarks/validate_cpu.py --repeat 5 > quality-and-latency.jsonl
python benchmarks/batch_latency.py --repeat 5 > batch-latency.jsonl
python benchmarks/cpu_latency.py > thread-sweep.jsonl
python benchmarks/graph_probe.py > graph-probe.jsonl
```

The synthetic generators require Pillow and default to a Linux DejaVu font;
use `--font` on other systems. Graph probes additionally require numpy and
onnxruntime. Use the same machine, release build, runtime, and representative
inputs when comparing variants. The integration suite's latency smoke test is
only a gross check, not a portable performance regression benchmark.

**Experiments deliberately not promoted to defaults**

- Disabling detector upscaling reduced tiny's small-text token recall in a
  labeled case and changed screenshot detections substantially. It is exposed
  as `det_min_side=0`, rather than silently trading recall for speed.
- Width 64 changed a real small-model `RF` label to `RE`; width 96 restored it.
  Medium retains width 320 because it did not receive equivalent validation.
- Approximate GELU gave inconsistent end-to-end results and changed a tiny
  detection. `OCR_APPROX_GELU=1` is available for controlled experiments.
- A MatMul-only dynamic INT8 probe gave modest, shape-dependent gains; it did
  not address the convolution-dominated backbone. Shipping static INT8 needs
  representative calibration and accuracy validation. No quantized weights
  are substituted automatically.
- A newer Python ONNX Runtime showed promising isolated timings in the review,
  but that is not a validated cross-platform Rust wheel/runtime migration.
- Alphabet pruning and dropping softmax change language support or confidence
  semantics. Direct single-pass crop warping changes the current two-stage
  interpolation. They are not silently enabled as generic optimizations.

**Verification**

Rust tests cover geometry against reference implementations, fused resizing,
padding reuse, batch caps, CPU affinity parsing, and CTC. Python tests cover
both tiny and small, short labels, sparse wide jobs, ordered multi-image output,
per-image coordinate transforms, mixed shapes, reused buffers, concurrent
calls, explicit CPU budgets, affinity, invalid input and recovery.

Final local validation passed 23 release-mode Rust tests and all 15 Python
integration tests against the installed release wheel. Builds and inference
tests were restricted to two CPU cores at low scheduling priority. The wheel
and source distribution built successfully; archive checks confirmed no
duplicate model data in the wheel and the required build script, lockfile,
sources, and tiny/small model assets in the source distribution. Medium assets
and incidental test caches were excluded. Cross-platform wheel builds remain
for CI to validate.
