CPU latency review of `772173c`, targeting `tiny` and `small`.

This is the historical baseline audit. See [the implemented changes](PERFORMANCE_CHANGES.md)
for the current defaults, fixes, validation, and remaining experimental tradeoffs.

**The first priority is recognition width/batch policy and CPU scheduling.** The
existing implementation already does several useful things: native Rust image
processing, borrowed ONNX outputs, graph optimization level 3, shared prepacked
recognition weights, width sorting, parallel crops, and an integer rectangle
crop fast path. Replacing these wholesale is unlikely to be the shortest path
to lower latency.

No production code or model weights were changed for this review. The benchmark
harnesses and raw measurements are under [benchmarks](benchmarks/).

**Measured end-to-end results**

Host: Ryzen 9 9950X3D, 16 physical / 32 logical CPUs, Linux under a Microsoft
hypervisor. Release build with the checked-in Cargo.lock: ort/ort-sys rc.12,
downloaded ONNX Runtime 1.24.2. Defaults on this host mean 32 detector threads,
8 recognizer sessions with 2 threads each, and the default Rayon pool.

The candidate uses `OCR_DET_THREADS=8 REC_POOL=16 REC_BUDGET=1`, leaving the
constructor's `threads` argument unset. Budget 1 forces singleton batches in
the current planner; it does not shrink the recognizer's input width to 1.

| Input | Model | Default median | Candidate median | Latency reduction |
|---|---|---:|---:|---:|
| Bundled 804×505 screenshot | tiny | 186.4 ms | 110.4 ms | 40.8% |
| Bundled 804×505 screenshot | small | 497.0 ms | 332.8 ms | 33.0% |
| Synthetic 2400×3200 resize | tiny | 232.7 ms | 191.3 ms | 17.8% |
| Synthetic 2400×3200 resize | small | 456.8 ms | 365.2 ms | 20.1% |

These confirmation medians combine ten timed calls per setting: two fresh
processes, each with one warmup and five measurements, alternating default and
candidate. Engine initialization is excluded. OCR calls include decoding and
Python result construction. Debug logging is off. Output signatures match
between default and candidate for each input/model; signatures include ordered
box coordinates and text, **not confidence values**.

This is a local tuning result, not a universal default or an accuracy benchmark.
There is only one independent image. The enlarged input stretches the screenshot
and yields 61 boxes, versus 140 for tiny and 97 for small on the original; it is
not representative of a dense scanned page. CPU affinity and clocks were not
fixed, and these sample counts do not establish p95/p99 behavior.

The broader sweep also found that simply increasing the recognition pool from
8 to 16 made the screenshot slower: tiny 186→194 ms and small 474→558 ms in
that sweep. Increasing the pixel budget to 1600 was slower still. More threads
and bigger batches are not reliable latency improvements here.

Construction also matters: the singleton candidate's 16 sessions took roughly
175 ms for tiny and 530–560 ms for small in the followup, versus roughly
108 ms and 298 ms for the original sweep defaults. Reuse engines; the warm
winner need not win for one-shot cold requests.

**Pipeline and where time goes**

`src/lib.rs` decodes image bytes, optionally preprocesses, locks the engine,
runs OCR, reconstructs two text layouts, maps coordinates, and creates Python
dictionaries. `src/ocr.rs` resizes/normalizes for detection, runs the detector,
decodes DB components into quads, extracts all full-resolution crops, plans
recognition batches, runs them across a session pool, and greedily decodes CTC.
`src/cv.rs` supplies hulls, rectangles, scores, and warps. `src/preprocess.rs`
contains optional resize/NLM/deskew/Sauvola. `src/layout.rs` builds both texts.

A separate instrumented baseline on the screenshot showed these approximate
warm stage times. They are diagnostic samples with logging, not the medians
above:

| Stage | tiny | small |
|---|---:|---:|
| Detector ORT inference | 78–80 ms | 116–118 ms |
| DB postprocessing and box sort | 7–18 ms | 10–11 ms |
| Crop extraction | 2–4 ms | 2–3 ms |
| Recognition stage | 85–97 ms | 345–380 ms |
| Image decode | ~1 ms | ~1 ms |
| Layout | <1 ms at logged precision | <1 ms at logged precision |

Optimize inference and the work fed into it first. Layout/string cleanups will
not produce comparable gains on this workload.

**1. Fix the recognition width contract, then reduce padding**

Locations: `src/ocr.rs:202–214`, `:278–294`, and `plan_rec_batches`.

The planner uses `ceil(48*w/h)` clamped to 3200, but the runner imposes a minimum
width of 320 and otherwise truncates its computed width. Thus the planner and
runner disagree both about the minimum and fractional rounding. `rec_batch`
is advertised as a cap but is multiplied by 16 internally; its default 4
actually allows up to 64 crops.

The baseline log contains a small-model batch with **16 crops at width 320**:
5120 width-units of work against a configured budget of 800. That batch took
about 300 ms in one logged run and helped determine the whole recognition
stage's completion time.

Implementation order:

1. Introduce one effective-width calculation used by sorting, budgeting,
   tensor allocation, and resizing. Initially retain the 320 floor to isolate
   scheduling effects. Make the public batch cap honest.
2. Benchmark singleton batches against small, correctly costed batches.
3. Remove or lower the 320 floor using validated width buckets, initially
   testing widths such as 64/128/192/256/320 and wider buckets. Preserve the
   current height, normalization, and crop geometry.
4. Reuse tensor buffers within each worker, accounting for padding that must
   be cleared when shapes change. Test memory patterns with recurring shapes.

Both bundled recognizers actually accept width 64, 128, and 192. Direct CPU
probes using ONNX Runtime 1.24.2, batch 1, one thread, random normalized input,
three warmups and ten samples produced:

| Width | tiny recognition only | small recognition only |
|---|---:|---:|
| 64 | 0.46 ms | 2.20 ms |
| 128 | 0.79 ms | 3.97 ms |
| 192 | 1.15 ms | 5.81 ms |
| 320 | 1.90 ms | 9.60 ms |
| 640 | 3.87 ms | 18.93 ms |

Width 64 is about 4× faster than 320 in this isolated probe. This is evidence
that the floor wastes work for short crops, **not a forecast of 4× faster OCR**.
Only short crops benefit, and detection is unchanged.

Changing padding can change text: tiny has width-dependent reductions, and
small additionally has attention. One high-budget tiny sweep configuration
already changed the text/box signature despite using the same weights and
detections. Compare text and confidence on real short words, punctuation,
non-Latin strings, and long lines before making width changes the default.

**2. Replace static recognition assignment and tune CPU budgets**

Locations: `src/lib.rs:235–259`, `src/ocr.rs:76–104`, `:221–235`.

Batches are sorted by width and assigned round-robin to sessions. Workers do
not take remaining work from a slower worker. Since actual padded cost is
currently wrong, a supposedly cheap narrow batch can dominate one worker.
Use a shared job index/queue with one mutable session per worker, and consider
dispatching largest estimated jobs first. Scatter results back by crop index
to preserve output order. Benchmark queue overhead for sparse images.

Provide independent detector-thread, recognition-worker, and per-recognizer
thread controls. The present physical-core count reads all of `/proc/cpuinfo`
and does not intersect CPU affinity or container quotas; off Linux it falls
back to logical parallelism. The default pool cap of 8 also means hosts with
more than 8 physical cores no longer get single-threaded recognizers.

Detector spinning is explicitly disabled, but recognizer spinning is always
enabled. With multithreaded recognizers, idle sessions can compete with active
ones or subsequent pipeline phases. Measure bounded/off spinning and shared
CPU budgets, including concurrent requests. ORT documents explicit thread
counts, affinity behavior, and contention between session pools in its
[threading guidance](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).

The engine mutex serializes inference on one engine and currently remains
held during layout assembly. Release it immediately after `eng.run` returns.
That is a small concurrency improvement, not a single-image inference gain.
Creating many independent engines instead requires a process-wide CPU budget.

**3. Make detector resolution an explicit speed/recall choice**

Location: `det_resize_dims`, `src/ocr.rs:401` onward.

Despite the README's “never upscales” statement, a short side below 736 is
upscaled before the long-side cap. The 804×505 fixture becomes 1184×736:
about **2.15× as many detector pixels** as the original. A configurable
minimum side or no-upscale mode is a material opportunity for UI screenshots.

For large documents, test maximum sides 960, 1280, and 1600. Moving 1600→1280
nominally removes 36% of detector pixels at the same aspect ratio, before
rounding; the overall speedup depends on detector share and recall. A missed
box cannot be recovered by a high-resolution recognition crop. Do not assume
the README's negligible-loss assertion holds across datasets.

The optional `resize=True` works on the full OCR source. If both source sizes
already hit the same detector cap, it may add a resize without appreciably
reducing detector work, while degrading recognition detail. Keep optional
denoise/deskew/binarization off for the fastest baseline. For applications with
known text regions, a detect-free recognition API or ROI-only OCR could avoid
the detector altogether; those are application-specific API additions.

**4. Optimize the actual model kernels: INT8 and runtime versions**

The graphs contain convolutions, GELU patterns, and output projections;
small also has attention. Single-thread width-320 ORT profiles put convolutions
at roughly half of tiny's recorded node time and roughly two thirds of small's.
GELU and layout reorders also matter. This argues for profiling convolution
quantization and fusion, rather than optimizing only CTC.

Test static calibrated INT8 QDQ for detector and convolution-heavy recognizer
regions, then selectively quantize MatMul/head operations. Start with small,
where absolute inference cost is higher. Quantize detector and recognizer
separately so accuracy regressions can be attributed. Calibration needs actual
normalized document tensors and recognition crops at deployed widths. Measure
quantize/dequantize overhead and unsupported/fallback operations; smaller model
files do not prove faster execution. ORT recommends static quantization for
CNNs and S8S8 QDQ as a CPU starting point, with hardware-dependent caveats in
its [quantization documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

Also benchmark a runtime upgrade. The same synthetic width-320 probe with
Python ORT 1.29.0 measured 1.58/8.81 ms versus 1.90/9.60 ms with 1.24.2 for
tiny/small. These separate short probes are a reason to run an end-to-end
runtime comparison, not proof of those gains in the Rust extension. Cargo.toml
mentions rc.10 but Cargo.lock resolves rc.12; record the actual binary version.

Do not start by manually deleting exported Identity nodes: level-3 optimization
is already enabled, and the profiles show optimized GELU, NCHWc, and fused
MatMul operations. Inspect the optimized graph for remaining missed fusions.
Offline optimized models can reduce startup work, but must match execution
provider and hardware constraints described by
[ORT graph optimization guidance](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html).

**5. Specialize recognition outputs where the application permits it**

Tiny's final weight matrix is 80×6906; small's is 120×18710. At width 320,
batch 1 emits 40 timesteps: approximately 1.10 MB and 2.99 MB of FP32
probabilities respectively. Outputs are already borrowed, but they still must
be computed, written, and scanned.

Options, in increasing semantic impact:

- Fuse argmax and winning-probability extraction into an optimized output
  kernel, returning only indices and scores. Merely adding standard ONNX
  ArgMax/ReduceMax can still materialize the large intermediate and add scans;
  benchmark it against the existing Rust decoder.
- Offer a text-only mode that removes the final softmax and takes argmax over
  logits. This preserves the mathematical winning class for finite logits,
  subject to numerical/tie validation, but does not preserve probability-based
  confidence. Exact confidence still needs the softmax denominator. Softmax
  was only a small share of the profile, so this alone has a limited ceiling.
- For a known language/domain, physically slice final projection columns and
  bias, remap CTC indices, and retain blank/space/required characters. Changing
  only `char_dict.json`, or filtering after inference, saves no head compute.
  This restricts supported text and changes probability normalization.
- Consider independently selectable detector/recognizer sizes, or tiny-first
  recognition with small fallback for difficult crops. Cascades can improve
  average latency but worsen difficult-case latency; confidence requires
  calibration and the extra engine consumes memory.

The large output head is a useful target, especially for small, but it is not
the majority of the measured network cost. Do not promise proportional
end-to-end gains from reducing the alphabet.

**6. Improve Rust image/geometry work after the larger wins**

Locations: `src/ocr.rs` detection normalization, `db_postprocess`, `crop_quad`,
`rec_batch_run`; `src/cv.rs` hulls, rectangle scoring, and `warp_crop`.

- Fuse resize, channel conversion, and normalization into tensor writes.
  Detector normalization currently makes only three Rayon tasks, one per
  channel. Use row/tile work and reusable scratch buffers. Preserving the
  intermediate u8 rounding matters for output equivalence.
- Sample crops directly into recognition-sized tensors. Currently the full
  crop is copied/warped, possibly rotated, then resized and normalized. A
  direct warp avoids large intermediate crops, but one resampling pass differs
  from the existing two-pass interpolation and requires quality validation.
- Replace pixel-by-pixel DFS components with a measured scanline/run-length
  implementation and collect component row extrema or boundary points for the
  hull. Current code stores every foreground pixel as two f64s, then copies
  and sorts all points. Preserve 8-connectivity, candidate limits, ordering,
  and box scoring semantics.
- `min_area_rect` is labeled rotating calipers, but projects every hull point
  for every hull edge: O(h²). True calipers can be O(h); profile actual hull
  sizes before investing. Avoid per-row heap allocation of four polygon
  intersections in `box_score_fast`.
- Use native CPU compilation/SIMD experiments for Rust hot loops. This does
  not recompile the prebuilt ONNX Runtime kernels. Keep portable wheels
  separate from machine-specific builds.

Crop extraction was only a few milliseconds in the fixture. Geometry becomes
more valuable on noisy maps and high-resolution/dense pages, which the current
fixture does not cover.

**Validation and implementation sequence**

1. Retain this baseline and add independent real scans, receipts, screenshots,
   sparse images, long lines, rotated text, and the required languages. Record
   stage times, actual batch widths, padding ratio, box recall, CER/WER, peak
   RSS, initialization, and warm latency distributions. Separate latency from
   multi-request throughput.
2. Fix effective-width budgeting and batch cap; add dynamic work assignment;
   compare singleton and small batches with controlled CPU budgets.
3. Validate narrower recognition inputs and explicit detector resolution modes.
4. Benchmark runtime upgrades and calibrated INT8 against that leaner pipeline.
5. Pursue domain-specific heads/detect-free APIs and measured Rust hotspots.

`cargo build --release` and `cargo test --release` succeeded; all 17 Rust tests
passed. The full Python integration suite was not run. Existing tests mostly
exercise tiny on one fixture; the speed guard is an absolute six-second limit,
and the CI workflow builds/publishes without a test job. Those checks cannot
validate the proposed changes or detect modest latency regressions.

**Reproducing this audit**

With a release-built extension importable and Pillow installed, run from the
repository root:

```bash
python benchmarks/cpu_latency.py > sweep.jsonl
OCR_DET_THREADS=8 REC_POOL=16 REC_BUDGET=1 \
  python benchmarks/cpu_latency.py tiny singleton 0
OCR_DET_THREADS=8 REC_POOL=16 REC_BUDGET=1 \
  python benchmarks/cpu_latency.py small singleton 1
python benchmarks/graph_probe.py > graph-probe.jsonl
```

The worker's last argument selects the original fixture (0) or enlarged one
(1). Worker mode honors environment settings; sweep mode clears tuning
variables before each configuration. Graph probes additionally require numpy
and onnxruntime; pin the runtime version when comparing results. Original
JSONL samples, output signatures, and diagnostic logs are saved as review
artifacts under `benchmarks/results/` (not performance assertions).
