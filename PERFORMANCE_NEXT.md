CPU optimization follow-up to v1.0.2 (`9dda394`), 2026-09-06.

The strongest measured next option is a newer ONNX Runtime, followed by
convolution-focused quantization and further operator/layout fusion. The current
pipeline is **not proven DRAM-bandwidth-bound**. Optimize time per image; reaching
a hardware limit is useful evidence, but increasing memory traffic to saturate
RAM would work against that objective.

## New measurements

Ryzen 9 9950X3D under a Microsoft hypervisor. All inference was confined to CPUs
0 and 2 at nice priority 10, with **one inference thread**, sequential calls,
and CPUExecutionProvider only. No GPU settings or drivers were touched.
Inputs were deterministic random FP32 tensors, batch one. These are graph
microbenchmarks, not full OCR or accuracy benchmarks.

The runtime comparison used two fresh processes per version, one warmup and
seven timed calls per shape per process (14 warm samples total). Process order
was new, old, old, new. Runtime profiling was collected separately from timed
calls. Model weights and graph inputs were identical; approximate GELU was off.

| Graph / shape | ORT 1.24.2 median | ORT 1.29.0 median | Reduction |
|---|---:|---:|---:|
| tiny detector, 736×1184 | 59.66 ms | 49.78 ms | 16.6% |
| small detector, 736×1184 | 147.31 ms | 127.84 ms | 13.2% |
| tiny recognizer, width 64 | 0.441 ms | 0.409 ms | 7.3% |
| tiny recognizer, width 320 | 1.914 ms | 1.626 ms | 15.0% |
| small recognizer, width 96 | 2.934 ms | 2.774 ms | 5.5% |
| small recognizer, width 320 | 9.131 ms | 8.624 ms | 5.6% |

On these six inputs, detector and tiny-recognizer arrays matched exactly.
Small-recognizer timestep argmax IDs matched; maximum probability differences
were 1.37e-6 at width 96 and 2.98e-6 at width 320. Random inputs are a weak check
of OCR quality: these results do not replace testing real documents, confidence
thresholds, multilingual text, and long lines.

An earlier diagnostic pass on 1.24.2 showed:

| Graph | Convolution kernel share | GELU share | Layout reorder share |
|---|---:|---:|---:|
| tiny detector | 41.8% | 12.6% | 23.5% |
| small detector | 49.2% | 10.3% | 25.3% |
| tiny recognizer, width 320 | 54.8% | 22.6% | 8.2% |
| small recognizer, width 320 | 66.6% | 12.5% | 5.3% |

Shares are instrumented kernel durations from one warm run, not end-to-end
stage shares or hardware-stall classifications. Convolution time itself may
include cache/memory stalls. The first 1.29 profile reduced detector reorder
share to approximately 13.5% / 16.7%, and GELU duration also fell. This aligns
with upstream improvements to the [NCHWc layout transformer](https://github.com/microsoft/onnxruntime/pull/27691)
and [activation handling](https://github.com/microsoft/onnxruntime/pull/27821).

## Priority order

1. **Upgrade the inference runtime and validate the complete Rust pipeline.**
   This is the best measured candidate that retains FP32 weights. The shipped
   Rust dependency currently embeds ORT 1.24.2; installing a newer Python ORT
   does not upgrade it. Integrate compatible native binaries/bindings, run real
   OCR comparisons, and build every supported platform before changing the
   package default. Recompiling Rust with `target-cpu=native` does not rebuild
   or retune the separately supplied ONNX Runtime kernels.

2. **Evaluate calibrated INT8 convolution plus MatMul.** The previous dynamic
   MatMul-only probe misses the dominant convolution backbone. W8A8 can reduce
   quantized weight/activation storage and use integer dot-product kernels; it
   does not guarantee a fourfold whole-pipeline speedup. The host exposes
   AVX512-VNNI. Start with per-channel S8S8 QDQ, then measure actual optimized
   kernels, conversion costs, detection recall, character error rate, and
   confidence behavior on representative calibration/holdout images. Keep FP32
   fallback on unsupported or slower hardware. ORT recommends static
   quantization for CNNs and S8S8 QDQ as the initial CPU configuration in its
   [quantization guidance](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).
   Merely storing FP16/BF16 weights is not enough: the selected CPU provider
   needs efficient kernels for the entire operator chain.

3. **Keep convolution, activation, and residual operations in the same layout.**
   The measured reorder/GELU cost makes this a better target than Python or text
   formatting. First inspect what the newer runtime still leaves unfused.
   Further gains may require custom kernels or a CPU compiler/backend that
   fuses these operations and tiles intermediates into cache. Benchmark a whole
   subgraph: isolated fast kernels can lose their advantage to conversion and
   dispatch overhead. ORT already performs level-3 layout optimization, so
   another generic “enable optimizations” flag is insufficient. Hardware-specific
   optimized graphs must match the target environment, as described in the
   [graph optimization documentation](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html).

4. **Tune recognition concurrency for cache locality after the kernel changes.**
   The present shared prepacked weights and bounded worker pools are a good
   baseline. More active workers also mean more live activations and independent
   session arenas. A smaller pool or width-compatible batches could improve
   reuse; it needs representative page measurements on the actual host. Batch
   size increases arithmetic intensity by reusing weights, which can move a
   workload toward a compute limit while still making it faster. This VM's
   reported cache topology is not sufficient evidence for automatic CCD pinning.

5. **Consider a fused recognition projection/softmax/top-1 kernel only if its
   residual share warrants it.** At width 320 the current output is 1,104,960
   bytes for tiny or 2,993,600 bytes for small; IDs plus probabilities need only
   480 bytes. A fused kernel could avoid writing and rescanning the full
   probability tensor. All classes still participate in confidence normalization;
   removing softmax or pruning the alphabet changes the API semantics. This is
   a smaller target than the convolution backbone, especially for small.

## Options tested without a consistent win

The first probe compared dynamic shapes, exact fixed input shapes, compact
outputs, and both together. Five timed calls per variant alternated forward and
reverse execution order. Each variant had its own warmed session, so its
absolute times should not be compared to the single-session runtime table.

| Recognizer | Dynamic | Fixed shape | Compact output | Fixed + compact |
|---|---:|---:|---:|---:|
| tiny, width 64 | 0.557 ms | 0.613 ms | 0.635 ms | 0.570 ms |
| tiny, width 320 | 2.402 ms | 2.172 ms | 2.311 ms | 2.247 ms |
| small, width 96 | 3.275 ms | 3.390 ms | 3.559 ms | 3.410 ms |
| small, width 320 | 10.150 ms | 10.774 ms | 10.581 ms | 11.648 ms |

Compact output appends ONNX ArgMax and ReduceMax after the existing Softmax.
It preserves the tested IDs/probabilities exactly but **still materializes the
full internal probability tensor** and adds reduction kernels. It is not the
fused kernel proposed above. The production Rust path already borrows ORT
outputs without a full tensor copy. These results do not support shipping
compact output as a generic speed improvement.

Fixed shapes did not consistently improve recognizer timing. The small detector
improved in the initial five-sample test (186.3→171.5 ms), while tiny did not
(70.9→70.8 ms). This is a candidate for a recurring, fixed-resolution workload,
not justification for a session cache spanning arbitrary image sizes. That
cache could multiply memory consumption and startup costs. Recognition width
bucketing also introduces padding that can erase specialization gains.

## What would establish a bandwidth limit

Use a roofline measurement at the relevant memory level: cache and DRAM have
different attainable bandwidths. Compute and logical tensor sizes alone cannot
classify the bottleneck. The recognizer weights are approximately 4.21 MiB and
20.10 MiB; reuse from cache is plausible, although activations, packed weights,
other workers, and actual cache topology determine residency.

The probe reports resolved Conv/MatMul FLOPs and original-graph edge bytes.
Those bytes include identities and do not account for optimizer fusion, tiling,
packing, cache hits, or physical loads/stores. Some small-model shapes remain
unresolved and are explicitly counted. **Do not divide these logical bytes by
latency and label the result DRAM bandwidth.**

To demonstrate the target, collect actual memory-controller traffic and compare
with sustainable bandwidth measured on the same host/CPU allocation, alongside
cache misses, IPC, vector utilization, and thread-scaling curves. Cache-miss
counts alone are not a reliable byte counter. This guest has no exposed uncore
memory-controller event source and no installed `perf` executable; no direct
DRAM-bandwidth or stall measurement was obtained. No PMU permissions, drivers,
or host settings were changed. Virtualized counter availability depends on the
hypervisor, as documented by [AMD](https://docs.amd.com/r/en-US/57368-uProf-user-guide/Virtualization-Support).

A suitable success criterion is reduced p50/p95 OCR latency at unchanged agreed
accuracy, plus evidence that the remaining dominant kernels approach their
measured compute/cache/memory limit. Full-machine bandwidth saturation was not
attempted after the earlier system disruption.

## Reproduce and inspect

Added [memory_probe.py](benchmarks/memory_probe.py); it modifies candidate graphs
only in memory. Install `numpy`, `onnx`, and the chosen `onnxruntime` version in
an isolated environment, then run from the repository root:

```bash
# Choose CPUs available to your process; these were available on this host.
taskset -c 0,2 nice -n 10 python benchmarks/memory_probe.py --detector
# Matched runtime comparison, with one candidate session at a time:
taskset -c 0,2 nice -n 10 python benchmarks/memory_probe.py \
    --detector --variants dynamic --repeat 7 --output-dir target/probe-arrays
```

Raw timed samples, graph estimates, profiles, and numerical comparisons are in:

- [Variant probe, ORT 1.24.2](benchmarks/results/memory-probe-ort-1.24.2.jsonl)
- [Runtime baseline, ORT 1.24.2](benchmarks/results/memory-runtime-1.24.2.jsonl)
- [Runtime candidate, ORT 1.29.0](benchmarks/results/memory-runtime-1.29.0.jsonl)
- [Cross-runtime numerical comparison](benchmarks/results/memory-runtime-numerics.jsonl)

No production defaults, dependencies, model files, or release tags were changed
in this follow-up. The probe executed successfully for both bundled models,
both stages, all listed variants, and both runtime versions.
