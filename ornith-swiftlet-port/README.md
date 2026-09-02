# Ornith 1.5 35B on Swiftlet: port kit

This archive is a concrete, reviewable starting point for running
`ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` with Swiftlet under a hard memory
budget.

It is deliberately **not** a copied Swiftlet repository. Current Swiftlet
already implements the Qwen3.5 MoE text graph used by Ornith: nested
`text_config`, split DeltaNet projections, `language_model.` weight prefixes,
normalized top-k routing, fixed-stride qpack expert blobs, and the Metal
runtime. The useful port is therefore a small integration overlay plus the
SlotStream-style memory planner, not a second model implementation.

## What is in the archive

### 1. A standalone Swift package

`Sources/OrnithSwiftletPort` contains Metal-independent reference logic:

- exact Ornith 1.5 35B-A3B architecture profile;
- checkpoint config parsing and strict compatibility checks;
- MLX affine INT4/group-64 expert blob and SSD traffic calculations;
- a testable global LFU-plus-recency expert-cache policy;
- a hard-budget memory governor inspired by SlotStream;
- a separate prefill chunk planner;
- deterministic, normalized top-k router reference code;
- a small inspection CLI.

This package builds on Linux and macOS and does not need model weights.

### 2. A Swiftlet overlay

`swiftlet-overlay` contains files intended to be copied into a current Swiftlet
checkout:

- `OrnithSupport.swift`: explicit architecture profile and checkpoint guard;
- `SlotStreamMemoryGovernor.swift`: converts a total RAM target into whole
  expert-cache slots after reserving dense weights, the FP32 DeltaNet recurrence
  and convolution history, current Swiftlet FP32 KV, scratch, and safety headroom;
- `OrnithRuntimeFactory.swift`: validates the Ornith config and full qpack
  geometry, verifies the actual dense/expert payload file sizes, computes the
  plan from the qpack dense-file size and expert stride, and constructs
  `QwenMetalModel` with the resulting cache budget;
- Swift Testing unit tests;
- an installer script and a patch containing the same additions.

The root package currently has **27 passing tests**. The Swiftlet overlay has
**12 passing smoke tests** against a checked-in compatibility surface mirroring
the current upstream APIs. The complete test transcripts are included as
`TEST_RESULTS.txt` and `OVERLAY_SMOKE_RESULTS.txt`.

### 3. Design and benchmark notes

The `design` directory separates what is ready now from later performance work.
It includes the intended file-level port map, validation gates, and an ordered
benchmark plan.

## Tested calculations

For the native K=8 Ornith configuration and MLX affine INT4/group-64 experts:

- 40 layers: 30 DeltaNet and 10 full-attention;
- 256 routed experts per layer, 8 selected per token;
- one packed routed expert: 1,769,472 bytes;
- all routed experts: 18,119,393,280 bytes, or 16.875 GiB;
- zero-cache routed-expert traffic: 566,231,040 bytes, or 540 MiB/token;
- architectural FP16 KV lower bound: 20,480 bytes/token;
- current Swiftlet session KV allocation: 40,960 bytes/token because its cache
  is held in Swift `Float` arrays;
- fixed FP32 DeltaNet state: 62,914,560 bytes of recurrence matrices plus
  2,949,120 bytes of convolution history, or 65,863,680 bytes total
  (62.8125 MiB).

These figures are asserted by unit tests rather than documented as unchecked
estimates.

## Run the standalone tests

```sh
cd ornith-swiftlet-port
swift test
```

Inspect a config and calculate a memory plan:

```sh
swift run ornith-port-inspect \
  Tests/OrnithSwiftletPortTests/Fixtures/ornith-1.5-35b-config.json \
  --memory-gb 4 \
  --context 8192
```

The included fixture is a compact, faithful copy of the architecture-bearing
parts of the official config. It uses a compact `quantization` block instead of
the official checkpoint's much larger `quantization_config`; tests cover both
keys and verify the same default affine INT4/group-64 expert quantization.

## Apply the overlay to Swiftlet

```sh
git clone https://github.com/leonickson1/Swiftlet.git
cd Swiftlet
swift build -c release

/path/to/ornith-swiftlet-port/swiftlet-overlay/scripts/install.sh "$PWD"
swift test
```

The installer adds files; it does not overwrite Swiftlet's existing model,
kernel, cache, or qpack implementations.

The equivalent patch is:

```sh
git apply /path/to/ornith-swiftlet-port/swiftlet-overlay/patches/0001-ornith-port-overlay.patch
```

## Repack and run the model on Apple Silicon

After applying the overlay to Swiftlet:

```sh
swift build -c release

.build/release/swiftlet-repack \
  --from-hf ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit \
  --output "$HOME/models/ornith-1.5-35b.qpack"
```

The unmodified Swiftlet CLI can run the qpack with an explicit expert-cache
budget:

```sh
.build/release/swiftlet generate \
  "$HOME/models/ornith-1.5-35b.qpack" \
  --gpu --chat --cache-gb 2 \
  --prompt "Explain why sparse MoE models can be streamed from SSD."
```

Application code can use the new total-memory factory instead:

```swift
let build = try OrnithRuntimeFactory.make(
    modelDir: URL(fileURLWithPath: modelPath),
    options: OrnithRuntimeOptions(
        targetMemoryGiB: 4,
        plannedContextTokens: 8_192
    )
)

let model = build.model
print(build.memoryPlan.cacheBudgetGiB)
```

The factory uses the qpack's actual `model.safetensors` byte size and expert
stride. Its `preflight` entry point performs the same checks and returns the
memory plan without initializing Metal, which makes deployment validation
testable in CI. Scratch and safety reserves remain tunable planning inputs.
Before constructing Metal state, it also checks QPACK version/magic, native K=8
geometry, INT4/group-64 metadata, all nine expert sections, the 30/10 layer
pattern, all 40 expert-file manifest entries, and on-disk payload sizes.

## What this port does not pretend to complete

The archive does not include the 19+ GB checkpoint and did not execute full
model inference in this environment. Full end-to-end validation requires Apple
Silicon, Metal, the official MLX checkpoint, and an mlx-lm reference run.

The overlay source and tests compiled in a Linux smoke harness against the
reviewed current Swiftlet API surface, but this environment could not run Metal
or a complete upstream Apple-platform build. It also does
not yet implement:

- Ornith's one-layer MTP speculative path;
- vision input or vision weights;
- bounded parallel expert reads in place of Swiftlet's current serial miss
  fills;
- a fused/persistent top-8 MoE Metal kernel;
- genuinely batched prefill.

Those are kept out of the baseline on purpose. Native K=8 token parity and
memory invariance should be established before changing scheduling or numerical
reduction order.

## Recommended sequence

1. Apply the overlay and run all Swiftlet tests.
2. Repack the official MLX INT4 checkpoint.
3. Compare greedy token IDs against mlx-lm with a short deterministic corpus.
4. Sweep cache budgets and capture memory, misses, I/O, dispatches, and tok/s.
5. Optimize the measured bottleneck: dispatch/kernel work first when compute
   bound; I/O concurrency only when miss waits are visible.
6. Add MTP only after the baseline is stable.

See `design/BENCHMARK_PLAN.md` for concrete acceptance gates. For a
reproducible API-surface compile check without Metal, run
`scripts/overlay-smoke-test.sh`.

## Source snapshot and licensing

The overlay was designed against Swiftlet `main` as reviewed on 2026-09-02
(commit `aaa910a`). See `design/UPSTREAM_SNAPSHOT.md` for the exact API
assumptions.
Swiftlet and this kit use Apache License 2.0. This archive does not vendor model
weights or upstream Swiftlet source files.

Upstream references:

- https://github.com/leonickson1/Swiftlet
- https://github.com/carloslfu/slotstream
- https://github.com/drumih/turbo-fieldfare
- https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit
