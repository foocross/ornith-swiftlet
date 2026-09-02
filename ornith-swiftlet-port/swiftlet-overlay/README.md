# Swiftlet overlay

This directory contains the files that make the Ornith port explicit inside a
current Swiftlet checkout.

## Install by copying

```sh
./scripts/install.sh /path/to/Swiftlet
cd /path/to/Swiftlet
swift test
```

## Install as patches

From the root of a clean Swiftlet checkout:

```sh
git apply /path/to/0001-ornith-port-overlay.patch
swift test
```

Patch `0002-register-info-alias.patch` is optional. It adds
`swiftlet info ornith-1.5-35b`; runtime loading and repacking do not depend on
`ArchConfig.known`.

```sh
git apply /path/to/0002-register-info-alias.patch
```

## Runtime use

`OrnithRuntimeFactory` is intended for apps and servers that want a total-memory
knob rather than an expert-cache-only knob:

```swift
let build = try OrnithRuntimeFactory.make(
    modelDir: modelDirectory,
    options: OrnithRuntimeOptions(
        targetMemoryGiB: 4,
        plannedContextTokens: 8_192
    )
)

let model = build.model
let cacheGiB = build.memoryPlan.cacheBudgetGiB
```

The factory does not guess the expert stride or dense qpack size. It reads both
from `packed_experts/layout.json` and `manifest.json`, validates the complete
Ornith INT4 expert layout and manifest, and checks each required payload file
against its declared size before creating `QwenMetalModel`. The same work is
available without Metal initialization through `OrnithRuntimeFactory.preflight`.
The memory plan reflects Swiftlet's current FP32 session KV arrays and includes
both DeltaNet recurrence matrices and convolution history.

## Why no kernel changes are in this overlay

Ornith uses the Qwen3.5 MoE text architecture already present in Swiftlet. The
first milestone is checkpoint and greedy-output parity at native K=8. Parallel
reads, fused top-8 MoE, batched prefill, and MTP each affect scheduling or
numerical behavior and should be separate measured changes after that baseline.

## Cross-platform overlay smoke test

From the archive root:

```sh
scripts/overlay-smoke-test.sh
```

This compiles the added source against a minimal copy of the current Swiftlet
API surface and runs the overlay tests without importing Metal. It is a compile
guard, not a replacement for running the complete upstream suite on Apple
Silicon.
