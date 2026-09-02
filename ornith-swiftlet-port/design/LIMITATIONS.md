# Limitations and known gaps

## Environment validation performed

- Standalone Swift package built with Swift 6.2.1 on x86_64 Linux.
- Twenty-seven standalone unit tests passed.
- The three Swiftlet overlay source files compiled against a checked-in
  compatibility shim matching the current public interfaces.
- Twelve overlay tests passed in that smoke package, including config drift,
  qpack geometry, quantization, manifest completeness, on-disk payload checks,
  runtime preflight, and memory planning.
- The inspection CLI ran against the included Ornith config fixture.

## Validation not performed

- No Apple Metal device was available in this execution environment.
- The official model checkpoint was not downloaded or repacked.
- No full Ornith forward pass or generation was executed.
- No mlx-lm versus Swiftlet token/logit comparison was possible.
- The overlay was not built inside a live upstream checkout because repository
  cloning was unavailable to the build container; it was checked against the
  reviewed current APIs through a local smoke harness.

## Planning assumptions

The standalone CLI defaults the resident dense core to 1,350 MiB. The Swiftlet
overlay does better: `OrnithRuntimeFactory` reads the qpack manifest's actual
`model.safetensors` size. Neither value automatically accounts for every
transient driver allocation, tokenizer object, server buffer, or OS allocation;
that is why scratch and safety reserves are explicit.

The planner reserves 40,960 bytes per token for K/V because Swiftlet's current
`QwenCPUModel.DecodeState` stores the ten full-attention layers' cache in Swift
`Float` arrays. The architectural FP16 lower bound is 20,480 bytes per token.
If Swiftlet moves K/V to FP16 Metal buffers, update the explicit runtime property
and its tests rather than silently reusing the lower bound.

The fixed linear-attention reservation includes both the FP32 GatedDelta
recurrence matrices and each layer's `(convKernel - 1) * convDim` history. It
still relies on the scratch and safety allowances to cover allocator overhead.

The expert layout calculation applies to routed gate/up/down projections and
assumes MLX affine INT4 with group size 64 and BF16/F16 scale and bias values.
The official checkpoint may retain resident router/shared-gate tensors at a
higher precision; those are not part of the streamed expert blob. The runtime
factory reads the qpack's actual expert stride and then requires it to match the
official routed-expert layout. Supporting another quantization should be a
separate profile rather than silently weakening the guard.

## Runtime gaps

~~Swiftlet's current expert-cache fill path performs a synchronous read for
each miss. The overlay intentionally does not replace it without real
traces.~~ Traced (see `CONVERSION_PLAN.md`, "Decode throughput"): serial fill
was ~46% of decode wall time and fully idled the GPU. Fixed in
`ExpertCache.buffers()` with bounded concurrent `pread` per miss batch
(width via `SWIFTLET_EXPERT_READ_CONCURRENCY`, default 8) plus a thread-safe
fd-open in `QpackExpertReader`; ~56% decode throughput improvement measured
on the reference qpack, generated token IDs unchanged. Not yet re-verified
against other cache budgets, longer contexts, or the iOS memory-pressure path.

Swiftlet's current qpack repacker excludes MTP and vision weights. This kit is a
text-only, baseline-K=8 port.

The factory plans for a maximum context but cannot force callers to stop at that
length. The application layer must either enforce the planned context or ask the
memory governor for a revised cache allocation before a longer request.

Shrinking the cache is already supported by Swiftlet, but cache growth after
model construction needs a deliberate API if the server should reclaim memory
between requests in both directions.
