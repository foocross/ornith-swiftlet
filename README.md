# ornith-swiftlet

Performance-engineering project: converting and running `Ornith-1.5-35B-A3B`
(a Qwen3.5-MoE-architecture checkpoint) on
[Swiftlet](https://github.com/leonickson1/Swiftlet) — a Swift/Metal LLM
inference engine — under a hard memory budget on a single Apple Silicon
machine, plus a series of decode-throughput and memory optimizations layered
on top of a working port.

## Start here

- **[`synopsis.md`](synopsis.md)** — what this project is, why the
  requested GGUF checkpoint needed converting before Swiftlet could load it
  at all, and a running log of what's shipped.
- **[`CONVERSION_PLAN.md`](CONVERSION_PLAN.md)** — the detailed
  GGUF→MLX→qpack conversion pipeline, plus full write-ups and benchmark
  numbers for each throughput/memory optimization (router-aware expert
  prefetch, MoE kernel fusion, KV-cache INT8 quantization, CPU attention-core
  vectorization, MTP speculative decoding).
- **[`handoff.md`](handoff.md)** — session-to-session status, written for
  whoever picks this work up next.
- **[`PORT.md`](PORT.md)** — the standalone Swift port kit's own docs: what's
  in it, tested memory/traffic calculations, and how to build/test it and
  apply the Swiftlet overlay.

## Layout

- **`Sources/`, `Tests/`** — the standalone Swift package (`OrnithSwiftletPort`):
  architecture profile, checkpoint compatibility checks, expert-cache policy,
  memory governor, prefill planner, and router reference code. Builds on
  Linux and macOS without model weights; see [`PORT.md`](PORT.md).
- **`swiftlet-overlay/`** — the integration overlay (source + patches +
  installer) meant to be copied into a Swiftlet checkout.
- **`design/`** — port map, validation gates, and benchmark plan for the
  port kit.
- **`research/`** — background research notes that fed into the
  optimization work.
- **`scratch/`** — benchmark/sweep scripts and their logs, backing the
  numbers cited in `CONVERSION_PLAN.md`. Not needed to build or run
  anything; kept for reproducibility and audit.

## Dependencies

This repo does not vendor [Swiftlet](https://github.com/leonickson1/Swiftlet)
itself — clone it as a sibling directory (`../Swiftlet`, the default the
`scratch/` scripts assume) or point their `SWIFTLET`/`MODEL` environment
variables at your own build. Model weights (the Ornith GGUF/MLX checkpoints
and derived `.qpack` files) are never committed; see `synopsis.md` and
`.gitignore` for how they're produced and regenerated locally.

## License

Apache-2.0 — see [LICENSE](LICENSE).
