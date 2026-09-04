#!/bin/sh
set -eu

# Router-aware expert prefetch A/B (CONVERSION_PLAN.md "Router-aware expert
# prefetch"). Phase 0 (SWIFTLET_PREFETCH_DEBUG=1, no cache changes) measured
# hit@8 ~79%, hit@wideK ~94% against the real model -- this sweep checks
# whether Phase 1's real ExpertCache.prefetch(), wired into stepOneFast and
# gated behind SWIFTLET_EXPERT_PREFETCH, actually turns that into a decode
# throughput win, and at which --cache-gb it matters (or doesn't -- see the
# already-documented "--cache-gb gap": hit-rate gains not translating
# 1:1 into wall-clock gains is a known shape in this codebase, not
# automatically evidence of a bug).
#
# Runs against scratch/ornith-1.5-35b-base.qpack. Usage:
#   cd scratch && ./expert_prefetch_sweep.sh
# Output: scratch/expert-prefetch-sweep-<timestamp>/, one file pair per run
# (prefetch_<0|1>_cachegb_<N>_run<R>.{stdout,stderr}).

SWIFTLET=${SWIFTLET:-../Swiftlet/.build/release/swiftlet}
MODEL=${MODEL:-ornith-1.5-35b-base.qpack}
PROMPT="Explain in a few sentences how a mixture-of-experts language model routes tokens to experts."
MAX_NEW=${MAX_NEW:-150}
REPEATS=${REPEATS:-3}
CACHE_GBS=${CACHE_GBS:-"2 4 6"}
OUT=${OUT_DIR:-expert-prefetch-sweep-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"

if [ ! -x "$SWIFTLET" ]; then
    echo "error: $SWIFTLET not found or not executable (build it first: cd Swiftlet && swift build -c release)" >&2
    exit 1
fi
if [ ! -e "$MODEL" ]; then
    echo "error: $MODEL not found in $(pwd)" >&2
    exit 1
fi

for cache in $CACHE_GBS; do
    for prefetch in 0 1; do
        for run in $(seq 1 "$REPEATS"); do
            name="prefetch_${prefetch}_cachegb_${cache}_run${run}"
            echo "=== $name ===" >&2
            SWIFTLET_EXPERT_PREFETCH=$prefetch "$SWIFTLET" generate "$MODEL" \
                --gpu --chat --cache-gb "$cache" --max-new "$MAX_NEW" \
                --prompt "$PROMPT" \
                >"$OUT/${name}.stdout" 2>"$OUT/${name}.stderr"
        done
    done
done

echo "done: $OUT" >&2
echo "summarize with: grep -H 'decode:\|decode expert cache:\|prefetch-debug' $OUT/*.stderr" >&2
