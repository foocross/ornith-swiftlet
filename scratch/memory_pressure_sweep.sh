#!/bin/sh
set -eu

# Memory-pressure diagnostic for the open "--cache-gb gap" question
# (CONVERSION_PLAN.md "`--cache-gb` gap revisited: hazard-tracking
# hypothesis tested, rejected"). That section left real memory pressure
# on this 18GB-unified-memory machine as the leading *untested*
# explanation for a GPU wait-minus-exec gap that grows with --cache-gb
# (2.76s at 1,213 slots -> 4.83s at 4,854 slots, real GPU exec time flat
# throughout). This script samples system-wide (vm_stat) and per-process
# (footprint) memory-pressure signals in parallel with the reference
# swiftlet generate command at --cache-gb 2/4/6/8, to check whether either
# signal grows in a shape that tracks that gap.
#
# Runs against scratch/ornith-1.5-35b-base.qpack, NOT
# ornith-1.5-35b-crack.qpack (the model used for every number in the
# original sweep -- deleted per synopsis.md's disk-management notes, not
# present on this machine). Absolute tok/s here won't match the CRACK-build
# figures cited in CONVERSION_PLAN.md; the gap-vs-cache-size *shape* is
# what's under test, and that's a property of the runtime/memory layout
# (expert-cache slot buffers, resident dense weights), not the specific
# model weights.
#
# Usage: cd scratch && ./memory_pressure_sweep.sh
# Output: scratch/memory-pressure-sweep-<timestamp>/, one set of files per
# run (cachegb_<N>_run<R>.{stdout,stderr,vmstat.txt,footprint.json}).
# Analyze with ./analyze_memory_pressure.py <dir> afterward.

SWIFTLET=${SWIFTLET:-../Swiftlet/.build/release/swiftlet}
MODEL=${MODEL:-ornith-1.5-35b-base.qpack}
PROMPT="Write a short paragraph about the history of the Netherlands."
OUT=${OUT_DIR:-memory-pressure-sweep-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"

if [ ! -x "$SWIFTLET" ]; then
    echo "error: $SWIFTLET not found or not executable (build it first: cd Swiftlet && swift build -c release)" >&2
    exit 1
fi
if [ ! -e "$MODEL" ]; then
    echo "error: $MODEL not found in $(pwd)" >&2
    exit 1
fi

run_one() {
    cache=$1
    run=$2
    name="cachegb_${cache}_run${run}"
    echo "=== $name ===" >&2

    "$SWIFTLET" generate "$MODEL" \
        --gpu --chat --cache-gb "$cache" --max-new 200 \
        --prompt "$PROMPT" \
        >"$OUT/${name}.stdout" 2>"$OUT/${name}.stderr" &
    pid=$!

    # System-wide memory pressure (compressor activity, pageins/pageouts,
    # swapins/swapouts) at 1s resolution, plain text, one row per second.
    vm_stat 1 >"$OUT/${name}.vmstat.txt" 2>&1 &
    vmpid=$!
    # Per-process resident/compressed footprint at 0.5s resolution. -j
    # writes valid JSON only if footprint receives SIGINT (its own
    # documented "<ctrl-c> to stop" behavior) rather than SIGTERM/SIGKILL
    # -- confirmed empirically before writing this script.
    footprint -p "$pid" --sample 0.5 -j "$OUT/${name}.footprint.json" \
        >/dev/null 2>&1 &
    fppid=$!

    wait "$pid" || echo "  warning: swiftlet exited non-zero" >&2
    sleep 0.3
    kill -INT "$fppid" 2>/dev/null || true
    kill "$vmpid" 2>/dev/null || true
    wait "$fppid" 2>/dev/null || true
    wait "$vmpid" 2>/dev/null || true

    grep -E "decode:|decode Metal S3a|expert cache:" "$OUT/${name}.stderr" >&2 || true
}

for cache in 2 4 6 8; do
    run_one "$cache" 1
done
# Repeat the largest cache size, matching the repeat-run precedent already
# used for --cache-gb 8 in the hazard-tracking investigation.
run_one 8 2

echo "results in: $OUT" >&2
echo "next: ./analyze_memory_pressure.py $OUT" >&2
