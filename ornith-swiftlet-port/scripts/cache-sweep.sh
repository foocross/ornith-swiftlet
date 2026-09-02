#!/bin/sh
set -eu

if [ "$#" -lt 2 ]; then
    echo "usage: $0 /path/to/swiftlet /path/to/ornith.qpack [prompt]" >&2
    exit 2
fi

SWIFTLET=$1
MODEL=$2
PROMPT=${3:-Explain expert streaming in one paragraph.}
OUT=${OUT_DIR:-cache-sweep-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"

for CACHE in 0.25 0.5 1 2 4 8; do
    NAME=$(printf '%s' "$CACHE" | tr '.' '_')
    "$SWIFTLET" generate "$MODEL" \
        --gpu --chat --greedy --cache-gb "$CACHE" --max-new 64 \
        --prompt "$PROMPT" \
        >"$OUT/cache_${NAME}.stdout" \
        2>"$OUT/cache_${NAME}.stderr"
done

echo "results: $OUT"
