#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
swift test
swift build -c release
swift run ornith-port-inspect \
  Tests/OrnithSwiftletPortTests/Fixtures/ornith-1.5-35b-config.json \
  --memory-gb 4 --context 8192
