#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 /path/to/Swiftlet" >&2
    exit 2
fi

SOURCE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TARGET_DIR=$1

if [ ! -f "$TARGET_DIR/Package.swift" ] || [ ! -d "$TARGET_DIR/Sources/SwiftletCore" ]; then
    echo "not a Swiftlet checkout: $TARGET_DIR" >&2
    exit 2
fi

for relative in \
    Sources/SwiftletCore/OrnithSupport.swift \
    Sources/SwiftletCore/SlotStreamMemoryGovernor.swift \
    Sources/SwiftletCore/OrnithRuntimeFactory.swift \
    Tests/SwiftletCoreTests/OrnithSupportTests.swift \
    assets/model-configs/ornith-1.5-35b-mlx4bit.json
do
    if [ -e "$TARGET_DIR/$relative" ]; then
        echo "refusing to overwrite existing file: $TARGET_DIR/$relative" >&2
        exit 2
    fi
done

cp "$SOURCE_DIR/Sources/SwiftletCore/OrnithSupport.swift" \
   "$TARGET_DIR/Sources/SwiftletCore/OrnithSupport.swift"
cp "$SOURCE_DIR/Sources/SwiftletCore/SlotStreamMemoryGovernor.swift" \
   "$TARGET_DIR/Sources/SwiftletCore/SlotStreamMemoryGovernor.swift"
cp "$SOURCE_DIR/Sources/SwiftletCore/OrnithRuntimeFactory.swift" \
   "$TARGET_DIR/Sources/SwiftletCore/OrnithRuntimeFactory.swift"
cp "$SOURCE_DIR/Tests/SwiftletCoreTests/OrnithSupportTests.swift" \
   "$TARGET_DIR/Tests/SwiftletCoreTests/OrnithSupportTests.swift"
mkdir -p "$TARGET_DIR/assets/model-configs"
cp "$SOURCE_DIR/assets/model-configs/ornith-1.5-35b-mlx4bit.json" \
   "$TARGET_DIR/assets/model-configs/ornith-1.5-35b-mlx4bit.json"

echo "Ornith overlay installed. Run: cd $TARGET_DIR && swift test"
