#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TMP=${TMPDIR:-/tmp}/ornith-swiftlet-overlay-smoke
rm -rf "$TMP"
mkdir -p "$TMP/Sources/SwiftletCore" "$TMP/Tests/SwiftletCoreTests"

cat > "$TMP/Package.swift" <<'SWIFT'
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SwiftletOverlaySmoke",
    products: [.library(name: "SwiftletCore", targets: ["SwiftletCore"])],
    targets: [
        .target(name: "SwiftletCore"),
        .testTarget(name: "SwiftletCoreTests", dependencies: ["SwiftletCore"]),
    ]
)
SWIFT

cp "$ROOT/smoke/UpstreamAPIShims.swift" "$TMP/Sources/SwiftletCore/"
cp "$ROOT"/swiftlet-overlay/Sources/SwiftletCore/*.swift "$TMP/Sources/SwiftletCore/"
cp "$ROOT"/swiftlet-overlay/Tests/SwiftletCoreTests/*.swift "$TMP/Tests/SwiftletCoreTests/"

cd "$TMP"
swift test
