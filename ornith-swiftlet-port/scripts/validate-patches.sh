#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TMP=$(mktemp -d "${TMPDIR:-/tmp}/ornith-patch-check.XXXXXX")
trap 'rm -rf "$TMP"' EXIT HUP INT TERM

mkdir -p "$TMP/Sources/SwiftletCore"
cat > "$TMP/Sources/SwiftletCore/ArchConfig.swift" <<'SWIFT'
import Foundation
public struct ArchConfig: Sendable, Equatable {
    public static let known: [String: ArchConfig] = [
        "qwen3-next-80b": .qwen3Next80B,
        "qwen3.5-397b": .qwen3_5_397B,
        "qwen3.6-35b": .qwen3_6_35B,
    ]
}
SWIFT

cd "$TMP"
git init -q
git config user.email "port-kit@example.invalid"
git config user.name "Ornith Port Kit"
git add Sources/SwiftletCore/ArchConfig.swift
git commit -qm base

git apply --check "$ROOT/swiftlet-overlay/patches/0001-ornith-port-overlay.patch"
git apply "$ROOT/swiftlet-overlay/patches/0001-ornith-port-overlay.patch"
git apply --check "$ROOT/swiftlet-overlay/patches/0002-register-info-alias.patch"
git apply "$ROOT/swiftlet-overlay/patches/0002-register-info-alias.patch"

grep -q '"ornith-1.5-35b": .ornith1_5_35B' Sources/SwiftletCore/ArchConfig.swift
printf '%s\n' "patch application: PASS"
