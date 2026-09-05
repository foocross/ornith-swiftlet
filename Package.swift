// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "OrnithSwiftletPort",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "OrnithSwiftletPort", targets: ["OrnithSwiftletPort"]),
        .executable(name: "ornith-port-inspect", targets: ["ornith-port-inspect"])
    ],
    targets: [
        .target(name: "OrnithSwiftletPort"),
        .executableTarget(
            name: "ornith-port-inspect",
            dependencies: ["OrnithSwiftletPort"]
        ),
        .testTarget(
            name: "OrnithSwiftletPortTests",
            dependencies: ["OrnithSwiftletPort"],
            resources: [.copy("Fixtures")]
        )
    ]
)
