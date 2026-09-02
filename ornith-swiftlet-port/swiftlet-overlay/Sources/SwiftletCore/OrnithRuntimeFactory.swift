import Foundation

public struct OrnithRuntimeOptions: Sendable, Equatable {
  public let targetMemoryGiB: Double
  public let plannedContextTokens: Int
  public let scratchBytes: Int64
  public let safetyBytes: Int64
  public let safetyFraction: Double

  public init(
    targetMemoryGiB: Double,
    plannedContextTokens: Int,
    scratchBytes: Int64 = 256 * 1_048_576,
    safetyBytes: Int64 = 256 * 1_048_576,
    safetyFraction: Double = 0.08
  ) {
    self.targetMemoryGiB = targetMemoryGiB
    self.plannedContextTokens = plannedContextTokens
    self.scratchBytes = scratchBytes
    self.safetyBytes = safetyBytes
    self.safetyFraction = safetyFraction
  }
}

public struct OrnithRuntimeBuild {
  public let model: QwenMetalModel
  public let memoryPlan: ExpertCacheMemoryPlan
}

public enum OrnithRuntimeFactoryError: Error, CustomStringConvertible {
  case notQpack(URL)
  case missingDenseSize
  case payloadSizeMismatch(path: String, expected: Int, actual: Int?)

  public var description: String {
    switch self {
    case .notQpack(let url):
      return "Ornith runtime requires a repacked .qpack directory: \(url.path)"
    case .missingDenseSize:
      return "manifest.json does not contain model.safetensors size"
    case .payloadSizeMismatch(let path, let expected, let actual):
      let actualDescription = actual.map(String.init) ?? "missing"
      return
        "qpack payload size mismatch for \(path): expected \(expected) B, got \(actualDescription)"
    }
  }
}

/// Concrete entry point for the port. It validates the checkpoint, reads the
/// actual qpack expert stride and dense-file size, computes a hard memory plan,
/// and passes only the resulting expert-cache budget to Swiftlet.
public enum OrnithRuntimeFactory {
  /// Runs every validation and memory-planning step without initializing Metal.
  /// This is useful for CI, deployment checks, and unit tests with sparse files.
  public static func preflight(
    modelDir: URL,
    options: OrnithRuntimeOptions
  ) throws -> ExpertCacheMemoryPlan {
    let configURL = modelDir.appendingPathComponent("config.json")
    let layoutURL = modelDir.appendingPathComponent("packed_experts/layout.json")
    let manifestURL = modelDir.appendingPathComponent("manifest.json")
    guard FileManager.default.fileExists(atPath: layoutURL.path),
      FileManager.default.fileExists(atPath: manifestURL.path)
    else {
      throw OrnithRuntimeFactoryError.notQpack(modelDir)
    }

    let config = try QwenConfig(url: configURL)
    try config.requireOrnith1_5_35B()

    let layout = try JSONDecoder().decode(
      Qpack.Layout.self,
      from: Data(contentsOf: layoutURL)
    )
    let manifest = try JSONDecoder().decode(
      Qpack.Manifest.self,
      from: Data(contentsOf: manifestURL)
    )
    try layout.requireOrnith1_5_35B(manifest: manifest)
    guard let denseSize = manifest.files["model.safetensors"], denseSize > 0 else {
      throw OrnithRuntimeFactoryError.missingDenseSize
    }
    try verifyPayloadFiles(
      modelDir: modelDir,
      manifest: manifest,
      architecture: .ornith1_5_35B
    )

    return try ExpertCacheMemoryGovernor.plan(
      architecture: .ornith1_5_35B,
      targetMemoryGiB: options.targetMemoryGiB,
      contextTokens: options.plannedContextTokens,
      denseCoreBytes: Int64(denseSize),
      expertStrideBytes: Int64(layout.expertStride),
      scratchBytes: options.scratchBytes,
      fixedSafetyBytes: options.safetyBytes,
      safetyFraction: options.safetyFraction
    )
  }

  public static func make(
    modelDir: URL,
    options: OrnithRuntimeOptions
  ) throws -> OrnithRuntimeBuild {
    let plan = try preflight(modelDir: modelDir, options: options)
    let model = try QwenMetalModel(
      modelDir: modelDir,
      cacheBudgetGB: plan.cacheBudgetGiB
    )
    return OrnithRuntimeBuild(model: model, memoryPlan: plan)
  }

  private static func verifyPayloadFiles(
    modelDir: URL,
    manifest: Qpack.Manifest,
    architecture: ArchConfig
  ) throws {
    let required =
      ["model.safetensors"]
      + (0..<architecture.layerCount).map { layer in
        String(format: "packed_experts/layer_%02d.bin", layer)
      }
    let fileManager = FileManager.default

    for relativePath in required {
      guard let expected = manifest.files[relativePath] else { continue }
      let url = modelDir.appendingPathComponent(relativePath)
      let attributes = try? fileManager.attributesOfItem(atPath: url.path)
      let actual = (attributes?[.size] as? NSNumber)?.intValue
      guard actual == expected else {
        throw OrnithRuntimeFactoryError.payloadSizeMismatch(
          path: relativePath,
          expected: expected,
          actual: actual
        )
      }
    }
  }
}
