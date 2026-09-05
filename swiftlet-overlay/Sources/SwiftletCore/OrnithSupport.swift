import Foundation

/// Explicit checkpoint profile for Ornith 1.5. The compute graph does not need
/// a new implementation: Ornith uses the qwen3_5_moe family that Swiftlet
/// already executes. This profile makes the accepted geometry reviewable and
/// gives tests a stable target independent of Qwen3.6 naming.
extension ArchConfig {
  public static let ornith1_5_35B = ArchConfig(
    family: .qwen3_5Moe,
    name: "Ornith-1.5-35B-A3B",
    repackSource: "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit",
    hiddenSize: 2_048,
    layerCount: 40,
    fullAttentionInterval: 4,
    vocabSize: 248_320,
    tieWordEmbeddings: false,
    rmsNormEps: 1e-6,
    attnHeads: 16,
    kvHeads: 2,
    headDim: 256,
    partialRotaryFactor: 0.25,
    ropeTheta: 10_000_000,
    maxPositionEmbeddings: 262_144,
    linearVHeads: 32,
    linearKHeads: 16,
    linearKHeadDim: 128,
    linearVHeadDim: 128,
    convKernelSize: 4,
    expertCount: 256,
    expertTopK: 8,
    moeIntermediateSize: 512,
    sharedExpertIntermediateSize: 512,
    normTopKProb: true
  )
}

public struct OrnithConfigValidation: Sendable, Equatable {
  public let mismatches: [String]
  public var isCompatible: Bool { mismatches.isEmpty }
}

public struct OrnithConfigMismatchError: Error, Sendable, CustomStringConvertible {
  public let mismatches: [String]
  public var description: String {
    "checkpoint is not Ornith-1.5-35B-A3B compatible:\n" + mismatches.joined(separator: "\n")
  }
}

extension QwenConfig {
  /// Validates every architecture field Swiftlet currently consumes. Fields
  /// absent from QwenConfig (MTP count and explicit layer_types) are covered
  /// by the standalone kit and should be added upstream before MTP support.
  public func validateAsOrnith1_5_35B() -> OrnithConfigValidation {
    let expected = ArchConfig.ornith1_5_35B
    var mismatches: [String] = []

    func check<T: Equatable>(_ field: String, _ actual: T, _ wanted: T) {
      guard actual != wanted else { return }
      mismatches.append("\(field): expected \(wanted), got \(actual)")
    }

    guard modelType.hasPrefix("qwen3_5_moe") else {
      mismatches.append("model_type: expected qwen3_5_moe family, got \(modelType)")
      return OrnithConfigValidation(mismatches: mismatches)
    }

    check("hidden_size", hiddenSize, expected.hiddenSize)
    check("num_hidden_layers", numHiddenLayers, expected.layerCount)
    check("num_attention_heads", numAttentionHeads, expected.attnHeads)
    check("num_key_value_heads", numKeyValueHeads, expected.kvHeads)
    check("head_dim", headDim, expected.headDim)
    check("partial_rotary_factor", partialRotaryFactor, expected.partialRotaryFactor)
    check("rope_theta", ropeTheta, expected.ropeTheta)
    check("rms_norm_eps", rmsNormEps, expected.rmsNormEps)
    check("vocab_size", vocabSize, expected.vocabSize)
    check("tie_word_embeddings", tieWordEmbeddings, expected.tieWordEmbeddings)
    check("full_attention_interval", fullAttentionInterval, expected.fullAttentionInterval)
    check("linear_num_value_heads", linearNumValueHeads, expected.linearVHeads)
    check("linear_num_key_heads", linearNumKeyHeads, expected.linearKHeads)
    check("linear_key_head_dim", linearKeyHeadDim, expected.linearKHeadDim)
    check("linear_value_head_dim", linearValueHeadDim, expected.linearVHeadDim)
    check("linear_conv_kernel_dim", linearConvKernelDim, expected.convKernelSize)
    check("num_experts", numExperts, expected.expertCount)
    check("num_experts_per_tok", numExpertsPerTok, expected.expertTopK)
    check("moe_intermediate_size", moeIntermediateSize, expected.moeIntermediateSize)
    check(
      "shared_expert_intermediate_size",
      sharedExpertIntermediateSize,
      expected.sharedExpertIntermediateSize
    )
    check("norm_topk_prob", normTopkProb, expected.normTopKProb)
    check("weight prefix", weightPrefix, "language_model.")

    switch deltaLayout {
    case .split:
      break
    case .fusedInterleaved:
      mismatches.append("DeltaNet projections: expected split qwen3_5 layout")
    }

    let linearCount = (0..<numHiddenLayers).filter(isLinearLayer).count
    check("linear layer count", linearCount, expected.linearLayerCount)
    check(
      "full-attention layer count", numHiddenLayers - linearCount, expected.fullAttentionLayerCount)

    return OrnithConfigValidation(mismatches: mismatches)
  }

  public func requireOrnith1_5_35B() throws {
    let validation = validateAsOrnith1_5_35B()
    guard validation.isCompatible else {
      throw OrnithConfigMismatchError(mismatches: validation.mismatches)
    }
  }
}

public struct OrnithQpackValidation: Sendable, Equatable {
  public let mismatches: [String]
  public var isCompatible: Bool { mismatches.isEmpty }
}

public struct OrnithQpackMismatchError: Error, Sendable, CustomStringConvertible {
  public let mismatches: [String]
  public var description: String {
    "qpack is not Ornith-1.5-35B-A3B MLX-int4 compatible:\n"
      + mismatches.joined(separator: "\n")
  }
}

extension Qpack.Layout {
  /// Checks the streamable expert container, not just config.json. This
  /// prevents a geometrically compatible config from opening a truncated,
  /// differently quantized, or otherwise mismatched qpack.
  public func validateAsOrnith1_5_35B(manifest: Qpack.Manifest) -> OrnithQpackValidation {
    let architecture = ArchConfig.ornith1_5_35B
    var mismatches: [String] = []

    func check<T: Equatable>(_ field: String, _ actual: T, _ wanted: T) {
      guard actual != wanted else { return }
      mismatches.append("\(field): expected \(wanted), got \(actual)")
    }

    check("manifest.magic", manifest.magic, "QPACK")
    check("manifest.version", manifest.version, Qpack.manifestVersion)
    if !manifest.modelName.hasPrefix("qwen3_5_moe") {
      mismatches.append(
        "manifest.modelName: expected qwen3_5_moe family, got \(manifest.modelName)"
      )
    }
    check("manifest.quantBits", manifest.quantBits, 4)
    check("manifest.quantGroupSize", manifest.quantGroupSize, 64)

    check("layout.expertCount", expertCount, architecture.expertCount)
    check("layout.layerCount", layerCount, architecture.layerCount)
    check(
      "layout.expertStride",
      expertStride,
      architecture.expertBlobBytesInt4G64
    )

    let expectedLinearLayers = (0..<architecture.layerCount).map(architecture.isLinearLayer)
    check("layout.linearLayers", linearLayers, expectedLinearLayers)

    struct ExpectedSection {
      let shape: [Int]
      let offset: Int
      let size: Int
      let weight: Bool
    }
    let expectedSections: [String: ExpectedSection] = [
      "gate_proj.weight": .init(shape: [512, 256], offset: 0, size: 524_288, weight: true),
      "gate_proj.scales": .init(shape: [512, 32], offset: 524_288, size: 32_768, weight: false),
      "gate_proj.biases": .init(shape: [512, 32], offset: 557_056, size: 32_768, weight: false),
      "up_proj.weight": .init(shape: [512, 256], offset: 589_824, size: 524_288, weight: true),
      "up_proj.scales": .init(shape: [512, 32], offset: 1_114_112, size: 32_768, weight: false),
      "up_proj.biases": .init(shape: [512, 32], offset: 1_146_880, size: 32_768, weight: false),
      "down_proj.weight": .init(
        shape: [2_048, 64], offset: 1_179_648, size: 524_288, weight: true),
      "down_proj.scales": .init(shape: [2_048, 8], offset: 1_703_936, size: 32_768, weight: false),
      "down_proj.biases": .init(shape: [2_048, 8], offset: 1_736_704, size: 32_768, weight: false),
    ]

    var actualByName: [String: Qpack.Section] = [:]
    for section in sections {
      if actualByName.updateValue(section, forKey: section.name) != nil {
        mismatches.append("layout.sections: duplicate section \(section.name)")
      }
      if section.offset < 0 || section.size <= 0 {
        mismatches.append(
          "layout.sections.\(section.name): invalid range offset=\(section.offset), size=\(section.size)"
        )
      } else if section.offset > expertStride - section.size {
        mismatches.append(
          "layout.sections.\(section.name): range exceeds expert stride"
        )
      }
    }

    let actualNames = Set(actualByName.keys)
    let expectedNames = Set(expectedSections.keys)
    for name in expectedNames.subtracting(actualNames).sorted() {
      mismatches.append("layout.sections: missing \(name)")
    }
    for name in actualNames.subtracting(expectedNames).sorted() {
      mismatches.append("layout.sections: unexpected \(name)")
    }

    for (name, expected) in expectedSections.sorted(by: { $0.key < $1.key }) {
      guard let actual = actualByName[name] else { continue }
      check("layout.sections.\(name).shape", actual.shape, expected.shape)
      check("layout.sections.\(name).offset", actual.offset, expected.offset)
      check("layout.sections.\(name).size", actual.size, expected.size)
      if expected.weight {
        check("layout.sections.\(name).dtype", actual.dtype, "U32")
      } else if actual.dtype != "F16" && actual.dtype != "BF16" {
        mismatches.append(
          "layout.sections.\(name).dtype: expected F16 or BF16, got \(actual.dtype)"
        )
      }
    }

    let sortedSections = sections.sorted { lhs, rhs in
      lhs.offset == rhs.offset ? lhs.name < rhs.name : lhs.offset < rhs.offset
    }
    for pair in zip(sortedSections, sortedSections.dropFirst()) {
      let leftEnd = pair.0.offset + pair.0.size
      if leftEnd > pair.1.offset {
        mismatches.append(
          "layout.sections: \(pair.0.name) overlaps \(pair.1.name)"
        )
      }
    }

    let expectedLayerBytes = architecture.expertCount * expertStride
    for layer in 0..<architecture.layerCount {
      let name = String(format: "packed_experts/layer_%02d.bin", layer)
      guard let size = manifest.files[name] else {
        mismatches.append("manifest.files: missing \(name)")
        continue
      }
      check("manifest.files[\(name)]", size, expectedLayerBytes)
    }

    guard let denseSize = manifest.files["model.safetensors"], denseSize > 0 else {
      mismatches.append("manifest.files: missing or empty model.safetensors")
      return OrnithQpackValidation(mismatches: mismatches)
    }

    return OrnithQpackValidation(mismatches: mismatches)
  }

  public func requireOrnith1_5_35B(manifest: Qpack.Manifest) throws {
    let validation = validateAsOrnith1_5_35B(manifest: manifest)
    guard validation.isCompatible else {
      throw OrnithQpackMismatchError(mismatches: validation.mismatches)
    }
  }
}
