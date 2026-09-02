import Foundation

public struct CompatibilityIssue: Sendable, Equatable, CustomStringConvertible {
  public enum Severity: String, Sendable {
    case error
    case warning
  }

  public let severity: Severity
  public let field: String
  public let expected: String
  public let actual: String

  public var description: String {
    "[\(severity.rawValue)] \(field): expected \(expected), got \(actual)"
  }
}

public struct OrnithCompatibilityReport: Sendable, Equatable {
  public let issues: [CompatibilityIssue]

  public var errors: [CompatibilityIssue] {
    issues.filter { $0.severity == .error }
  }

  public var warnings: [CompatibilityIssue] {
    issues.filter { $0.severity == .warning }
  }

  public var isCompatible: Bool {
    errors.isEmpty
  }
}

public enum OrnithCompatibility {
  public static func inspect(
    _ config: OrnithCheckpointConfig,
    against profile: OrnithProfile = .v1_5_35BA3B
  ) -> OrnithCompatibilityReport {
    var issues: [CompatibilityIssue] = []

    func check<T: Equatable & CustomStringConvertible>(
      _ field: String,
      _ actual: T,
      _ expected: T,
      severity: CompatibilityIssue.Severity = .error
    ) {
      guard actual != expected else { return }
      issues.append(
        CompatibilityIssue(
          severity: severity,
          field: field,
          expected: expected.description,
          actual: actual.description
        ))
    }

    check("model_type", config.modelType, profile.modelType)
    check("hidden_size", config.hiddenSize, profile.hiddenSize)
    check("num_hidden_layers", config.layerCount, profile.layerCount)
    check("num_attention_heads", config.attentionHeads, profile.attentionHeads)
    check("num_key_value_heads", config.keyValueHeads, profile.keyValueHeads)
    check("head_dim", config.headDimension, profile.headDimension)
    check("vocab_size", config.vocabSize, profile.vocabSize)
    check("full_attention_interval", config.fullAttentionInterval, profile.fullAttentionInterval)
    check("linear_num_value_heads", config.linearValueHeads, profile.linearValueHeads)
    check("linear_num_key_heads", config.linearKeyHeads, profile.linearKeyHeads)
    check("linear_key_head_dim", config.linearKeyHeadDimension, profile.linearKeyHeadDimension)
    check(
      "linear_value_head_dim", config.linearValueHeadDimension, profile.linearValueHeadDimension)
    check("linear_conv_kernel_dim", config.convolutionKernelSize, profile.convolutionKernelSize)
    check("num_experts", config.expertCount, profile.expertCount)
    check("num_experts_per_tok", config.expertsPerToken, profile.expertsPerToken)
    check("moe_intermediate_size", config.expertIntermediateSize, profile.expertIntermediateSize)
    check(
      "shared_expert_intermediate_size",
      config.sharedExpertIntermediateSize,
      profile.sharedExpertIntermediateSize
    )
    check("norm_topk_prob", config.normalizeTopK, profile.normalizeTopK)
    check(
      "mtp_num_hidden_layers", config.mtpHiddenLayerCount, profile.mtpHiddenLayerCount,
      severity: .warning)
    check("weight_prefix", config.weightPrefix, "language_model.")
    check("delta_projection_layout", config.deltaProjectionLayout.rawValue, "split")
    check("layer_types.count", config.layerTypes.count, profile.layerCount)

    if config.layerTypes.count == profile.layerCount {
      for index in config.layerTypes.indices
      where config.layerTypes[index] != profile.layerTypes[index] {
        issues.append(
          CompatibilityIssue(
            severity: .error,
            field: "layer_types[\(index)]",
            expected: profile.layerTypes[index].rawValue,
            actual: config.layerTypes[index].rawValue
          ))
      }
    }

    if let architecture = config.architecture {
      check("architectures[0]", architecture, profile.architecture)
    } else {
      issues.append(
        CompatibilityIssue(
          severity: .warning,
          field: "architectures[0]",
          expected: profile.architecture,
          actual: "missing"
        ))
    }

    if let quantization = config.quantization {
      check("quantization.bits", quantization.bits, 4)
      check("quantization.group_size", quantization.groupSize, 64)
    } else {
      issues.append(
        CompatibilityIssue(
          severity: .warning,
          field: "quantization",
          expected: "MLX affine int4, group_size 64",
          actual: "not declared in config.json; verify tensor metadata"
        ))
    }

    return OrnithCompatibilityReport(issues: issues)
  }
}
