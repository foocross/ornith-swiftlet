import Foundation

public struct OrnithCheckpointConfig: Sendable, Equatable {
  public enum DeltaProjectionLayout: String, Sendable {
    case split
    case fusedInterleaved
  }

  public enum ParseError: Error, Equatable, CustomStringConvertible {
    case invalidTopLevel
    case missingField(String)
    case invalidField(String)

    public var description: String {
      switch self {
      case .invalidTopLevel:
        return "config root is not a JSON object"
      case .missingField(let field):
        return "missing required field: \(field)"
      case .invalidField(let field):
        return "invalid field: \(field)"
      }
    }
  }

  public let architecture: String?
  public let modelType: String
  public let hiddenSize: Int
  public let layerCount: Int
  public let attentionHeads: Int
  public let keyValueHeads: Int
  public let headDimension: Int
  public let partialRotaryFactor: Double
  public let ropeTheta: Double
  public let rmsNormEpsilon: Double
  public let vocabSize: Int
  public let tieWordEmbeddings: Bool
  public let fullAttentionInterval: Int
  public let layerTypes: [OrnithLayerType]

  public let linearValueHeads: Int
  public let linearKeyHeads: Int
  public let linearKeyHeadDimension: Int
  public let linearValueHeadDimension: Int
  public let convolutionKernelSize: Int

  public let expertCount: Int
  public let expertsPerToken: Int
  public let expertIntermediateSize: Int
  public let sharedExpertIntermediateSize: Int
  public let normalizeTopK: Bool
  public let mtpHiddenLayerCount: Int

  public let weightPrefix: String
  public let deltaProjectionLayout: DeltaProjectionLayout
  public let quantization: QuantizationSpec?

  public init(url: URL) throws {
    try self.init(data: Data(contentsOf: url))
  }

  public init(data: Data) throws {
    guard let top = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      throw ParseError.invalidTopLevel
    }
    let text = (top["text_config"] as? [String: Any]) ?? top
    let nested = top["text_config"] != nil

    func int(_ key: String, fallback: Int? = nil) throws -> Int {
      if let value = text[key] as? Int { return value }
      if let value = text[key] as? NSNumber { return value.intValue }
      if let fallback { return fallback }
      throw ParseError.missingField(key)
    }

    func double(_ key: String, fallback: Double? = nil) throws -> Double {
      if let value = text[key] as? Double { return value }
      if let value = text[key] as? Int { return Double(value) }
      if let value = text[key] as? NSNumber { return value.doubleValue }
      if let fallback { return fallback }
      throw ParseError.missingField(key)
    }

    architecture = (top["architectures"] as? [String])?.first
    modelType =
      (top["model_type"] as? String)
      ?? (text["model_type"] as? String)
      ?? "qwen3_next"

    hiddenSize = try int("hidden_size")
    let resolvedLayerCount = try int("num_hidden_layers")
    layerCount = resolvedLayerCount
    attentionHeads = try int("num_attention_heads")
    keyValueHeads = try int("num_key_value_heads")
    headDimension = try int("head_dim")
    rmsNormEpsilon = try double("rms_norm_eps", fallback: 1e-6)
    vocabSize = try int("vocab_size")
    tieWordEmbeddings =
      (text["tie_word_embeddings"] as? Bool)
      ?? (top["tie_word_embeddings"] as? Bool)
      ?? false
    let resolvedFullAttentionInterval = try int("full_attention_interval", fallback: 4)
    fullAttentionInterval = resolvedFullAttentionInterval

    let rope = text["rope_parameters"] as? [String: Any]
    if let rope {
      if let value = rope["rope_theta"] as? NSNumber {
        ropeTheta = value.doubleValue
      } else {
        ropeTheta = 10_000_000
      }
      if let value = rope["partial_rotary_factor"] as? NSNumber {
        partialRotaryFactor = value.doubleValue
      } else {
        partialRotaryFactor = try double("partial_rotary_factor", fallback: 0.25)
      }
    } else {
      ropeTheta = try double("rope_theta")
      partialRotaryFactor = try double("partial_rotary_factor", fallback: 0.25)
    }

    linearValueHeads = try int("linear_num_value_heads")
    linearKeyHeads = try int("linear_num_key_heads")
    linearKeyHeadDimension = try int("linear_key_head_dim")
    linearValueHeadDimension = try int("linear_value_head_dim")
    convolutionKernelSize = try int("linear_conv_kernel_dim")

    expertCount = try int("num_experts")
    expertsPerToken = try int("num_experts_per_tok")
    expertIntermediateSize = try int("moe_intermediate_size")
    sharedExpertIntermediateSize = try int("shared_expert_intermediate_size")
    mtpHiddenLayerCount = try int("mtp_num_hidden_layers", fallback: 0)

    let qwen35Family =
      modelType.hasPrefix("qwen3_5")
      || modelType.hasPrefix("qwen3_6")
      || nested
    normalizeTopK = (text["norm_topk_prob"] as? Bool) ?? qwen35Family
    deltaProjectionLayout = qwen35Family ? .split : .fusedInterleaved
    weightPrefix = nested ? "language_model." : ""

    if let rawLayerTypes = text["layer_types"] as? [String] {
      layerTypes = try rawLayerTypes.enumerated().map { index, raw in
        guard let value = OrnithLayerType(rawValue: raw) else {
          throw ParseError.invalidField("layer_types[\(index)]")
        }
        return value
      }
    } else {
      layerTypes = (0..<resolvedLayerCount).map { index in
        (index + 1).isMultiple(of: resolvedFullAttentionInterval)
          ? .fullAttention
          : .linearAttention
      }
    }

    let quant =
      (top["quantization"] as? [String: Any])
      ?? (top["quantization_config"] as? [String: Any])
    if let quant {
      guard let bits = (quant["bits"] as? NSNumber)?.intValue else {
        throw ParseError.invalidField("quantization.bits")
      }
      guard let groupSize = (quant["group_size"] as? NSNumber)?.intValue else {
        throw ParseError.invalidField("quantization.group_size")
      }
      guard (1...16).contains(bits) else {
        throw ParseError.invalidField("quantization.bits")
      }
      guard groupSize > 0 else {
        throw ParseError.invalidField("quantization.group_size")
      }
      quantization = QuantizationSpec(bits: bits, groupSize: groupSize)
    } else {
      quantization = nil
    }
  }

  public func isLinearLayer(_ index: Int) -> Bool {
    precondition((0..<layerCount).contains(index), "layer index out of range")
    return layerTypes[index] == .linearAttention
  }
}
