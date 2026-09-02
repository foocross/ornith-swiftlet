import Foundation

public struct ArchConfig: Sendable, Equatable {
  public enum Family: String, Sendable {
    case qwen3Next = "qwen3_next"
    case qwen3_5Moe = "qwen3_5_moe"
  }
  public let family: Family
  public let name: String
  public let repackSource: String
  public let hiddenSize: Int
  public let layerCount: Int
  public let fullAttentionInterval: Int
  public let vocabSize: Int
  public let tieWordEmbeddings: Bool
  public let rmsNormEps: Double
  public let zeroCenteredNorms: Bool = true
  public let attnHeads: Int
  public let kvHeads: Int
  public let headDim: Int
  public let partialRotaryFactor: Double
  public let ropeTheta: Double
  public let maxPositionEmbeddings: Int
  public let linearVHeads: Int
  public let linearKHeads: Int
  public let linearKHeadDim: Int
  public let linearVHeadDim: Int
  public let convKernelSize: Int
  public let expertCount: Int
  public let expertTopK: Int
  public let moeIntermediateSize: Int
  public let sharedExpertIntermediateSize: Int
  public let normTopKProb: Bool
  public var fullAttentionLayerCount: Int { layerCount / fullAttentionInterval }
  public var linearLayerCount: Int { layerCount - fullAttentionLayerCount }
  public func isLinearLayer(_ index: Int) -> Bool { (index + 1) % fullAttentionInterval != 0 }
  public var expertParamCount: Int { 3 * hiddenSize * moeIntermediateSize }
  public var routedExpertTotal: Int { layerCount * expertCount }
  public var routedFetchesPerToken: Int { layerCount * expertTopK }
  public var expertBlobBytesInt4G64: Int {
    let bits = expertParamCount * 4 + (expertParamCount / 64) * 32
    return bits / 8
  }
  public var kvBytesPerToken: Int { fullAttentionLayerCount * kvHeads * headDim * 2 * 2 }
  public var deltaNetStateBytes: Int {
    linearLayerCount * linearVHeads * linearKHeadDim * linearVHeadDim * 4
  }
}

public struct QwenConfig: Sendable {
  public enum DeltaProjectionLayout: Sendable {
    case fusedInterleaved
    case split
  }
  public var modelType: String
  public var hiddenSize: Int
  public var numHiddenLayers: Int
  public var numAttentionHeads: Int
  public var numKeyValueHeads: Int
  public var headDim: Int
  public var partialRotaryFactor: Double
  public var ropeTheta: Double
  public var rmsNormEps: Double
  public var vocabSize: Int
  public var tieWordEmbeddings: Bool
  public var fullAttentionInterval: Int
  public var linearNumValueHeads: Int
  public var linearNumKeyHeads: Int
  public var linearKeyHeadDim: Int
  public var linearValueHeadDim: Int
  public var linearConvKernelDim: Int
  public var numExperts: Int
  public var numExpertsPerTok: Int
  public var moeIntermediateSize: Int
  public var sharedExpertIntermediateSize: Int
  public var normTopkProb: Bool
  public var deltaLayout: DeltaProjectionLayout
  public var weightPrefix: String
  public enum Error: Swift.Error { case missingField(String) }

  public init(url: URL) throws {
    let top = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] ?? [:]
    let text = (top["text_config"] as? [String: Any]) ?? top
    let isNested = top["text_config"] != nil
    func int(_ key: String) throws -> Int {
      guard let value = text[key] as? Int else { throw Error.missingField(key) }
      return value
    }
    func dbl(_ key: String, _ fallback: Double? = nil) throws -> Double {
      if let value = text[key] as? Double { return value }
      if let value = text[key] as? Int { return Double(value) }
      if let fallback { return fallback }
      throw Error.missingField(key)
    }
    modelType = (text["model_type"] as? String) ?? (top["model_type"] as? String) ?? "qwen3_next"
    let isQwen35 = modelType.hasPrefix("qwen3_5") || modelType.hasPrefix("qwen3_6") || isNested
    hiddenSize = try int("hidden_size")
    numHiddenLayers = try int("num_hidden_layers")
    numAttentionHeads = try int("num_attention_heads")
    numKeyValueHeads = try int("num_key_value_heads")
    headDim = try int("head_dim")
    rmsNormEps = try dbl("rms_norm_eps", 1e-6)
    vocabSize = try int("vocab_size")
    tieWordEmbeddings =
      (text["tie_word_embeddings"] as? Bool) ?? (top["tie_word_embeddings"] as? Bool) ?? false
    fullAttentionInterval = (text["full_attention_interval"] as? Int) ?? 4
    if let rope = text["rope_parameters"] as? [String: Any] {
      ropeTheta =
        (rope["rope_theta"] as? Double) ?? Double(rope["rope_theta"] as? Int ?? 10_000_000)
      partialRotaryFactor = (rope["partial_rotary_factor"] as? Double) ?? 0.25
    } else {
      ropeTheta = try dbl("rope_theta")
      partialRotaryFactor = try dbl("partial_rotary_factor", 0.25)
    }
    linearNumValueHeads = try int("linear_num_value_heads")
    linearNumKeyHeads = try int("linear_num_key_heads")
    linearKeyHeadDim = try int("linear_key_head_dim")
    linearValueHeadDim = try int("linear_value_head_dim")
    linearConvKernelDim = try int("linear_conv_kernel_dim")
    numExperts = try int("num_experts")
    numExpertsPerTok = try int("num_experts_per_tok")
    moeIntermediateSize = try int("moe_intermediate_size")
    sharedExpertIntermediateSize = try int("shared_expert_intermediate_size")
    normTopkProb = (text["norm_topk_prob"] as? Bool) ?? isQwen35
    deltaLayout = isQwen35 ? .split : .fusedInterleaved
    weightPrefix = isNested ? "language_model." : ""
  }
  public func isLinearLayer(_ index: Int) -> Bool { (index + 1) % fullAttentionInterval != 0 }
}

public enum Qpack {
  public static let manifestVersion = 1
  public struct Section: Codable, Sendable {
    public let name: String
    public let dtype: String
    public let shape: [Int]
    public let offset: Int
    public let size: Int
  }
  public struct Layout: Codable, Sendable {
    public let expertCount: Int
    public let layerCount: Int
    public let expertStride: Int
    public let sections: [Section]
    public let linearLayers: [Bool]
  }
  public struct Manifest: Codable, Sendable {
    public let magic: String
    public let version: Int
    public let modelName: String
    public let sourceCheckpoint: String
    public let quantBits: Int?
    public let quantGroupSize: Int?
    public let files: [String: Int]
  }
}

public final class QwenMetalModel {
  public init(modelDir: URL, cacheBudgetGB: Double = 8) throws {}
}
