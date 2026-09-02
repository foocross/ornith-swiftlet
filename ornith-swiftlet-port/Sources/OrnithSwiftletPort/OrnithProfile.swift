import Foundation

public enum OrnithLayerType: String, Codable, Sendable {
  case linearAttention = "linear_attention"
  case fullAttention = "full_attention"
}

public struct OrnithProfile: Sendable, Equatable {
  public let name: String
  public let sourceCheckpoint: String
  public let modelType: String
  public let architecture: String

  public let hiddenSize: Int
  public let layerCount: Int
  public let fullAttentionInterval: Int
  public let vocabSize: Int
  public let rmsNormEpsilon: Double

  public let attentionHeads: Int
  public let keyValueHeads: Int
  public let headDimension: Int
  public let partialRotaryFactor: Double
  public let ropeTheta: Double
  public let maxPositionEmbeddings: Int

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

  public var layerTypes: [OrnithLayerType] {
    (0..<layerCount).map { index in
      (index + 1).isMultiple(of: fullAttentionInterval)
        ? .fullAttention
        : .linearAttention
    }
  }

  public var fullAttentionLayerCount: Int {
    layerTypes.lazy.filter { $0 == .fullAttention }.count
  }

  public var linearAttentionLayerCount: Int {
    layerTypes.lazy.filter { $0 == .linearAttention }.count
  }

  public func isLinearLayer(_ index: Int) -> Bool {
    precondition((0..<layerCount).contains(index), "layer index out of range")
    return layerTypes[index] == .linearAttention
  }

  public var routedExpertCount: Int {
    layerCount * expertCount
  }

  public var routedExpertFetchesPerToken: Int {
    layerCount * expertsPerToken
  }

  /// K and V scalar count added for one token across all full-attention layers.
  public var kvElementsPerToken: Int64 {
    Int64(fullAttentionLayerCount)
      * Int64(keyValueHeads)
      * Int64(headDimension)
      * 2
  }

  /// Architectural lower bound when K/V are stored as FP16.
  public var kvBytesPerTokenFP16: Int64 {
    kvElementsPerToken * 2
  }

  /// Current Swiftlet decode-state cost: K/V are Swift `Float` arrays.
  public var kvBytesPerTokenFP32: Int64 {
    kvElementsPerToken * 4
  }

  /// Persistent GatedDelta recurrence matrices, one FP32 matrix per linear layer.
  public var deltaNetRecurrenceBytesFP32: Int64 {
    Int64(linearAttentionLayerCount)
      * Int64(linearValueHeads)
      * Int64(linearKeyHeadDimension)
      * Int64(linearValueHeadDimension)
      * 4
  }

  /// Causal convolution history retained by every GatedDelta layer.
  public var deltaNetConvTailBytesFP32: Int64 {
    let keyDimension = Int64(linearKeyHeads * linearKeyHeadDimension)
    let valueDimension = Int64(linearValueHeads * linearValueHeadDimension)
    let convolutionDimension = 2 * keyDimension + valueDimension
    let tailRows = Int64(max(0, convolutionKernelSize - 1))
    return Int64(linearAttentionLayerCount) * tailRows * convolutionDimension * 4
  }

  public var deltaNetStateBytesFP32: Int64 {
    deltaNetRecurrenceBytesFP32 + deltaNetConvTailBytesFP32
  }

  public static let v1_5_35BA3B = OrnithProfile(
    name: "Ornith-1.5-35B-A3B",
    sourceCheckpoint: "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit",
    modelType: "qwen3_5_moe",
    architecture: "Qwen3_5MoeForConditionalGeneration",
    hiddenSize: 2_048,
    layerCount: 40,
    fullAttentionInterval: 4,
    vocabSize: 248_320,
    rmsNormEpsilon: 1e-6,
    attentionHeads: 16,
    keyValueHeads: 2,
    headDimension: 256,
    partialRotaryFactor: 0.25,
    ropeTheta: 10_000_000,
    maxPositionEmbeddings: 262_144,
    linearValueHeads: 32,
    linearKeyHeads: 16,
    linearKeyHeadDimension: 128,
    linearValueHeadDimension: 128,
    convolutionKernelSize: 4,
    expertCount: 256,
    expertsPerToken: 8,
    expertIntermediateSize: 512,
    sharedExpertIntermediateSize: 512,
    normalizeTopK: true,
    mtpHiddenLayerCount: 1
  )
}
