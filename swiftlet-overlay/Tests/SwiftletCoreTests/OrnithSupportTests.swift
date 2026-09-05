import Foundation
import Testing

@testable import SwiftletCore

@Suite struct OrnithSupportTests {
  static func configURL(numExperts: Int = 256) throws -> URL {
    let linear = [String](repeating: "linear_attention", count: 3)
    let pattern = Array(repeating: linear + ["full_attention"], count: 10).flatMap { $0 }
    let text: [String: Any] = [
      "model_type": "qwen3_5_moe_text",
      "hidden_size": 2_048,
      "num_hidden_layers": 40,
      "num_attention_heads": 16,
      "num_key_value_heads": 2,
      "head_dim": 256,
      "full_attention_interval": 4,
      "layer_types": pattern,
      "linear_num_value_heads": 32,
      "linear_num_key_heads": 16,
      "linear_key_head_dim": 128,
      "linear_value_head_dim": 128,
      "linear_conv_kernel_dim": 4,
      "num_experts": numExperts,
      "num_experts_per_tok": 8,
      "moe_intermediate_size": 512,
      "shared_expert_intermediate_size": 512,
      "mtp_num_hidden_layers": 1,
      "vocab_size": 248_320,
      "rms_norm_eps": 1e-6,
      "tie_word_embeddings": false,
      "rope_parameters": [
        "rope_theta": 10_000_000,
        "partial_rotary_factor": 0.25,
      ],
    ]
    let top: [String: Any] = [
      "model_type": "qwen3_5_moe",
      "architectures": ["Qwen3_5MoeForConditionalGeneration"],
      "text_config": text,
      "tie_word_embeddings": false,
    ]
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent("ornith-config-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = directory.appendingPathComponent("config.json")
    try JSONSerialization.data(withJSONObject: top).write(to: url)
    return url
  }

  static func qpack(
    expertCount: Int = 256,
    quantBits: Int? = 4,
    omitLayer: Int? = nil
  ) -> (Qpack.Layout, Qpack.Manifest) {
    let sections = [
      Qpack.Section(
        name: "gate_proj.weight", dtype: "U32", shape: [512, 256], offset: 0, size: 524_288),
      Qpack.Section(
        name: "gate_proj.scales", dtype: "BF16", shape: [512, 32], offset: 524_288, size: 32_768),
      Qpack.Section(
        name: "gate_proj.biases", dtype: "BF16", shape: [512, 32], offset: 557_056, size: 32_768),
      Qpack.Section(
        name: "up_proj.weight", dtype: "U32", shape: [512, 256], offset: 589_824, size: 524_288),
      Qpack.Section(
        name: "up_proj.scales", dtype: "BF16", shape: [512, 32], offset: 1_114_112, size: 32_768),
      Qpack.Section(
        name: "up_proj.biases", dtype: "BF16", shape: [512, 32], offset: 1_146_880, size: 32_768),
      Qpack.Section(
        name: "down_proj.weight", dtype: "U32", shape: [2_048, 64], offset: 1_179_648, size: 524_288
      ),
      Qpack.Section(
        name: "down_proj.scales", dtype: "BF16", shape: [2_048, 8], offset: 1_703_936, size: 32_768),
      Qpack.Section(
        name: "down_proj.biases", dtype: "BF16", shape: [2_048, 8], offset: 1_736_704, size: 32_768),
    ]
    let architecture = ArchConfig.ornith1_5_35B
    let layout = Qpack.Layout(
      expertCount: expertCount,
      layerCount: architecture.layerCount,
      expertStride: architecture.expertBlobBytesInt4G64,
      sections: sections,
      linearLayers: (0..<architecture.layerCount).map(architecture.isLinearLayer)
    )
    var files = ["model.safetensors": 1_350 * 1_048_576]
    let layerBytes = architecture.expertCount * architecture.expertBlobBytesInt4G64
    for layer in 0..<architecture.layerCount where layer != omitLayer {
      files[String(format: "packed_experts/layer_%02d.bin", layer)] = layerBytes
    }
    let manifest = Qpack.Manifest(
      magic: "QPACK",
      version: Qpack.manifestVersion,
      modelName: "qwen3_5_moe_text",
      sourceCheckpoint: "Ornith-1.5-35B-A3B-MLX-4bit",
      quantBits: quantBits,
      quantGroupSize: 64,
      files: files
    )
    return (layout, manifest)
  }

  static func writeSparseFile(_ url: URL, size: Int) throws {
    _ = FileManager.default.createFile(atPath: url.path, contents: Data())
    let handle = try FileHandle(forWritingTo: url)
    try handle.truncate(atOffset: UInt64(size))
    try handle.close()
  }

  static func qpackDirectory(corrupting relativePathToCorrupt: String? = nil) throws -> URL {
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent("ornith-qpack-\(UUID().uuidString)")
    let packed = directory.appendingPathComponent("packed_experts")
    try FileManager.default.createDirectory(at: packed, withIntermediateDirectories: true)

    let sourceConfig = try configURL()
    defer {
      try? FileManager.default.removeItem(at: sourceConfig.deletingLastPathComponent())
    }
    try Data(contentsOf: sourceConfig).write(to: directory.appendingPathComponent("config.json"))

    let (layout, manifest) = qpack()
    try JSONEncoder().encode(layout).write(to: packed.appendingPathComponent("layout.json"))
    try JSONEncoder().encode(manifest).write(to: directory.appendingPathComponent("manifest.json"))

    for (relativePath, declaredSize) in manifest.files {
      let url = directory.appendingPathComponent(relativePath)
      try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(),
        withIntermediateDirectories: true
      )
      let actualSize = relativePath == relativePathToCorrupt ? declaredSize - 1 : declaredSize
      try writeSparseFile(url, size: actualSize)
    }
    return directory
  }

  @Test func explicitProfileMatchesOrnithGeometry() {
    let profile = ArchConfig.ornith1_5_35B
    #expect(profile.family == .qwen3_5Moe)
    #expect(profile.linearLayerCount == 30)
    #expect(profile.fullAttentionLayerCount == 10)
    #expect(profile.expertCount == 256)
    #expect(profile.expertTopK == 8)
    #expect(profile.expertBlobBytesInt4G64 == 1_769_472)
    #expect(profile.kvBytesPerToken == 20_480)
    #expect(profile.swiftletDecodeKVBytesPerToken == 40_960)
    #expect(profile.deltaNetStateBytes == 62_914_560)
    #expect(profile.deltaNetConvTailBytes == 2_949_120)
  }

  @Test func officialConfigIsAccepted() throws {
    let config = try QwenConfig(url: Self.configURL())
    let validation = config.validateAsOrnith1_5_35B()
    #expect(
      validation.isCompatible, Comment(rawValue: validation.mismatches.joined(separator: "\n")))
    #expect(config.normTopkProb)
    #expect(config.weightPrefix == "language_model.")
  }

  @Test func shapeDriftIsRejected() throws {
    let config = try QwenConfig(url: Self.configURL(numExperts: 128))
    let validation = config.validateAsOrnith1_5_35B()
    #expect(!validation.isCompatible)
    #expect(validation.mismatches.contains { $0.contains("num_experts") })
  }

  @Test func validQpackLayoutIsAccepted() {
    let (layout, manifest) = Self.qpack()
    let validation = layout.validateAsOrnith1_5_35B(manifest: manifest)
    #expect(
      validation.isCompatible, Comment(rawValue: validation.mismatches.joined(separator: "\n")))
  }

  @Test func qpackQuantizationDriftIsRejected() {
    let (layout, manifest) = Self.qpack(quantBits: 8)
    let validation = layout.validateAsOrnith1_5_35B(manifest: manifest)
    #expect(!validation.isCompatible)
    #expect(validation.mismatches.contains { $0.contains("quantBits") })
  }

  @Test func incompleteQpackManifestIsRejected() {
    let (layout, manifest) = Self.qpack(omitLayer: 39)
    let validation = layout.validateAsOrnith1_5_35B(manifest: manifest)
    #expect(!validation.isCompatible)
    #expect(validation.mismatches.contains { $0.contains("layer_39.bin") })
  }

  @Test func memoryPlanRespectsTarget() throws {
    let plan = try ExpertCacheMemoryGovernor.plan(
      architecture: .ornith1_5_35B,
      targetMemoryGiB: 4,
      contextTokens: 8_192,
      denseCoreBytes: 1_350 * 1_048_576,
      expertStrideBytes: 1_769_472
    )
    #expect(plan.totalPlannedBytes <= plan.targetBytes)
    #expect(plan.expertSlots >= 16)
    #expect(plan.expertCacheBytes == Int64(plan.expertSlots) * 1_769_472)
    #expect(plan.kvBytesPerToken == 40_960)
    #expect(plan.deltaRecurrenceBytes == 62_914_560)
    #expect(plan.deltaConvTailBytes == 2_949_120)
  }

  @Test func plannerRejectsCacheFloorBelowSwiftletMinimum() {
    #expect(throws: ExpertCacheMemoryPlanError.self) {
      try ExpertCacheMemoryGovernor.plan(
        architecture: .ornith1_5_35B,
        targetMemoryGiB: 4,
        contextTokens: 8_192,
        denseCoreBytes: 1_350 * 1_048_576,
        expertStrideBytes: 1_769_472,
        minimumSlots: 8
      )
    }
  }

  @Test func nonFiniteMemoryTargetIsRejected() {
    #expect(throws: ExpertCacheMemoryPlanError.self) {
      try ExpertCacheMemoryGovernor.plan(
        architecture: .ornith1_5_35B,
        targetMemoryGiB: .infinity,
        contextTokens: 8_192,
        denseCoreBytes: 1_350 * 1_048_576,
        expertStrideBytes: 1_769_472
      )
    }
  }

  @Test func runtimePreflightValidatesPayloadsAndPlansCurrentState() throws {
    let directory = try Self.qpackDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }

    let plan = try OrnithRuntimeFactory.preflight(
      modelDir: directory,
      options: OrnithRuntimeOptions(targetMemoryGiB: 4, plannedContextTokens: 8_192)
    )

    #expect(plan.denseCoreBytes == 1_350 * 1_048_576)
    #expect(plan.expertStrideBytes == 1_769_472)
    #expect(plan.kvBytesPerToken == 40_960)
    #expect(plan.totalPlannedBytes <= plan.targetBytes)
  }

  @Test func runtimePreflightRejectsWrongOnDiskPayloadSize() throws {
    let badPath = "packed_experts/layer_07.bin"
    let directory = try Self.qpackDirectory(corrupting: badPath)
    defer { try? FileManager.default.removeItem(at: directory) }

    #expect(throws: OrnithRuntimeFactoryError.self) {
      try OrnithRuntimeFactory.preflight(
        modelDir: directory,
        options: OrnithRuntimeOptions(targetMemoryGiB: 4, plannedContextTokens: 8_192)
      )
    }
  }

  @Test func longerContextTradesCacheForKV() throws {
    let short = try ExpertCacheMemoryGovernor.plan(
      architecture: .ornith1_5_35B,
      targetMemoryGiB: 4,
      contextTokens: 4_096,
      denseCoreBytes: 1_350 * 1_048_576,
      expertStrideBytes: 1_769_472
    )
    let long = try ExpertCacheMemoryGovernor.plan(
      architecture: .ornith1_5_35B,
      targetMemoryGiB: 4,
      contextTokens: 16_384,
      denseCoreBytes: 1_350 * 1_048_576,
      expertStrideBytes: 1_769_472
    )
    #expect(short.expertSlots > long.expertSlots)
  }
}
