import XCTest

@testable import OrnithSwiftletPort

final class OrnithConfigTests: XCTestCase {
  private func fixtureURL() throws -> URL {
    try XCTUnwrap(
      Bundle.module.url(
        forResource: "ornith-1.5-35b-config",
        withExtension: "json",
        subdirectory: "Fixtures"
      )
    )
  }

  func testOfficialGeometryParsesAsCompatible() throws {
    let config = try OrnithCheckpointConfig(url: fixtureURL())
    let report = OrnithCompatibility.inspect(config)

    XCTAssertTrue(report.isCompatible, report.issues.map(\.description).joined(separator: "\n"))
    XCTAssertTrue(report.issues.isEmpty)
    XCTAssertEqual(config.architecture, "Qwen3_5MoeForConditionalGeneration")
    XCTAssertEqual(config.modelType, "qwen3_5_moe")
    XCTAssertEqual(config.hiddenSize, 2_048)
    XCTAssertEqual(config.layerCount, 40)
    XCTAssertEqual(config.expertCount, 256)
    XCTAssertEqual(config.expertsPerToken, 8)
    XCTAssertEqual(config.weightPrefix, "language_model.")
    XCTAssertEqual(config.deltaProjectionLayout, .split)
    XCTAssertTrue(config.normalizeTopK, "Qwen3.5 must default norm_topk_prob to true")
    XCTAssertEqual(config.quantization, .mlxInt4Group64)
  }

  func testLayerPatternHasThirtyLinearAndTenFullAttentionLayers() throws {
    let config = try OrnithCheckpointConfig(url: fixtureURL())

    XCTAssertEqual(config.layerTypes.filter { $0 == .linearAttention }.count, 30)
    XCTAssertEqual(config.layerTypes.filter { $0 == .fullAttention }.count, 10)
    XCTAssertTrue(config.isLinearLayer(0))
    XCTAssertFalse(config.isLinearLayer(3))
    XCTAssertFalse(config.isLinearLayer(39))
  }

  func testMissingLayerTypesAreDerivedFromInterval() throws {
    let data = try Data(contentsOf: fixtureURL())
    var json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    var text = try XCTUnwrap(json["text_config"] as? [String: Any])
    text.removeValue(forKey: "layer_types")
    json["text_config"] = text

    let modified = try JSONSerialization.data(withJSONObject: json)
    let config = try OrnithCheckpointConfig(data: modified)

    XCTAssertEqual(config.layerTypes, OrnithProfile.v1_5_35BA3B.layerTypes)
  }

  func testOfficialQuantizationConfigKeyParsesDefaultExpertQuantization() throws {
    let data = try Data(contentsOf: fixtureURL())
    var json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    let quantization = try XCTUnwrap(json.removeValue(forKey: "quantization"))
    json["quantization_config"] = quantization

    let modified = try JSONSerialization.data(withJSONObject: json)
    let config = try OrnithCheckpointConfig(data: modified)

    XCTAssertEqual(config.quantization, .mlxInt4Group64)
  }

  func testMalformedQuantizationThrowsParseErrorInsteadOfTrapping() throws {
    let data = try Data(contentsOf: fixtureURL())
    var json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    json["quantization"] = ["bits": 0, "group_size": 64]

    let modified = try JSONSerialization.data(withJSONObject: json)
    XCTAssertThrowsError(try OrnithCheckpointConfig(data: modified)) { error in
      XCTAssertEqual(
        error as? OrnithCheckpointConfig.ParseError,
        .invalidField("quantization.bits")
      )
    }
  }

  func testShapeDriftIsRejected() throws {
    let data = try Data(contentsOf: fixtureURL())
    var json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    var text = try XCTUnwrap(json["text_config"] as? [String: Any])
    text["num_experts"] = 128
    json["text_config"] = text

    let modified = try JSONSerialization.data(withJSONObject: json)
    let report = OrnithCompatibility.inspect(try OrnithCheckpointConfig(data: modified))

    XCTAssertFalse(report.isCompatible)
    XCTAssertTrue(report.errors.contains { $0.field == "num_experts" })
  }
}
