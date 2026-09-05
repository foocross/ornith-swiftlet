import XCTest

@testable import OrnithSwiftletPort

final class ExpertBlobLayoutTests: XCTestCase {
  func testOrnithInt4LayoutMatchesSwiftletQpackMath() {
    let layout = ExpertBlobLayout.ornithInt4

    XCTAssertEqual(layout.sections.map(\.name), ["gate_proj", "up_proj", "down_proj"])
    XCTAssertEqual(layout.sections.map(\.byteCount), [589_824, 589_824, 589_824])
    XCTAssertEqual(layout.payloadBytes, 1_769_472)
    XCTAssertEqual(layout.strideBytes, 1_769_472)
    XCTAssertEqual(layout.strideBytes % 16_384, 0)
  }

  func testPoolAndColdReadSizes() {
    let profile = OrnithProfile.v1_5_35BA3B
    let layout = ExpertBlobLayout.ornithInt4

    let pool = layout.totalPoolBytes(
      layerCount: profile.layerCount,
      expertCount: profile.expertCount
    )
    let coldRead = layout.coldReadBytesPerToken(
      layerCount: profile.layerCount,
      expertsPerToken: profile.expertsPerToken
    )

    XCTAssertEqual(pool, 18_119_393_280)
    XCTAssertEqual(Double(pool) / Double(BinarySize.gibibyte), 16.875, accuracy: 0.000_001)
    XCTAssertEqual(coldRead, 566_231_040)
    XCTAssertEqual(Double(coldRead) / Double(BinarySize.mebibyte), 540.0, accuracy: 0.000_001)
  }

  func testProjectionOffsetsAreContiguous() {
    let sections = ExpertBlobLayout.ornithInt4.sections
    XCTAssertEqual(sections[0].offset, 0)
    XCTAssertEqual(sections[1].offset, sections[0].byteCount)
    XCTAssertEqual(sections[2].offset, sections[0].byteCount + sections[1].byteCount)
  }
}
