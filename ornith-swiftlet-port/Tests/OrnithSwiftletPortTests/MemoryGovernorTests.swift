import XCTest

@testable import OrnithSwiftletPort

final class MemoryGovernorTests: XCTestCase {
  func testOrnithStateMath() {
    let profile = OrnithProfile.v1_5_35BA3B

    XCTAssertEqual(profile.fullAttentionLayerCount, 10)
    XCTAssertEqual(profile.linearAttentionLayerCount, 30)
    XCTAssertEqual(profile.kvElementsPerToken, 10_240)
    XCTAssertEqual(profile.kvBytesPerTokenFP16, 20_480)
    XCTAssertEqual(profile.kvBytesPerTokenFP32, 40_960)
    XCTAssertEqual(profile.deltaNetRecurrenceBytesFP32, 62_914_560)
    XCTAssertEqual(profile.deltaNetConvTailBytesFP32, 2_949_120)
    XCTAssertEqual(profile.deltaNetStateBytesFP32, 65_863_680)
  }

  func testPlanStaysWithinHardBudgetAndUsesAlignedSlots() throws {
    let plan = try OrnithMemoryGovernor.plan(
      request: OrnithMemoryRequest(
        targetResidentBytes: BinarySize.gibibytes(4),
        contextTokens: 8_192
      ))

    XCTAssertLessThanOrEqual(plan.totalPlannedBytes, plan.targetResidentBytes)
    XCTAssertEqual(
      plan.expertCacheBytes,
      Int64(plan.expertSlots) * ExpertBlobLayout.ornithInt4.strideBytes
    )
    XCTAssertGreaterThanOrEqual(plan.expertSlots, 16)
    XCTAssertGreaterThan(plan.expertCoverageFraction, 0)
    XCTAssertLessThanOrEqual(plan.expertCoverageFraction, 1)
  }

  func testMoreMemoryProducesMoreExpertSlots() throws {
    let four = try OrnithMemoryGovernor.plan(
      request: OrnithMemoryRequest(
        targetResidentBytes: BinarySize.gibibytes(4),
        contextTokens: 8_192
      ))
    let six = try OrnithMemoryGovernor.plan(
      request: OrnithMemoryRequest(
        targetResidentBytes: BinarySize.gibibytes(6),
        contextTokens: 8_192
      ))

    XCTAssertGreaterThan(six.expertSlots, four.expertSlots)
  }

  func testLongerContextReducesExpertCacheBudget() throws {
    let short = try OrnithMemoryGovernor.plan(
      request: OrnithMemoryRequest(
        targetResidentBytes: BinarySize.gibibytes(4),
        contextTokens: 4_096
      ))
    let long = try OrnithMemoryGovernor.plan(
      request: OrnithMemoryRequest(
        targetResidentBytes: BinarySize.gibibytes(4),
        contextTokens: 16_384
      ))

    XCTAssertGreaterThan(short.expertSlots, long.expertSlots)
    XCTAssertEqual(
      long.kvCacheBytes - short.kvCacheBytes,
      Int64(16_384 - 4_096) * 40_960
    )
  }

  func testTooSmallBudgetFailsClearly() {
    XCTAssertThrowsError(
      try OrnithMemoryGovernor.plan(
        request: OrnithMemoryRequest(
          targetResidentBytes: BinarySize.gibibytes(1),
          contextTokens: 4_096
        ))
    ) { error in
      guard case OrnithMemoryPlannerError.insufficientBudget = error else {
        return XCTFail("unexpected error: \(error)")
      }
    }
  }
}
