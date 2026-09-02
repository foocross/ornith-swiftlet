import XCTest

@testable import OrnithSwiftletPort

final class PrefillPlannerTests: XCTestCase {
  func testPicksLargestCandidateWithinScratchBudget() throws {
    let plan = try PrefillPlanner.plan(
      promptTokens: 1_000,
      scratchBudgetBytes: 100 * BinarySize.mebibyte,
      scratchBytesPerToken: 256 * BinarySize.kibibyte
    )

    XCTAssertEqual(plan.chunkTokens, 256)
    XCTAssertEqual(plan.chunkCount, 4)
    XCTAssertLessThanOrEqual(plan.estimatedChunkScratchBytes, plan.scratchBudgetBytes)
  }

  func testSmallPromptUsesExactPromptSize() throws {
    let plan = try PrefillPlanner.plan(
      promptTokens: 20,
      scratchBudgetBytes: 10 * BinarySize.mebibyte,
      scratchBytesPerToken: 128 * BinarySize.kibibyte
    )

    XCTAssertEqual(plan.chunkTokens, 20)
    XCTAssertEqual(plan.chunkCount, 1)
  }

  func testZeroPromptNeedsNoScratch() throws {
    let plan = try PrefillPlanner.plan(
      promptTokens: 0,
      scratchBudgetBytes: 0,
      scratchBytesPerToken: 1
    )

    XCTAssertEqual(plan.chunkTokens, 0)
    XCTAssertEqual(plan.chunkCount, 0)
  }

  func testRejectsBudgetBelowMinimumChunk() {
    XCTAssertThrowsError(
      try PrefillPlanner.plan(
        promptTokens: 1_000,
        scratchBudgetBytes: 10 * BinarySize.mebibyte,
        scratchBytesPerToken: 2 * BinarySize.mebibyte
      ))
  }
}
