import XCTest

@testable import OrnithSwiftletPort

final class ExpertCachePolicyTests: XCTestCase {
  private let a = ExpertKey(layer: 0, expert: 1)
  private let b = ExpertKey(layer: 0, expert: 2)
  private let c = ExpertKey(layer: 1, expert: 3)
  private let d = ExpertKey(layer: 1, expert: 4)

  func testLFUEvictsLessFrequentlyUsedExpert() throws {
    var cache = try LFURecencyExpertCache(capacity: 2)
    _ = try cache.accessBatch([a, b])
    _ = try cache.accessBatch([a])
    let access = try XCTUnwrap(try cache.accessBatch([c]).first)

    XCTAssertFalse(access.hit)
    XCTAssertEqual(access.evicted, b)
    XCTAssertEqual(cache.residentKeys, Set([a, c]))
    XCTAssertEqual(cache.hitCount, 1)
    XCTAssertEqual(cache.missCount, 3)
  }

  func testRecencyBreaksFrequencyTie() throws {
    var cache = try LFURecencyExpertCache(capacity: 2)
    _ = try cache.accessBatch([a])
    _ = try cache.accessBatch([b])
    let access = try XCTUnwrap(try cache.accessBatch([c]).first)

    XCTAssertEqual(access.evicted, a)
    XCTAssertEqual(cache.residentKeys, Set([b, c]))
  }

  func testCurrentBatchMembersAreProtectedFromEviction() throws {
    var cache = try LFURecencyExpertCache(capacity: 2)
    _ = try cache.accessBatch([a, b])
    let accesses = try cache.accessBatch([c, d])

    XCTAssertEqual(accesses.count, 2)
    XCTAssertEqual(cache.residentKeys, Set([c, d]))
    XCTAssertNotEqual(accesses[0].slot, accesses[1].slot)
  }

  func testDuplicateExpertInBatchReusesSlot() throws {
    var cache = try LFURecencyExpertCache(capacity: 2)
    let accesses = try cache.accessBatch([a, a])

    XCTAssertFalse(accesses[0].hit)
    XCTAssertTrue(accesses[1].hit)
    XCTAssertEqual(accesses[0].slot, accesses[1].slot)
    XCTAssertEqual(cache.hitCount, 1)
    XCTAssertEqual(cache.missCount, 1)
  }

  func testBatchLargerThanCacheFailsBeforeMutation() throws {
    var cache = try LFURecencyExpertCache(capacity: 2)

    XCTAssertThrowsError(try cache.accessBatch([a, b, c])) { error in
      XCTAssertEqual(
        error as? ExpertCachePolicyError,
        .batchExceedsCapacity(uniqueExperts: 3, capacity: 2)
      )
    }
    XCTAssertTrue(cache.residentKeys.isEmpty)
  }
}
