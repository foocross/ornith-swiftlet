import XCTest

@testable import OrnithSwiftletPort

final class TopKRouterTests: XCTestCase {
  func testSelectsTopKAndRenormalizesSelectedMass() throws {
    let routes = try TopKRouter.route(
      logits: Array(0..<10).map(Double.init),
      k: 3,
      normalizeSelected: true
    )

    XCTAssertEqual(routes.map(\.expert), [9, 8, 7])
    XCTAssertEqual(routes.reduce(0) { $0 + $1.probability }, 1, accuracy: 1e-12)
    XCTAssertGreaterThan(routes[0].probability, routes[1].probability)
  }

  func testTiesUseLowerExpertIndexDeterministically() throws {
    let routes = try TopKRouter.route(logits: [1, 1, 1, 0], k: 2)
    XCTAssertEqual(routes.map(\.expert), [0, 1])
  }

  func testUnnormalizedModePreservesRawSoftmaxMass() throws {
    let routes = try TopKRouter.route(
      logits: [2, 1, 0, -1],
      k: 2,
      normalizeSelected: false
    )
    let selectedMass = routes.reduce(0) { $0 + $1.probability }
    XCTAssertLessThan(selectedMass, 1)
    XCTAssertGreaterThan(selectedMass, 0)
  }

  func testInvalidRouterInputsFail() {
    XCTAssertThrowsError(try TopKRouter.route(logits: [], k: 1))
    XCTAssertThrowsError(try TopKRouter.route(logits: [1, 2], k: 0))
    XCTAssertThrowsError(try TopKRouter.route(logits: [1, .infinity], k: 1))
  }
}
