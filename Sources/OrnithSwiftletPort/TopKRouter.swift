import Foundation

public struct ExpertRoute: Sendable, Equatable {
  public let expert: Int
  public let probability: Double
}

public enum TopKRouterError: Error, Equatable, CustomStringConvertible {
  case emptyLogits
  case invalidK(k: Int, count: Int)
  case nonFiniteLogit(index: Int)

  public var description: String {
    switch self {
    case .emptyLogits:
      return "router logits cannot be empty"
    case .invalidK(let k, let count):
      return "top-k value \(k) is invalid for \(count) experts"
    case .nonFiniteLogit(let index):
      return "router logit at index \(index) is not finite"
    }
  }
}

public enum TopKRouter {
  /// Stable softmax, deterministic top-k, and optional selected-mass
  /// renormalization matching Qwen3.5's norm_topk_prob behavior.
  public static func route(
    logits: [Double],
    k: Int,
    normalizeSelected: Bool = true
  ) throws -> [ExpertRoute] {
    guard !logits.isEmpty else { throw TopKRouterError.emptyLogits }
    guard k > 0 && k <= logits.count else {
      throw TopKRouterError.invalidK(k: k, count: logits.count)
    }
    for (index, value) in logits.enumerated() where !value.isFinite {
      throw TopKRouterError.nonFiniteLogit(index: index)
    }

    let maximum = logits.max()!
    let exponentials = logits.map { Foundation.exp($0 - maximum) }
    let denominator = exponentials.reduce(0, +)
    let probabilities = exponentials.map { $0 / denominator }

    let selectedIndices = probabilities.indices.sorted { lhs, rhs in
      if probabilities[lhs] == probabilities[rhs] {
        return lhs < rhs
      }
      return probabilities[lhs] > probabilities[rhs]
    }.prefix(k)

    let selectedMass = selectedIndices.reduce(0.0) { result, index in
      result + probabilities[index]
    }

    return selectedIndices.map { index in
      let probability =
        normalizeSelected
        ? probabilities[index] / selectedMass
        : probabilities[index]
      return ExpertRoute(expert: index, probability: probability)
    }
  }
}
