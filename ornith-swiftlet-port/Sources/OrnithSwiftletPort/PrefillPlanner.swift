import Foundation

public struct PrefillPlan: Sendable, Equatable {
  public let promptTokens: Int
  public let chunkTokens: Int
  public let scratchBytesPerToken: Int64
  public let scratchBudgetBytes: Int64
  public let estimatedChunkScratchBytes: Int64
  public let chunkCount: Int
}

public enum PrefillPlannerError: Error, Equatable, CustomStringConvertible {
  case noViableChunk(minimumTokens: Int, scratchBudgetBytes: Int64)

  public var description: String {
    switch self {
    case .noViableChunk(let tokens, let bytes):
      return "cannot fit the minimum \(tokens)-token prefill chunk in \(BinarySize.format(bytes))"
    }
  }
}

/// Chooses the largest power-of-two-like prefill chunk that fits a hard scratch
/// budget. Decode remains token-at-a-time; prefill gets a separate plan.
public enum PrefillPlanner {
  public static let defaultCandidates = [32, 64, 128, 256, 512, 1_024, 2_048]

  public static func plan(
    promptTokens: Int,
    scratchBudgetBytes: Int64,
    scratchBytesPerToken: Int64,
    candidates: [Int] = defaultCandidates
  ) throws -> PrefillPlan {
    precondition(promptTokens >= 0)
    precondition(scratchBudgetBytes >= 0)
    precondition(scratchBytesPerToken > 0)

    if promptTokens == 0 {
      return PrefillPlan(
        promptTokens: 0,
        chunkTokens: 0,
        scratchBytesPerToken: scratchBytesPerToken,
        scratchBudgetBytes: scratchBudgetBytes,
        estimatedChunkScratchBytes: 0,
        chunkCount: 0
      )
    }

    let sorted = Array(Set(candidates.filter { $0 > 0 })).sorted()
    guard let minimum = sorted.first else {
      throw PrefillPlannerError.noViableChunk(
        minimumTokens: 1,
        scratchBudgetBytes: scratchBudgetBytes
      )
    }

    let maxByBudget = Int(scratchBudgetBytes / scratchBytesPerToken)
    let cappedByPrompt = max(1, min(promptTokens, maxByBudget))
    let chunk =
      sorted.last(where: { $0 <= cappedByPrompt })
      ?? (promptTokens < minimum && promptTokens <= maxByBudget ? promptTokens : nil)

    guard let chunk, chunk > 0 else {
      throw PrefillPlannerError.noViableChunk(
        minimumTokens: min(minimum, promptTokens),
        scratchBudgetBytes: scratchBudgetBytes
      )
    }

    let count = (promptTokens + chunk - 1) / chunk
    return PrefillPlan(
      promptTokens: promptTokens,
      chunkTokens: chunk,
      scratchBytesPerToken: scratchBytesPerToken,
      scratchBudgetBytes: scratchBudgetBytes,
      estimatedChunkScratchBytes: Int64(chunk) * scratchBytesPerToken,
      chunkCount: count
    )
  }
}
