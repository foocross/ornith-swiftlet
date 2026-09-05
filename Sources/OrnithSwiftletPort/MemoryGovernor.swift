import Foundation

public struct OrnithMemoryFacts: Sendable, Equatable {
  public let denseCoreBytes: Int64
  public let deltaNetRecurrenceBytes: Int64
  public let deltaNetConvTailBytes: Int64
  public let kvBytesPerToken: Int64
  public let expertStrideBytes: Int64
  public let totalExpertSlots: Int

  public init(
    denseCoreBytes: Int64,
    deltaNetRecurrenceBytes: Int64,
    deltaNetConvTailBytes: Int64,
    kvBytesPerToken: Int64,
    expertStrideBytes: Int64,
    totalExpertSlots: Int
  ) {
    precondition(denseCoreBytes >= 0)
    precondition(deltaNetRecurrenceBytes >= 0)
    precondition(deltaNetConvTailBytes >= 0)
    precondition(kvBytesPerToken >= 0)
    precondition(expertStrideBytes > 0)
    precondition(totalExpertSlots > 0)
    self.denseCoreBytes = denseCoreBytes
    self.deltaNetRecurrenceBytes = deltaNetRecurrenceBytes
    self.deltaNetConvTailBytes = deltaNetConvTailBytes
    self.kvBytesPerToken = kvBytesPerToken
    self.expertStrideBytes = expertStrideBytes
    self.totalExpertSlots = totalExpertSlots
  }

  public var deltaNetStateBytes: Int64 {
    deltaNetRecurrenceBytes + deltaNetConvTailBytes
  }

  /// The dense-core figure is intentionally a planning estimate. A production
  /// integration should replace it with the actual qpack manifest size plus
  /// measured persistent Metal allocations on the target machine.
  public static let ornithEstimated = OrnithMemoryFacts(
    denseCoreBytes: 1_350 * BinarySize.mebibyte,
    deltaNetRecurrenceBytes: OrnithProfile.v1_5_35BA3B.deltaNetRecurrenceBytesFP32,
    deltaNetConvTailBytes: OrnithProfile.v1_5_35BA3B.deltaNetConvTailBytesFP32,
    // Swiftlet currently stores session K/V in Swift Float arrays.
    kvBytesPerToken: OrnithProfile.v1_5_35BA3B.kvBytesPerTokenFP32,
    expertStrideBytes: ExpertBlobLayout.ornithInt4.strideBytes,
    totalExpertSlots: OrnithProfile.v1_5_35BA3B.routedExpertCount
  )
}

public struct OrnithMemoryRequest: Sendable, Equatable {
  public let targetResidentBytes: Int64
  public let contextTokens: Int
  public let scratchBytes: Int64
  public let fixedSafetyBytes: Int64
  public let safetyFraction: Double
  public let minimumExpertSlots: Int

  public init(
    targetResidentBytes: Int64,
    contextTokens: Int,
    scratchBytes: Int64 = 256 * BinarySize.mebibyte,
    fixedSafetyBytes: Int64 = 256 * BinarySize.mebibyte,
    safetyFraction: Double = 0.08,
    minimumExpertSlots: Int = 16
  ) {
    precondition(targetResidentBytes > 0)
    precondition(contextTokens >= 0)
    precondition(scratchBytes >= 0)
    precondition(fixedSafetyBytes >= 0)
    precondition((0...0.5).contains(safetyFraction))
    precondition(minimumExpertSlots >= 16, "current Swiftlet requires at least 16 slots")
    self.targetResidentBytes = targetResidentBytes
    self.contextTokens = contextTokens
    self.scratchBytes = scratchBytes
    self.fixedSafetyBytes = fixedSafetyBytes
    self.safetyFraction = safetyFraction
    self.minimumExpertSlots = minimumExpertSlots
  }
}

public struct OrnithMemoryPlan: Sendable, Equatable {
  public let targetResidentBytes: Int64
  public let denseCoreBytes: Int64
  public let deltaNetRecurrenceBytes: Int64
  public let deltaNetConvTailBytes: Int64
  public let kvBytesPerToken: Int64
  public let kvCacheBytes: Int64
  public let scratchBytes: Int64
  public let safetyBytes: Int64
  public let expertCacheBytes: Int64
  public let expertSlots: Int
  public let totalExpertSlots: Int
  public let totalPlannedBytes: Int64

  public var deltaNetStateBytes: Int64 {
    deltaNetRecurrenceBytes + deltaNetConvTailBytes
  }

  public var expertCoverageFraction: Double {
    guard totalExpertSlots > 0 else { return 0 }
    return Double(expertSlots) / Double(totalExpertSlots)
  }
}

public enum OrnithMemoryPlannerError: Error, Equatable, CustomStringConvertible {
  case insufficientBudget(requiredBytes: Int64, availableBytes: Int64)

  public var description: String {
    switch self {
    case .insufficientBudget(let required, let available):
      return
        "memory budget too small: need at least \(BinarySize.format(required)); got \(BinarySize.format(available))"
    }
  }
}

/// SlotStream-style hard-budget planner. It reserves resident state first and
/// gives every remaining aligned byte to the global expert cache.
public enum OrnithMemoryGovernor {
  public static func plan(
    facts: OrnithMemoryFacts = .ornithEstimated,
    request: OrnithMemoryRequest
  ) throws -> OrnithMemoryPlan {
    let kvCacheBytes = facts.kvBytesPerToken * Int64(request.contextTokens)
    let percentageSafety = Int64(
      (Double(request.targetResidentBytes) * request.safetyFraction).rounded(.up)
    )
    let safetyBytes = max(request.fixedSafetyBytes, percentageSafety)

    let residentWithoutExperts =
      facts.denseCoreBytes
      + facts.deltaNetRecurrenceBytes
      + facts.deltaNetConvTailBytes
      + kvCacheBytes
      + request.scratchBytes
      + safetyBytes
    let minimumExpertBytes = facts.expertStrideBytes * Int64(request.minimumExpertSlots)
    let minimumRequired = residentWithoutExperts + minimumExpertBytes

    guard request.targetResidentBytes >= minimumRequired else {
      throw OrnithMemoryPlannerError.insufficientBudget(
        requiredBytes: minimumRequired,
        availableBytes: request.targetResidentBytes
      )
    }

    let availableForExperts = request.targetResidentBytes - residentWithoutExperts
    let slots = min(
      Int(availableForExperts / facts.expertStrideBytes),
      facts.totalExpertSlots
    )
    let expertCacheBytes = Int64(slots) * facts.expertStrideBytes
    let total = residentWithoutExperts + expertCacheBytes

    return OrnithMemoryPlan(
      targetResidentBytes: request.targetResidentBytes,
      denseCoreBytes: facts.denseCoreBytes,
      deltaNetRecurrenceBytes: facts.deltaNetRecurrenceBytes,
      deltaNetConvTailBytes: facts.deltaNetConvTailBytes,
      kvBytesPerToken: facts.kvBytesPerToken,
      kvCacheBytes: kvCacheBytes,
      scratchBytes: request.scratchBytes,
      safetyBytes: safetyBytes,
      expertCacheBytes: expertCacheBytes,
      expertSlots: slots,
      totalExpertSlots: facts.totalExpertSlots,
      totalPlannedBytes: total
    )
  }
}
