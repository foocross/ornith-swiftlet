import Foundation

public struct ExpertCacheMemoryPlan: Sendable, Equatable {
  public let targetBytes: Int64
  public let denseCoreBytes: Int64
  public let deltaRecurrenceBytes: Int64
  public let deltaConvTailBytes: Int64
  public let kvBytesPerToken: Int64
  public let kvCacheBytes: Int64
  public let scratchBytes: Int64
  public let safetyBytes: Int64
  public let expertCacheBytes: Int64
  public let expertStrideBytes: Int64
  public let expertSlots: Int
  public let totalExpertSlots: Int
  public let totalPlannedBytes: Int64

  public var deltaStateBytes: Int64 {
    deltaRecurrenceBytes + deltaConvTailBytes
  }

  public var cacheBudgetGiB: Double {
    Double(expertCacheBytes) / Double(1 << 30)
  }

  public var expertCoverage: Double {
    guard totalExpertSlots > 0 else { return 0 }
    return Double(expertSlots) / Double(totalExpertSlots)
  }
}

public enum ExpertCacheMemoryPlanError: Error, Sendable, Equatable, CustomStringConvertible {
  case invalidInput(String)
  case insufficientBudget(requiredBytes: Int64, targetBytes: Int64)

  public var description: String {
    switch self {
    case .invalidInput(let message):
      return message
    case .insufficientBudget(let required, let target):
      return "target memory \(target) B is below the minimum \(required) B"
    }
  }
}

extension ArchConfig {
  /// Current Swiftlet uses `[Float]` for each session's K/V cache. Keep this
  /// separate from `kvBytesPerToken`, which models a future FP16 cache.
  public var swiftletDecodeKVBytesPerToken: Int {
    fullAttentionLayerCount * kvHeads * headDim * 2 * MemoryLayout<Float>.stride
  }

  /// Fast-path GatedDelta convolution history uses one Float buffer per linear
  /// layer with `(kernel - 1) * convDim` elements.
  public var deltaNetConvTailBytes: Int {
    let keyDimension = linearKHeads * linearKHeadDim
    let valueDimension = linearVHeads * linearVHeadDim
    let convolutionDimension = 2 * keyDimension + valueDimension
    return linearLayerCount * max(0, convKernelSize - 1) * convolutionDimension
      * MemoryLayout<Float>.stride
  }
}

/// SlotStream-style hard-budget planner for Swiftlet's existing global cache.
/// It reserves dense weights, the actual current FP32 session state, expected
/// KV, scratch, and a safety margin before converting the remainder into whole
/// expert slots.
public enum ExpertCacheMemoryGovernor {
  public static func plan(
    architecture: ArchConfig,
    targetMemoryGiB: Double,
    contextTokens: Int,
    denseCoreBytes: Int64,
    expertStrideBytes: Int64,
    scratchBytes: Int64 = 256 * 1_048_576,
    fixedSafetyBytes: Int64 = 256 * 1_048_576,
    safetyFraction: Double = 0.08,
    minimumSlots: Int = 16
  ) throws -> ExpertCacheMemoryPlan {
    guard targetMemoryGiB.isFinite, targetMemoryGiB > 0 else {
      throw ExpertCacheMemoryPlanError.invalidInput("targetMemoryGiB must be positive")
    }
    guard contextTokens >= 0 else {
      throw ExpertCacheMemoryPlanError.invalidInput("contextTokens cannot be negative")
    }
    guard denseCoreBytes >= 0, expertStrideBytes > 0, scratchBytes >= 0,
      fixedSafetyBytes >= 0, minimumSlots >= 16,
      (0...0.5).contains(safetyFraction)
    else {
      throw ExpertCacheMemoryPlanError.invalidInput("invalid memory-planner input")
    }

    let targetBytes = Int64((targetMemoryGiB * Double(1 << 30)).rounded(.down))
    let deltaRecurrenceBytes = Int64(architecture.deltaNetStateBytes)
    let deltaConvTailBytes = Int64(architecture.deltaNetConvTailBytes)
    let kvBytesPerToken = Int64(architecture.swiftletDecodeKVBytesPerToken)
    let kvCacheBytes = kvBytesPerToken * Int64(contextTokens)
    let percentageSafety = Int64((Double(targetBytes) * safetyFraction).rounded(.up))
    let safetyBytes = max(fixedSafetyBytes, percentageSafety)

    let residentWithoutExperts =
      denseCoreBytes
      + deltaRecurrenceBytes
      + deltaConvTailBytes
      + kvCacheBytes
      + scratchBytes
      + safetyBytes
    let minimumRequired = residentWithoutExperts + Int64(minimumSlots) * expertStrideBytes
    guard targetBytes >= minimumRequired else {
      throw ExpertCacheMemoryPlanError.insufficientBudget(
        requiredBytes: minimumRequired,
        targetBytes: targetBytes
      )
    }

    let totalExpertSlots = architecture.routedExpertTotal
    let available = targetBytes - residentWithoutExperts
    let slots = min(Int(available / expertStrideBytes), totalExpertSlots)
    let expertCacheBytes = Int64(slots) * expertStrideBytes

    return ExpertCacheMemoryPlan(
      targetBytes: targetBytes,
      denseCoreBytes: denseCoreBytes,
      deltaRecurrenceBytes: deltaRecurrenceBytes,
      deltaConvTailBytes: deltaConvTailBytes,
      kvBytesPerToken: kvBytesPerToken,
      kvCacheBytes: kvCacheBytes,
      scratchBytes: scratchBytes,
      safetyBytes: safetyBytes,
      expertCacheBytes: expertCacheBytes,
      expertStrideBytes: expertStrideBytes,
      expertSlots: slots,
      totalExpertSlots: totalExpertSlots,
      totalPlannedBytes: residentWithoutExperts + expertCacheBytes
    )
  }
}
