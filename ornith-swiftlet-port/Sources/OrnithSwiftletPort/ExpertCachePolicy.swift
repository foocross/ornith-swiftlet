import Foundation

public struct ExpertKey: Hashable, Sendable, Comparable, CustomStringConvertible {
  public let layer: Int
  public let expert: Int

  public init(layer: Int, expert: Int) {
    precondition(layer >= 0 && expert >= 0)
    self.layer = layer
    self.expert = expert
  }

  public static func < (lhs: ExpertKey, rhs: ExpertKey) -> Bool {
    (lhs.layer, lhs.expert) < (rhs.layer, rhs.expert)
  }

  public var description: String {
    "L\(layer):E\(expert)"
  }
}

public struct ExpertCacheAccess: Sendable, Equatable {
  public let key: ExpertKey
  public let slot: Int
  public let hit: Bool
  public let evicted: ExpertKey?
}

public enum ExpertCachePolicyError: Error, Equatable, CustomStringConvertible {
  case invalidCapacity
  case batchExceedsCapacity(uniqueExperts: Int, capacity: Int)

  public var description: String {
    switch self {
    case .invalidCapacity:
      return "expert cache capacity must be positive"
    case .batchExceedsCapacity(let count, let capacity):
      return "batch needs \(count) simultaneous experts but cache has \(capacity) slots"
    }
  }
}

/// Pure, Metal-independent model of Swiftlet's global LFU cache with recency
/// tie-break. It exists so cache correctness can be tested without a GPU.
public struct LFURecencyExpertCache: Sendable {
  public let capacity: Int

  private var slotKeys: [ExpertKey?]
  private var keyToSlot: [ExpertKey: Int]
  private var frequency: [ExpertKey: Int]
  private var lastUse: [Int: UInt64]
  private var tick: UInt64

  public private(set) var hitCount: Int
  public private(set) var missCount: Int

  public init(capacity: Int) throws {
    guard capacity > 0 else { throw ExpertCachePolicyError.invalidCapacity }
    self.capacity = capacity
    slotKeys = Array(repeating: nil, count: capacity)
    keyToSlot = [:]
    frequency = [:]
    lastUse = [:]
    tick = 0
    hitCount = 0
    missCount = 0
  }

  public var residentKeys: Set<ExpertKey> {
    Set(slotKeys.compactMap { $0 })
  }

  public func slot(for key: ExpertKey) -> Int? {
    keyToSlot[key]
  }

  public mutating func accessBatch(_ keys: [ExpertKey]) throws -> [ExpertCacheAccess] {
    let uniqueCount = Set(keys).count
    guard uniqueCount <= capacity else {
      throw ExpertCachePolicyError.batchExceedsCapacity(
        uniqueExperts: uniqueCount,
        capacity: capacity
      )
    }

    tick &+= 1
    var protectedSlots = Set<Int>()
    var accesses: [ExpertCacheAccess] = []
    accesses.reserveCapacity(keys.count)

    for key in keys {
      frequency[key, default: 0] += 1

      if let slot = keyToSlot[key] {
        hitCount += 1
        lastUse[slot] = tick
        protectedSlots.insert(slot)
        accesses.append(
          ExpertCacheAccess(
            key: key,
            slot: slot,
            hit: true,
            evicted: nil
          ))
        continue
      }

      missCount += 1
      let slot = chooseSlot(excluding: protectedSlots)
      let evicted = slotKeys[slot]
      if let evicted {
        keyToSlot.removeValue(forKey: evicted)
      }

      slotKeys[slot] = key
      keyToSlot[key] = slot
      lastUse[slot] = tick
      protectedSlots.insert(slot)
      accesses.append(
        ExpertCacheAccess(
          key: key,
          slot: slot,
          hit: false,
          evicted: evicted
        ))
    }

    return accesses
  }

  private func chooseSlot(excluding protectedSlots: Set<Int>) -> Int {
    if let free = slotKeys.indices.first(where: {
      slotKeys[$0] == nil && !protectedSlots.contains($0)
    }) {
      return free
    }

    var victim: Int?
    var victimFrequency = Int.max
    var victimLastUse = UInt64.max

    for slot in slotKeys.indices where !protectedSlots.contains(slot) {
      guard let key = slotKeys[slot] else { return slot }
      let candidateFrequency = frequency[key] ?? 0
      let candidateLastUse = lastUse[slot] ?? 0
      if candidateFrequency < victimFrequency
        || (candidateFrequency == victimFrequency && candidateLastUse < victimLastUse)
        || (candidateFrequency == victimFrequency
          && candidateLastUse == victimLastUse
          && slot < (victim ?? Int.max))
      {
        victim = slot
        victimFrequency = candidateFrequency
        victimLastUse = candidateLastUse
      }
    }

    precondition(victim != nil, "batch protection should leave at least one victim")
    return victim!
  }
}
