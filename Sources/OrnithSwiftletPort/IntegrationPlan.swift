import Foundation

public struct PortMilestone: Sendable, Equatable {
  public let name: String
  public let purpose: String
  public let status: String
}

public enum OrnithSwiftletIntegration {
  public static let upstreamRepository = "https://github.com/leonickson1/Swiftlet"
  public static let modelRepository = OrnithProfile.v1_5_35BA3B.sourceCheckpoint

  public static let milestones: [PortMilestone] = [
    PortMilestone(
      name: "Configuration and tensor-layout acceptance",
      purpose: "Recognize the exact Ornith geometry and reject silent shape drift.",
      status: "implemented in this kit"
    ),
    PortMilestone(
      name: "Qpack repack smoke test",
      purpose:
        "Run Swiftlet's existing one-pread-per-expert repacker on the official MLX int4 checkpoint.",
      status: "requires the model checkpoint on macOS"
    ),
    PortMilestone(
      name: "Reference parity",
      purpose: "Compare greedy token IDs and selected router experts with mlx-lm.",
      status: "requires the model checkpoint and Apple Metal"
    ),
    PortMilestone(
      name: "Elastic expert-cache budget",
      purpose:
        "Use the SlotStream-style planner to size the global Swiftlet cache from a hard memory target.",
      status: "planner implemented; runtime wiring documented"
    ),
    PortMilestone(
      name: "Bounded parallel expert reads",
      purpose:
        "Replace serial miss fills with bounded pread workers writing directly to shared Metal slots.",
      status: "design only; requires profiling before merge"
    ),
    PortMilestone(
      name: "Fused top-8 MoE kernel",
      purpose: "Reduce dispatches while preserving native K=8 numerics.",
      status: "future optimization"
    ),
    PortMilestone(
      name: "MTP verification",
      purpose: "Use Ornith's one MTP layer for speculative decoding after baseline parity.",
      status: "future optimization; current qpack excludes MTP"
    ),
  ]
}
