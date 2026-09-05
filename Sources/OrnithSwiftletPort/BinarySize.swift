import Foundation

public enum BinarySize {
  public static let kibibyte: Int64 = 1_024
  public static let mebibyte: Int64 = 1_024 * kibibyte
  public static let gibibyte: Int64 = 1_024 * mebibyte

  public static func alignUp(_ value: Int64, to alignment: Int64) -> Int64 {
    precondition(value >= 0, "value must be non-negative")
    precondition(alignment > 0, "alignment must be positive")
    return ((value + alignment - 1) / alignment) * alignment
  }

  public static func gibibytes(_ value: Double) -> Int64 {
    Int64((value * Double(gibibyte)).rounded(.down))
  }

  public static func format(_ bytes: Int64) -> String {
    if bytes >= gibibyte {
      return String(format: "%.2f GiB", Double(bytes) / Double(gibibyte))
    }
    if bytes >= mebibyte {
      return String(format: "%.1f MiB", Double(bytes) / Double(mebibyte))
    }
    if bytes >= kibibyte {
      return String(format: "%.1f KiB", Double(bytes) / Double(kibibyte))
    }
    return "\(bytes) B"
  }
}
