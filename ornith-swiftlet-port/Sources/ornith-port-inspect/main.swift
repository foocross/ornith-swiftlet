import Foundation
import OrnithSwiftletPort

struct Arguments {
  var configPath: String?
  var memoryGiB: Double = 4.0
  var contextTokens: Int = 8_192

  init(_ values: [String]) throws {
    var index = 0
    while index < values.count {
      let value = values[index]
      switch value {
      case "--memory-gb":
        index += 1
        guard index < values.count, let parsed = Double(values[index]), parsed.isFinite, parsed > 0
        else {
          throw CLIError.usage("--memory-gb requires a positive number")
        }
        memoryGiB = parsed
      case "--context":
        index += 1
        guard index < values.count, let parsed = Int(values[index]), parsed >= 0 else {
          throw CLIError.usage("--context requires a non-negative integer")
        }
        contextTokens = parsed
      case "--help", "-h":
        throw CLIError.help
      default:
        if value.hasPrefix("-") {
          throw CLIError.usage("unknown option: \(value)")
        }
        guard configPath == nil else {
          throw CLIError.usage("only one config path may be supplied")
        }
        configPath = value
      }
      index += 1
    }
  }
}

enum CLIError: Error, CustomStringConvertible {
  case help
  case usage(String)

  var description: String {
    switch self {
    case .help:
      return ""
    case .usage(let message):
      return message
    }
  }
}

func printUsage() {
  print("Usage: ornith-port-inspect [config.json] [--memory-gb N] [--context TOKENS]")
}

func run() throws {
  let arguments = try Arguments(Array(CommandLine.arguments.dropFirst()))
  let profile = OrnithProfile.v1_5_35BA3B
  let layout = ExpertBlobLayout.ornithInt4

  print("Model: \(profile.name)")
  print("Architecture: \(profile.architecture) / \(profile.modelType)")
  print(
    "Layers: \(profile.layerCount) (\(profile.linearAttentionLayerCount) DeltaNet, \(profile.fullAttentionLayerCount) full attention)"
  )
  print("Experts: top-\(profile.expertsPerToken) of \(profile.expertCount) per layer")
  print(
    "Expert payload/stride: \(BinarySize.format(layout.payloadBytes)) / \(BinarySize.format(layout.strideBytes))"
  )
  print(
    "Routed expert pool: \(BinarySize.format(layout.totalPoolBytes(layerCount: profile.layerCount, expertCount: profile.expertCount)))"
  )
  print(
    "Cold K=8 reads/token: \(BinarySize.format(layout.coldReadBytesPerToken(layerCount: profile.layerCount, expertsPerToken: profile.expertsPerToken)))"
  )
  print(
    "KV at \(arguments.contextTokens) tokens (current Swiftlet FP32): \(BinarySize.format(profile.kvBytesPerTokenFP32 * Int64(arguments.contextTokens)))"
  )
  print(
    "KV at \(arguments.contextTokens) tokens (FP16 lower bound): \(BinarySize.format(profile.kvBytesPerTokenFP16 * Int64(arguments.contextTokens)))"
  )

  if let path = arguments.configPath {
    let config = try OrnithCheckpointConfig(url: URL(fileURLWithPath: path))
    let report = OrnithCompatibility.inspect(config)
    print("Config: \(report.isCompatible ? "compatible" : "incompatible")")
    for issue in report.issues {
      print("  \(issue)")
    }
  }

  let request = OrnithMemoryRequest(
    targetResidentBytes: BinarySize.gibibytes(arguments.memoryGiB),
    contextTokens: arguments.contextTokens
  )
  let plan = try OrnithMemoryGovernor.plan(request: request)
  print("Memory target: \(BinarySize.format(plan.targetResidentBytes))")
  print("  dense estimate: \(BinarySize.format(plan.denseCoreBytes))")
  print("  DeltaNet recurrence: \(BinarySize.format(plan.deltaNetRecurrenceBytes))")
  print("  DeltaNet conv tail: \(BinarySize.format(plan.deltaNetConvTailBytes))")
  print(
    "  KV cache: \(BinarySize.format(plan.kvCacheBytes)) (\(BinarySize.format(plan.kvBytesPerToken))/token)"
  )
  print("  scratch/safety: \(BinarySize.format(plan.scratchBytes + plan.safetyBytes))")
  print(
    "  expert cache: \(BinarySize.format(plan.expertCacheBytes)) (\(plan.expertSlots) global slots, \(String(format: "%.1f", plan.expertCoverageFraction * 100))% pool coverage)"
  )
  print("  planned total: \(BinarySize.format(plan.totalPlannedBytes))")
}

do {
  try run()
} catch CLIError.help {
  printUsage()
} catch {
  let message = "error: \(error)\n"
  FileHandle.standardError.write(Data(message.utf8))
  printUsage()
  exit(2)
}
