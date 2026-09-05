import Foundation

public struct QuantizationSpec: Sendable, Equatable {
  public let bits: Int
  public let groupSize: Int
  public let scaleBytes: Int
  public let biasBytes: Int

  public init(bits: Int, groupSize: Int, scaleBytes: Int = 2, biasBytes: Int = 2) {
    precondition(bits > 0 && bits <= 16, "unsupported bit width")
    precondition(groupSize > 0, "group size must be positive")
    precondition(scaleBytes >= 0 && biasBytes >= 0, "metadata size must be non-negative")
    self.bits = bits
    self.groupSize = groupSize
    self.scaleBytes = scaleBytes
    self.biasBytes = biasBytes
  }

  public static let mlxInt4Group64 = QuantizationSpec(bits: 4, groupSize: 64)
}

public struct ExpertProjectionLayout: Sendable, Equatable {
  public let name: String
  public let inputFeatures: Int
  public let outputFeatures: Int
  public let weightBytes: Int64
  public let scaleBytes: Int64
  public let biasBytes: Int64
  public let offset: Int64

  public var byteCount: Int64 {
    weightBytes + scaleBytes + biasBytes
  }
}

public struct ExpertBlobLayout: Sendable, Equatable {
  public let quantization: QuantizationSpec
  public let pageAlignment: Int64
  public let sections: [ExpertProjectionLayout]
  public let payloadBytes: Int64
  public let strideBytes: Int64

  public init(
    hiddenSize: Int,
    intermediateSize: Int,
    quantization: QuantizationSpec = .mlxInt4Group64,
    pageAlignment: Int64 = 16_384
  ) {
    precondition(hiddenSize > 0 && intermediateSize > 0)
    precondition(pageAlignment > 0)

    self.quantization = quantization
    self.pageAlignment = pageAlignment

    let shapes: [(String, Int, Int)] = [
      ("gate_proj", hiddenSize, intermediateSize),
      ("up_proj", hiddenSize, intermediateSize),
      ("down_proj", intermediateSize, hiddenSize),
    ]

    var nextOffset: Int64 = 0
    var built: [ExpertProjectionLayout] = []
    built.reserveCapacity(shapes.count)

    for (name, input, output) in shapes {
      let weightCount = Int64(input) * Int64(output)
      precondition((weightCount * Int64(quantization.bits)).isMultiple(of: 8))
      precondition(weightCount.isMultiple(of: Int64(quantization.groupSize)))

      let weightBytes = weightCount * Int64(quantization.bits) / 8
      let groupCount = weightCount / Int64(quantization.groupSize)
      let scaleBytes = groupCount * Int64(quantization.scaleBytes)
      let biasBytes = groupCount * Int64(quantization.biasBytes)

      let section = ExpertProjectionLayout(
        name: name,
        inputFeatures: input,
        outputFeatures: output,
        weightBytes: weightBytes,
        scaleBytes: scaleBytes,
        biasBytes: biasBytes,
        offset: nextOffset
      )
      built.append(section)
      nextOffset += section.byteCount
    }

    sections = built
    payloadBytes = nextOffset
    strideBytes = BinarySize.alignUp(nextOffset, to: pageAlignment)
  }

  public static let ornithInt4 = ExpertBlobLayout(
    hiddenSize: OrnithProfile.v1_5_35BA3B.hiddenSize,
    intermediateSize: OrnithProfile.v1_5_35BA3B.expertIntermediateSize
  )

  public func totalPoolBytes(layerCount: Int, expertCount: Int) -> Int64 {
    precondition(layerCount >= 0 && expertCount >= 0)
    return strideBytes * Int64(layerCount) * Int64(expertCount)
  }

  public func coldReadBytesPerToken(layerCount: Int, expertsPerToken: Int) -> Int64 {
    precondition(layerCount >= 0 && expertsPerToken >= 0)
    return strideBytes * Int64(layerCount) * Int64(expertsPerToken)
  }
}
