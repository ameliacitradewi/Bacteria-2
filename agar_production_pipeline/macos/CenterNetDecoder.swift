import CoreML
import Foundation

struct ColonyBox: Sendable {
    var x1: Double
    var y1: Double
    var x2: Double
    var y2: Double
    var score: Double

    var width: Double { max(0, x2 - x1) }
    var height: Double { max(0, y2 - y1) }
    var centerX: Double { (x1 + x2) * 0.5 }
    var centerY: Double { (y1 + y2) * 0.5 }
}

enum CenterNetDecoderError: Error {
    case invalidShape(String)
}

enum CenterNetDecoder {
    /// Decoder untuk output `ColonyResNet50FPN.mlpackage`.
    /// Model mengeluarkan heatmap, size, dan offset dalam bentuk NCHW.
    static func decode(
        heatmap: MLMultiArray,
        size: MLMultiArray,
        offset: MLMultiArray,
        tileOriginX: Int,
        tileOriginY: Int,
        stride: Int = 4,
        scoreThreshold: Double = 0.25,
        topK: Int = 500,
        tileSize: Int = 512,
        tileNMSIoU: Double = 0.30
    ) throws -> [ColonyBox] {
        guard heatmap.shape.count == 4,
              size.shape.count == 4,
              offset.shape.count == 4 else {
            throw CenterNetDecoderError.invalidShape("Output harus NCHW.")
        }

        let height = heatmap.shape[2].intValue
        let width = heatmap.shape[3].intValue
        guard size.shape[1].intValue == 2,
              offset.shape[1].intValue == 2,
              size.shape[2].intValue == height,
              size.shape[3].intValue == width else {
            throw CenterNetDecoderError.invalidShape("Shape size/offset tidak cocok.")
        }

        var peaks: [(score: Double, x: Int, y: Int)] = []
        peaks.reserveCapacity(min(topK * 2, height * width))

        for y in 0..<height {
            for x in 0..<width {
                let score = value(heatmap, channel: 0, y: y, x: x)
                guard score >= scoreThreshold else { continue }
                guard isLocalMaximum(heatmap, x: x, y: y, width: width, height: height) else {
                    continue
                }
                peaks.append((score, x, y))
            }
        }
        peaks.sort { $0.score > $1.score }
        if peaks.count > topK {
            peaks.removeSubrange(topK..<peaks.count)
        }

        var decoded: [ColonyBox] = []
        decoded.reserveCapacity(peaks.count)
        for peak in peaks {
            let boxWidth = max(1.0, value(size, channel: 0, y: peak.y, x: peak.x) * Double(stride))
            let boxHeight = max(1.0, value(size, channel: 1, y: peak.y, x: peak.x) * Double(stride))
            let offsetX = value(offset, channel: 0, y: peak.y, x: peak.x)
            let offsetY = value(offset, channel: 1, y: peak.y, x: peak.x)
            let centerX = (Double(peak.x) + offsetX) * Double(stride)
            let centerY = (Double(peak.y) + offsetY) * Double(stride)

            let localX1 = max(0.0, centerX - boxWidth * 0.5)
            let localY1 = max(0.0, centerY - boxHeight * 0.5)
            let localX2 = min(Double(tileSize), centerX + boxWidth * 0.5)
            let localY2 = min(Double(tileSize), centerY + boxHeight * 0.5)
            guard localX2 - localX1 >= 1.0, localY2 - localY1 >= 1.0 else { continue }

            decoded.append(
                ColonyBox(
                    x1: localX1 + Double(tileOriginX),
                    y1: localY1 + Double(tileOriginY),
                    x2: localX2 + Double(tileOriginX),
                    y2: localY2 + Double(tileOriginY),
                    score: peak.score
                )
            )
        }
        return nonMaximumSuppression(decoded, iouThreshold: tileNMSIoU)
    }

    static func tileOrigins(length: Int, tileSize: Int = 512, overlap: Int = 128) -> [Int] {
        precondition(tileSize > 0 && overlap >= 0 && overlap < tileSize)
        guard length > tileSize else { return [0] }
        let step = tileSize - overlap
        var starts = Array(Swift.stride(from: 0, through: length - tileSize, by: step))
        let last = length - tileSize
        if starts.last != last { starts.append(last) }
        return starts
    }

    static func removeTileEdgeDetections(
        _ boxes: [ColonyBox],
        tileOriginX: Int,
        tileOriginY: Int,
        sourceWidth: Int,
        sourceHeight: Int,
        tileSize: Int = 512,
        margin: Int = 16
    ) -> [ColonyBox] {
        guard margin > 0 else { return boxes }
        return boxes.filter { box in
            let localCenterX = box.centerX - Double(tileOriginX)
            let localCenterY = box.centerY - Double(tileOriginY)
            if tileOriginX > 0 && localCenterX < Double(margin) { return false }
            if tileOriginY > 0 && localCenterY < Double(margin) { return false }
            if tileOriginX + tileSize < sourceWidth && localCenterX >= Double(tileSize - margin) {
                return false
            }
            if tileOriginY + tileSize < sourceHeight && localCenterY >= Double(tileSize - margin) {
                return false
            }
            return true
        }
    }

    static func nonMaximumSuppression(
        _ boxes: [ColonyBox],
        iouThreshold: Double
    ) -> [ColonyBox] {
        var candidates = boxes.sorted { $0.score > $1.score }
        var selected: [ColonyBox] = []
        while let best = candidates.first {
            selected.append(best)
            candidates.removeFirst()
            candidates.removeAll { intersectionOverUnion(best, $0) > iouThreshold }
        }
        return selected
    }

    static func keepCentersInsideMask(
        _ boxes: [ColonyBox],
        mask: [UInt8],
        width: Int,
        height: Int
    ) -> [ColonyBox] {
        precondition(mask.count == width * height)
        return boxes.filter { box in
            let x = Int(box.centerX.rounded())
            let y = Int(box.centerY.rounded())
            guard x >= 0, x < width, y >= 0, y < height else { return false }
            return mask[y * width + x] > 0
        }
    }

    private static func isLocalMaximum(
        _ array: MLMultiArray,
        x: Int,
        y: Int,
        width: Int,
        height: Int
    ) -> Bool {
        let center = value(array, channel: 0, y: y, x: x)
        for yy in max(0, y - 1)...min(height - 1, y + 1) {
            for xx in max(0, x - 1)...min(width - 1, x + 1) {
                if value(array, channel: 0, y: yy, x: xx) > center { return false }
            }
        }
        return true
    }

    private static func value(
        _ array: MLMultiArray,
        channel: Int,
        y: Int,
        x: Int
    ) -> Double {
        let offset = channel * array.strides[1].intValue
            + y * array.strides[2].intValue
            + x * array.strides[3].intValue
        return array[offset].doubleValue
    }

    private static func intersectionOverUnion(_ first: ColonyBox, _ second: ColonyBox) -> Double {
        let intersectionX1 = max(first.x1, second.x1)
        let intersectionY1 = max(first.y1, second.y1)
        let intersectionX2 = min(first.x2, second.x2)
        let intersectionY2 = min(first.y2, second.y2)
        let intersection = max(0, intersectionX2 - intersectionX1)
            * max(0, intersectionY2 - intersectionY1)
        let union = first.width * first.height + second.width * second.height - intersection
        return union > 0 ? intersection / union : 0
    }
}
