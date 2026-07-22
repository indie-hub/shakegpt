//
//  Training.swift
//  ShakeGPT
//
//  Created by Bruno O
//

import MLX

/// Creates one random mini-batch for next-token prediction.
///
/// Each input row contains `contextLength` consecutive token IDs. Its matching
/// target row contains the same sequence shifted one position forward, so every
/// input position is paired with the token that should follow it.
///
/// Both returned tensors have shape `[batchSize, contextLength]` and use
/// `Int32` values because token IDs are indices rather than continuous values.
func makeBatch(
    from tokenIDs: [Int],
    contextLength: Int,
    batchSize: Int
) -> (
    inputs: MLXArray,
    targets: MLXArray
) {
    precondition(contextLength > 0, "Context length must be positive")
    precondition(batchSize > 0, "Batch size must be positive")
    precondition(
        tokenIDs.count > contextLength,
        "The corpus must contain more tokens than the context length"
    )

    var inputs: [Int32] = []
    var targets: [Int32] = []

    for _ in 0..<batchSize {
        // Sampling a fresh start lets each training step see different text.
        let startIndex = Int.random(
            in: 0..<(tokenIDs.count - contextLength)
        )

        // Append complete rows in row-major order before giving them a 2D shape.
        inputs.append(
            contentsOf: tokenIDs[
                startIndex ..< startIndex + contextLength
            ].map(Int32.init)
        )
        targets.append(
            contentsOf: tokenIDs[
                startIndex + 1 ... startIndex + contextLength
            ].map(Int32.init)
        )
    }

    let shape = [batchSize, contextLength]

    return (
        inputs: MLXArray(inputs, shape),
        targets: MLXArray(targets, shape)
    )
}
