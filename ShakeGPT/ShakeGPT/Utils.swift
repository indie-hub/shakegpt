//
//  Utils.swift
//  ShakeGPT
//
//  Created by Bruno O
//

import MLX
import MLXNN

/// Runs an operation, prints its elapsed wall-clock time, and returns its result.
///
/// `rethrows` means this helper only throws when the operation itself throws.
func timed<T>(
    _ label: String = "",
    operation: () throws -> T
) rethrows -> T {
    print("[\(label)]: Starting...")

    let clock = ContinuousClock()
    let start = clock.now

    let result = try operation()

    print("[\(label)] Finished. Time taken: \(start.duration(to: clock.now))")
    return result
}

/// Continues a prompt one token at a time using greedy decoding.
///
/// The model predicts logits for every position, but only the final position
/// describes what should come next. `argMax` chooses its highest-scoring token.
func generate(
    after prompt: String,
    newTokenCount: Int,
    using model: ShakeGPT,
    tokeniser: BPE,
    contextLength: Int
) -> String {
    precondition(!prompt.isEmpty, "Prompt cannot be empty")
    precondition(newTokenCount >= 0, "Token count cannot be negative")
    precondition(contextLength > 0, "Context length must be positive")

    var tokenIDs = tokeniser.encode(prompt)

    // Dropout belongs to training, not generation. Restore the previous mode so
    // calling this helper cannot accidentally affect later training.
    let wasTraining = model.training
    model.train(false)
    defer { model.train(wasTraining) }

    // ponytail: This recomputes the entire context for every token. Add a KV
    // cache only when generation speed becomes important.
    for _ in 0..<newTokenCount {
        // Learned positional embeddings limit the model to its context window.
        let context = Array(tokenIDs.suffix(contextLength))

        // Generation uses one sequence, hence the batch dimension of one.
        let input = MLXArray(
            context.map { Int32($0) },
            [1, context.count]
        )

        // Shape: [batch: 1, context, vocabulary].
        let logits = model(input)

        assert(logits.shape == [1, context.count, tokeniser.vocabularySize])

        // Only the last position predicts the token that follows the context.
        // `item` copies the scalar token ID from MLX back into Swift.
        let nextToken = logits[0, -1].argMax()
        eval(nextToken)

        tokenIDs.append(Int(nextToken.item(UInt32.self)))
    }

    return tokeniser.decode(tokenIDs)
}
