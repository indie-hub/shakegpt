//
//  Utils.swift
//  ShakeGPT
//
//  Created by Bruno O
//

import Foundation
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

    print("\n[\(label)] Finished. Time taken: \(start.duration(to: clock.now))")
    return result
}

/// Continues a prompt one token at a time using temperature sampling.
///
/// The model predicts logits for every position, but only the final position
/// describes what should come next. `topK` can restrict sampling to the strongest
/// candidates; a temperature of zero switches to deterministic greedy decoding.
func generate(
    after prompt: String,
    newTokenCount: Int,
    using model: ShakeGPT,
    tokeniser: BPE,
    contextLength: Int,
    temperature: Float = 1.0,
    topK: Int? = nil
) -> String {
    precondition(!prompt.isEmpty, "Prompt cannot be empty")
    precondition(newTokenCount >= 0, "Token count cannot be negative")
    precondition(contextLength > 0, "Context length must be positive")
    precondition(temperature >= 0, "Temperature cannot be negative")
    precondition(topK.map { $0 > 0 } ?? true, "Top-k must be positive")

    var tokenIDs = tokeniser.encode(prompt)

    // Dropout belongs to training, not generation. Restore the previous mode so
    // calling this helper cannot accidentally affect later training.
    let wasTraining = model.training
    model.train(false)
    defer { model.train(wasTraining) }

    // This recomputes the entire context for every token. A future KV cache can
    // avoid that repeated work if generation speed becomes important.
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
        assert(
            logits.shape == [
                1,
                context.count,
                tokeniser.modelVocabularySize
            ]
        )

        // Only the last position predicts the token that follows the context.
        var nextTokenLogits = logits[0, -1]

        let nextToken: MLXArray

        if temperature == 0 {
            // Zero temperature means always choosing the strongest candidate.
            nextToken = nextTokenLogits.argMax()
        } else {
            // Lower temperatures sharpen differences between logits; higher
            // temperatures flatten them and make unusual choices more likely.
            nextTokenLogits = nextTokenLogits / temperature

            if let topK {
                let candidateCount = min(
                    topK,
                    tokeniser.modelVocabularySize
                )

                if candidateCount < tokeniser.modelVocabularySize {
                    // `top` returns the k largest values. Its minimum is the
                    // cutoff below which candidates should become impossible.
                    let cutoff = MLX.top(
                        nextTokenLogits,
                        k: candidateCount
                    ).min()

                    nextTokenLogits = MLX.which(
                        nextTokenLogits .< cutoff,
                        -Float.infinity,
                        nextTokenLogits
                    )
                }
            }

            // Categorical accepts unnormalised logits and performs the
            // probability conversion internally, so no softmax is needed.
            nextToken = MLXRandom.categorical(nextTokenLogits)
        }

        eval(nextToken)

        // `item` copies the scalar token ID from MLX back into Swift.
        let nextTokenID = Int(nextToken.item(UInt32.self))

        // The model may finish before reaching the requested maximum length.
        if nextTokenID == tokeniser.endOfTextTokenID {
            break
        }

        tokenIDs.append(nextTokenID)
    }

    return tokeniser.decode(tokenIDs)
}

func updateLine(_ text: String) {
    // `print` normally buffers text until it sees a newline. Writing directly
    // makes a single-line progress display appear immediately, while carriage
    // return and ANSI erase replace the previous contents of that line.
    let line = "\r\u{001B}[2K\(text)"
    FileHandle.standardOutput.write(Data(line.utf8))
}
