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


/// Samples from the smallest group of likely tokens whose probability reaches
/// `topP`. Sorting is necessary because "smallest group" means most likely first.
private func sampleTopP(
    from logits: MLXArray,
    topP: Float
) -> MLXArray {
    let probabilities = softmax(logits)

    let sortedIndices = argSort(-probabilities)
    let sortedProbabilities = takeAlong(probabilities, sortedIndices)
    let cumulativeProbabilities = sortedProbabilities.cumsum()
    let probabilitiesBeforeToken =
        cumulativeProbabilities - sortedProbabilities
    let shouldExclude = probabilitiesBeforeToken .>= topP

    let sortedLogits = takeAlong(logits, sortedIndices)
    let filteredLogits = MLX.which(shouldExclude, -Float.infinity, sortedLogits)
    let sortedChoice = MLXRandom.categorical(filteredLogits)

    return takeAlong(
        sortedIndices,
        sortedChoice.expandedDimensions(axis: 0)
    ).squeezed()
}

/// Reduces the chance of immediately reusing ordinary tokens from recent text.
///
/// Positive logits are divided and negative logits are multiplied, pushing
/// both away from selection without banning a token completely. Structural
/// special tokens are excluded because a play legitimately repeats markers
/// such as `<|speaker|>`.
func applyingRepetitionPenalty(
    to logits: MLXArray,
    tokenIDs: [Int],
    excluding excludedIDs: Set<Int>,
    penalty: Float,
    window: Int = 128
) -> MLXArray {
    guard penalty > 1 else { return logits }

    let repeatedIDs = Array(Set(tokenIDs.suffix(window)).subtracting(excludedIDs))
    guard !repeatedIDs.isEmpty else { return logits }

    var penalties = [Float](repeating: 1, count: logits.size)
    for id in repeatedIDs where penalties.indices.contains(id) {
        penalties[id] = penalty
    }
    let penaltyByToken = MLXArray(penalties)

    return MLX.which(
        logits .< 0,
        logits * penaltyByToken,
        logits / penaltyByToken
    )
}


/// Continues a prompt one token at a time using temperature sampling.
///
/// The model predicts logits for every position, but only the final position
/// describes what should come next. `topK` limits the number of candidates,
/// while `topP` limits their combined probability. A temperature of zero
/// switches to deterministic greedy decoding. `repetitionPenalty` gently lowers
/// recently generated ordinary tokens; a value of one leaves logits unchanged.
func generate(
    after prompt: String,
    newTokenCount: Int,
    using model: ShakeGPT,
    tokeniser: BPE,
    contextLength: Int,
    temperature: Float = 1.0,
    topK: Int? = nil,
    topP: Float? = nil,
    repetitionPenalty: Float = 1.0
) -> String {
    precondition(!prompt.isEmpty, "Prompt cannot be empty")
    precondition(newTokenCount >= 0, "Token count cannot be negative")
    precondition(contextLength > 0, "Context length must be positive")
    precondition(temperature >= 0, "Temperature cannot be negative")
    precondition(topK.map { $0 > 0 } ?? true, "Top-k must be positive")
    precondition(topP.map { $0 > 0 && $0 <= 1 } ?? true,
                     "Top-p must be greater than zero and no greater than one")
    precondition(repetitionPenalty >= 1, "Repetition penalty cannot be below one")

    var tokenIDs = tokeniser.encode(prompt)
    var generatedTokenIDs: [Int] = []
    let specialTokenIDs = Set(tokeniser.specialTokens.map(\.id))

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
        nextTokenLogits = applyingRepetitionPenalty(
            to: nextTokenLogits,
            tokenIDs: generatedTokenIDs,
            excluding: specialTokenIDs,
            penalty: repetitionPenalty
        )

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

            if let topP {
                nextToken = sampleTopP(from: nextTokenLogits, topP: topP)
            } else {
                // Categorical accepts unnormalised logits and performs the
                // probability conversion internally, so no softmax is needed.
                nextToken = MLXRandom.categorical(nextTokenLogits)
            }
        }

        eval(nextToken)

        // `item` copies the scalar token ID from MLX back into Swift.
        let nextTokenID = Int(nextToken.item(UInt32.self))

        // The model may finish before reaching the requested maximum length.
        if nextTokenID == tokeniser.endOfTextTokenID {
            break
        }

        tokenIDs.append(nextTokenID)
        generatedTokenIDs.append(nextTokenID)
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
