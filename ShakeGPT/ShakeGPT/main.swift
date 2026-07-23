//
//  main.swift
//  ShakeGPT
//
//  Created by Bruno O
//


import Foundation
import MLX

guard CommandLine.arguments.count == 2 else {
    print("Usage \(CommandLine.arguments[0]) <filepath>")
    exit(EXIT_FAILURE)
}

let fileURL = URL(fileURLWithPath: CommandLine.arguments[1])

let corpus: String

do {
    corpus = try String(contentsOf: fileURL, encoding: .utf8)
} catch {
    print("Could not read contents of \(fileURL.path): \(error)")
    exit(EXIT_FAILURE)
}


// Learn a fixed byte-pair vocabulary from the training corpus.
let model = timed("Vocabulary training") {
    BPE(trainOn: corpus, maximumVocabularySize: 1_024)
}

// Encode the same corpus into the token IDs used by the language model.
let tokenIDs = model.encode(corpus)

// Each sampled sequence contains eight tokens. The embedding size determines
// how many learned coordinates represent each token.
let contextLength = 8
let embeddingSize = 64
let headCount = 4

// Sample four sequences and their one-token-shifted prediction targets.
let batch = makeBatch(
    from: tokenIDs,
    contextLength: contextLength,
    batchSize: 4
)

print(batch.inputs)
print(batch.targets)

// Turn token IDs into `[batch, context, embedding]` vectors by adding the
// learned token and positional embeddings.
let embeddings = InputEmbeddings(
    vocabularySize: model.vocabularySize,
    maximumContextLength: contextLength,
    embeddingSize: embeddingSize
)

let embeddedInputs = embeddings(batch.inputs)

// Split the 64-dimensional representation across four parallel heads.
// Each head therefore works with 16 dimensions.
let attention = MultiHeadSelfAttention(
    embeddingSize: embeddingSize,
    headCount: headCount
)

// Q, K and V are learned views of the input, reshaped to
// `[batch, head, context, headSize]`.
let projected = attention.projections(
    of: embeddedInputs
)

print("Input:", embeddedInputs.shape)
print("Queries:", projected.queries.shape)
print("Keys:", projected.keys.shape)
print("Values:", projected.values.shape)

// Every head independently produces a `[context, context]` score matrix.
let scores = attention.scores(
    queries: projected.queries,
    keys: projected.keys
)

// Causal masking hides future tokens; softmax turns the allowed scores into
// relative weights; dropout randomly removes some weights during training.
let weights = attention.weights(from: scores)

// Executable documentation for the shapes produced by multi-head attention.
let batchSize = batch.inputs.shape[0]
let headSize = embeddingSize / headCount

assert(projected.queries.shape == [batchSize, headCount, contextLength, headSize])
assert(projected.keys.shape == projected.queries.shape)
assert(projected.values.shape == projected.queries.shape)
assert(scores.shape == [batchSize, headCount, contextLength, contextLength])
assert(weights.shape == scores.shape)

print(weights)
