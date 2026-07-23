//
//  main.swift
//  ShakeGPT
//
//  Created by Bruno O
//
import Foundation

// Small values for quickly checking the complete forward and generation path.
// Swap these with the active values below while developing on a small model.
/*
let contextLength: Int = 8
let embeddingSize: Int = 64
let headCount: Int = 4
let layerCount: Int = 2
let dropoutProbability: Float = 0.1
let maximumVocabularySize: Int = 260
*/

// GPT-2 Small-style model dimensions. Shakespeare is much smaller than GPT-2's
// training data, so these describe the architecture rather than a recommended
// final training configuration.
let contextLength: Int = 1_024
let embeddingSize: Int = 768
let headCount: Int = 12
let layerCount: Int = 12
let dropoutProbability: Float = 0.1
let maximumVocabularySize: Int = 1_024


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
let tokeniser = timed("Vocabulary training") {
    BPE(trainOn: corpus, maximumVocabularySize: maximumVocabularySize)
}

// The tokenizer determines the output vocabulary. Every other value controls
// the geometry and regularisation of the transformer.
let modelConfig = ShakeGPT.Config(
    vocabularySize: tokeniser.vocabularySize,
    contextLength: contextLength,
    embeddingSize: embeddingSize,
    headCount: headCount,
    layerCount: layerCount,
    dropoutProbability: dropoutProbability,
    qkvBias: false
)

let model = ShakeGPT(config: modelConfig)

// Parameter storage counts the learned tensors only. Training additionally
// needs memory for activations, gradients and optimizer state.
let formatter = ByteCountFormatter()
formatter.countStyle = .memory

print("Trainable parameters:", model.parameterCount.formatted())
print(
    "Parameter storage:",
    formatter.string(fromByteCount: Int64(model.parameterBytes))
)

// Generation needs a batch dimension of one, but no sampled training batch or
// shifted targets. The untrained model will still produce essentially random
// text; this call only checks that the complete inference path works.
let answer = timed("Generation") {
    generate(
        after: "To be, or ",
        newTokenCount: 8,
        using: model,
        tokeniser: tokeniser,
        contextLength: contextLength
    )
}

print("Answer:\n\(answer)\nDone.")
