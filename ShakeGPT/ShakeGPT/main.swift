//
//  main.swift
//  ShakeGPT
//
//  Created by Bruno O
//


import Foundation

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
let model = BPE(trainOn: corpus, maximumVocabularySize: 1_024)

// Encode the same corpus into the token IDs used by the language model.
let tokenIDs = model.encode(corpus)

// Sample four sequences and their one-token-shifted prediction targets.
let batch = makeBatch(from: tokenIDs, contextLength: 8, batchSize: 4)

print(batch.inputs)
print(batch.targets)
