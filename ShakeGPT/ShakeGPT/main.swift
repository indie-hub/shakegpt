//
//  main.swift
//  ShakeGPT
//
//  Created by Bruno O
//
import Foundation
import ArgumentParser
import MLX

/// Command-line inputs keep local corpus and checkpoint paths out of the source.
struct Options: ParsableArguments {
    @Option(
        help: "UTF-8 training corpus",
        transform: URL.init(fileURLWithPath:)
    )
    var corpus: URL

    @Option(
        help: "Optional UTF-8 validation corpus",
        transform: URL.init(fileURLWithPath:)
    )
    var validation: URL?

    @Option(
        help: "Saved BPE vocabulary, created when it does not exist",
        transform: URL.init(fileURLWithPath:)
    )
    var vocabulary: URL

    @Flag(help: "Train a new model instead of only generating text")
    var train: Bool = false

    @Flag(help: "Load the best checkpoint before continuing training")
    var resume: Bool = false

    @Argument(
        help: "Prompt whose text the model should continue"
    )
    var question: String
}

let options = Options.parseOrExit()

// A compact GPT-style configuration for the much smaller Shakespeare corpus.
// `embeddingSize` must be divisible by `headCount`, giving every attention head
// an equally sized portion of the embedding.
let contextLength: Int = 256
let embeddingSize: Int = 384
let headCount: Int = 6
let layerCount: Int = 6
let dropoutProbability: Float = 0.2
let maximumVocabularySize: Int = 4_096
let batchSize: Int = 8
let epochCount: Int = 10

let shouldLoadCheckpoint = options.resume || !options.train

let device = Device.defaultDevice()

print("MLX device:", device)
print("Using GPU:", device.deviceType == .gpu)

let corpusURL = options.corpus
let vocabularyURL = options.vocabulary

// Each vocabulary gets a matching checkpoint beside it. A new vocabulary path
// therefore starts a clean experiment without overwriting an older model.
let checkpointName = vocabularyURL.deletingPathExtension().lastPathComponent
    + "-best.safetensors"
let checkpointURL = vocabularyURL
    .deletingLastPathComponent()
    .appendingPathComponent(checkpointName)

let corpus: String

do {
    corpus = try String(contentsOf: corpusURL, encoding: .utf8)
} catch {
    print("Could not read contents of \(corpusURL.path): \(error)")
    exit(EXIT_FAILURE)
}

// Learn a fixed byte-pair vocabulary from the training corpus.
let tokeniser = timed("Vocabulary training") {
    if FileManager.default.fileExists(atPath: vocabularyURL.path) {
        do {
            print("Loading vocabulary from \(vocabularyURL.path)")
            return try BPE.load(from: vocabularyURL)
        } catch {
            fatalError(
                "Failed to load vocabulary from "
                    + "\(vocabularyURL.path): \(error)"
            )
        }
    }

    print("Training a new vocabulary.")

    // EOT is a reserved model token rather than ordinary training text. Replacing
    // its visible spelling prevents BPE from learning pieces of the marker.
    let corpusWithoutMarkers = corpus.replacingOccurrences(
        of: BPE.endOfTextMarker,
        with: "\n\n"
    )

    let tokeniser = BPE(
        trainOn: corpusWithoutMarkers,
        maximumVocabularySize: maximumVocabularySize
    )

    do {
        try tokeniser.save(to: vocabularyURL)
    } catch {
        fatalError(
            "Failed to save vocabulary to "
                + "\(vocabularyURL.path): \(error)"
        )
    }

    return tokeniser
}

// Uniform random guessing assigns probability 1 / vocabularySize to the answer,
// so its expected cross-entropy is the natural logarithm of the vocabulary size.
let randomBaselineLoss = log(
    Double(tokeniser.modelVocabularySize)
)

print(
    "Random baseline loss: ",
    randomBaselineLoss
)

// The tokenizer determines the output vocabulary. Every other value controls
// the geometry and regularisation of the transformer.
let modelConfig = ShakeGPT.Config(
    vocabularySize: tokeniser.modelVocabularySize,
    contextLength: contextLength,
    embeddingSize: embeddingSize,
    headCount: headCount,
    layerCount: layerCount,
    dropoutProbability: dropoutProbability,
    qkvBias: false
)

let model = ShakeGPT(config: modelConfig)

// Generation and resumed training need learned parameters. Fresh training starts
// from the model's random initial values instead.
if shouldLoadCheckpoint {
    guard FileManager.default.fileExists(atPath: checkpointURL.path) else {
        fatalError("No checkpoint exists at \(checkpointURL.path)")
    }

    do {
        let validationLoss = try loadCheckpoint(
            into: model,
            from: checkpointURL
        )
        print(
            "Continuing from checkpoint with validation loss",
            "\(validationLoss)."
        )
        print("AdamW starts with fresh optimizer state.")
    } catch {
        fatalError(
            "Failed to load checkpoint from \(checkpointURL.path): \(error)"
        )
    }
}

// Parameter storage counts the learned tensors only. Training additionally
// needs memory for activations, gradients and optimizer state.
let formatter = ByteCountFormatter()
formatter.countStyle = .memory

print("Trainable parameters:", model.parameterCount.formatted())
print(
    "Parameter storage:",
    formatter.string(fromByteCount: Int64(model.parameterBytes))
)

if options.train {
    // Unlike ordinary prompt encoding, corpus encoding converts every visible
    // document boundary into the transformer's dedicated EOT token.
    let tokens = timed("Tokenising corpus") {
        tokeniser.encodeWithSpecialMarkers(corpus)
    }

    let validationTokens: [Int]?

    if let validationURL = options.validation {
        do {
            let validationCorpus = try String(
                contentsOf: validationURL,
                encoding: .utf8
            )

            validationTokens = tokeniser.encodeWithSpecialMarkers(
                validationCorpus
            )
        } catch {
            fatalError(
                "Could not read validation corpus at "
                    + "\(validationURL.path): \(error)"
            )
        }
    } else {
        // This fallback is useful for a single unprepared corpus. Supplying a
        // validation path always takes precedence and read failures stop the run.
        print(
            "Warning: no validation corpus supplied; "
                + "using the final 10% of the training corpus."
        )
        validationTokens = nil
    }

    do {
        try timed("Training") {
            try train(
                model: model,
                on: tokens,
                validatedWith: validationTokens,
                tokeniser: tokeniser,
                contextLength: contextLength,
                batchSize: batchSize,
                epochs: epochCount,
                evalCadence: 100,
                evaluationText: "To be or ",
                checkpointURL: checkpointURL,
                isContinuation: options.resume
            )
        }
    } catch {
        fatalError(
            "Training failed while saving \(checkpointURL.path): \(error)"
        )
    }
}

// Generation needs one input sequence, but no random batch or shifted targets.
// Temperature and top-k trade deterministic choices for controlled variety.
let answer = timed("Generation") {
    generate(
        after: options.question,
        newTokenCount: contextLength,
        using: model,
        tokeniser: tokeniser,
        contextLength: contextLength,
        temperature: 0.7,
        topK: 40
    )
}

print("Answer:\n\(answer)\nDone.")
