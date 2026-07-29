//
//  main.swift
//  ShakeGPT
//

import ArgumentParser
import Foundation
import MLX

/// Ends a command-line run with a readable error instead of a crash backtrace.
func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("Error: \(message)\n".utf8))
    exit(EXIT_FAILURE)
}

/// CLI values are intentionally optional: JSON fills what the command line omits.
struct Options: ParsableArguments {
    @Option(help: "Version-1 JSON run configuration", transform: URL.init(fileURLWithPath:))
    var config: URL?
    @Option(help: "UTF-8 training corpus", transform: URL.init(fileURLWithPath:))
    var corpus: URL?
    @Option(help: "Optional UTF-8 validation corpus", transform: URL.init(fileURLWithPath:))
    var validation: URL?
    @Option(help: "Saved BPE vocabulary", transform: URL.init(fileURLWithPath:))
    var vocabulary: URL?
    @Option(help: "UTF-8 text used only to create BPE; repeat to combine corpora", transform: URL.init(fileURLWithPath:))
    var vocabularyCorpus: [URL] = []
    @Option(help: "Checkpoint to save or load", transform: URL.init(fileURLWithPath:))
    var checkpoint: URL?
    @Option(help: "Checkpoint to load before training", transform: URL.init(fileURLWithPath:))
    var resumeFrom: URL?
    @Flag(help: "Train a new model instead of only generating text") var train = false
    @Flag(help: "Load the best checkpoint before continuing training") var resume = false
    @Option(help: "AdamW learning rate") var learningRate: Float?
    @Option(help: "Sampling temperature; zero is greedy") var temperature: Float?
    @Option(help: "Highest-scoring tokens considered during sampling") var topK: Int?
    @Option(help: "Smallest likely-token probability mass") var topP: Float?
    @Option(help: "Penalty applied to recently used tokens; one disables it")
    var repetitionPenalty: Float?
    @Argument(help: "Prompt whose text the model should continue") var question: String

    mutating func validate() throws {
        if resume && resumeFrom != nil {
            throw ValidationError("Use either --resume or --resume-from, not both")
        }
    }

    var overrides: RunOverrides {
        .init(corpus: corpus, validation: validation, vocabulary: vocabulary,
              vocabularyCorpora: vocabularyCorpus, checkpoint: checkpoint,
              resumeFrom: resumeFrom, maximumModelVocabularySize: nil,
              contextLength: nil, embeddingSize: nil, headCount: nil, layerCount: nil,
              dropoutProbability: nil, qkvBias: nil, batchSize: nil, epochs: nil,
              learningRate: learningRate, evalCadence: nil, patience: nil,
              newTokenCount: nil, temperature: temperature, topK: topK, topP: topP,
              repetitionPenalty: repetitionPenalty)
    }
}

let options = Options.parseOrExit()
let preliminary: ResolvedRunConfiguration
do {
    preliminary = try ResolvedRunConfiguration.resolve(
        configurationURL: options.config, overrides: options.overrides, isTraining: options.train
    )
} catch {
    fail("Invalid run configuration: \(error.localizedDescription)")
}

let device = Device.defaultDevice()
print("MLX device:", device)
print("Using GPU:", device.deviceType == .gpu)

let vocabularyURL = URL(fileURLWithPath: preliminary.data.vocabulary)
let corpusURL = preliminary.data.corpus.map(URL.init(fileURLWithPath:))
let checkpointURL = URL(fileURLWithPath: preliminary.data.checkpoint)

// `resumeFrom` is an input used only when continuing training, while
// `checkpoint` is the best model this run saves and ordinary generation loads.
// `--resume` is shorthand for continuing from that same best checkpoint.
let checkpointToLoad: URL?
if options.train, let resume = preliminary.data.resumeFrom { checkpointToLoad = URL(fileURLWithPath: resume) }
else if options.resume || !options.train { checkpointToLoad = checkpointURL }
else { checkpointToLoad = nil }
let isContinuation = options.train && checkpointToLoad != nil

let corpus: String?
if let corpusURL {
    do { corpus = try String(contentsOf: corpusURL, encoding: .utf8) }
    catch { fail("Could not read \(corpusURL.path): \(error.localizedDescription)") }
} else { corpus = nil }

let tokeniser = timed("Vocabulary training") {
    if FileManager.default.fileExists(atPath: vocabularyURL.path) {
        do {
            print("Loading vocabulary from \(vocabularyURL.path)")
            return try BPE.load(from: vocabularyURL)
        } catch { fail("Failed to load vocabulary from \(vocabularyURL.path): \(error.localizedDescription)") }
    }
    let vocabularyTrainingCorpus: String
    if preliminary.data.vocabularyCorpora.isEmpty {
        guard let corpus else {
            fail("Vocabulary does not exist at \(vocabularyURL.path). Supply --vocabulary-corpus or create it during training.")
        }
        vocabularyTrainingCorpus = corpus
    } else {
        do {
            vocabularyTrainingCorpus = try preliminary.data.vocabularyCorpora.map {
                try String(contentsOf: URL(fileURLWithPath: $0), encoding: .utf8)
            }.joined(separator: "\n\n")
        } catch { fail("Could not read vocabulary corpus: \(error.localizedDescription)") }
    }
    let tokenizer = BPE(
        trainOn: vocabularyTrainingCorpus,
        maximumVocabularySize: preliminary.tokenizer.maximumModelVocabularySize,
        specialTokens: preliminary.tokenizer.specialTokens.map { .init(name: $0.name, text: $0.text) }
    )
    do { try tokenizer.save(to: vocabularyURL) }
    catch { fail("Failed to save vocabulary to \(vocabularyURL.path): \(error.localizedDescription)") }
    return tokenizer
}

let requestedSpecialTokens = preliminary.tokenizer.specialTokens.map {
    BPE.SpecialTokenDefinition(name: $0.name, text: $0.text)
}
let loadedSpecialTokens = tokeniser.specialTokens.map {
    BPE.SpecialTokenDefinition(name: $0.name, text: $0.text)
}
guard requestedSpecialTokens == loadedSpecialTokens else {
    fail("The configured special tokens do not match the saved vocabulary.")
}

let resolved: ResolvedRunConfiguration
do {
    resolved = try ResolvedRunConfiguration.resolve(
        configurationURL: options.config, overrides: options.overrides,
        isTraining: options.train, tokenizer: tokeniser
    )
} catch { fail("Invalid run configuration: \(error.localizedDescription)") }

let model = ShakeGPT(config: resolved.model)
print("Random baseline loss:", log(Double(tokeniser.modelVocabularySize)))

if let checkpointToLoad {
    guard FileManager.default.fileExists(atPath: checkpointToLoad.path) else {
        fail("No checkpoint exists at \(checkpointToLoad.path)")
    }
    do {
        let validationLoss = try loadCheckpoint(into: model, from: checkpointToLoad, configuration: resolved)
        print(isContinuation ? "Continuing from checkpoint with validation loss" : "Loaded checkpoint with validation loss", "\(validationLoss).")
        if isContinuation { print("AdamW starts with fresh optimizer state.") }
    } catch { fail("Failed to load checkpoint from \(checkpointToLoad.path): \(error.localizedDescription)") }
}

let formatter = ByteCountFormatter()
formatter.countStyle = .memory
print("Trainable parameters:", model.parameterCount.formatted())
print("Parameter storage:", formatter.string(fromByteCount: Int64(model.parameterBytes)))

if options.train {
    guard let corpus else { fail("--corpus is required when training") }
    let tokens = timed("Tokenising corpus") { tokeniser.encodeWithSpecialMarkers(corpus) }
    let validationTokens: [Int]?
    if let validation = resolved.data.validation {
        do { validationTokens = tokeniser.encodeWithSpecialMarkers(try String(contentsOf: URL(fileURLWithPath: validation), encoding: .utf8)) }
        catch { fail("Could not read validation corpus at \(validation): \(error.localizedDescription)") }
    } else {
        print("Warning: no validation corpus supplied; using the final 10% of the training corpus.")
        validationTokens = nil
    }
    do {
        try timed("Training") {
            try train(model: model, on: tokens, validatedWith: validationTokens, tokeniser: tokeniser,
                      contextLength: resolved.model.contextLength, batchSize: resolved.training.batchSize,
                      epochs: resolved.training.epochs, evalCadence: resolved.training.evalCadence,
                      evaluationText: "To be or ", patience: resolved.training.patience,
                      checkpointURL: checkpointURL, isContinuation: isContinuation,
                      learningRate: resolved.training.learningRate, configuration: resolved)
        }
    } catch { fail("Training failed while saving \(checkpointURL.path): \(error.localizedDescription)") }
}

let answer = timed("Generation") {
    generate(after: options.question, newTokenCount: resolved.generation.newTokenCount,
             using: model, tokeniser: tokeniser, contextLength: resolved.model.contextLength,
             temperature: resolved.generation.temperature, topK: resolved.generation.topK,
             topP: resolved.generation.topP,
             repetitionPenalty: resolved.generation.repetitionPenalty ?? 1)
}
print("Answer:\n\(answer)\nDone.")
