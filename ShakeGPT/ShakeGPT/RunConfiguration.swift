//
//  RunConfiguration.swift
//  ShakeGPT
//

import CryptoKit
import Foundation

enum RunConfigurationError: LocalizedError {
    case invalid(String)

    var errorDescription: String? {
        if case let .invalid(message) = self { return message }
        return nil
    }
}

/// The optional values decoded directly from a version-1 JSON file.
///
/// Keeping these properties optional lets the resolver combine three layers:
/// command-line overrides first, JSON values second, and built-in defaults last.
/// Paths remain strings here so relative JSON paths can later be resolved from
/// the configuration file's directory rather than the process's directory.
struct RunConfiguration: Codable {
    struct DataSettings: Codable {
        var corpus: String?
        var validation: String?
        var vocabularyCorpora: [String]?
        var vocabulary: String?
        var checkpoint: String?
        var resumeFrom: String?
    }
    struct TokenizerSettings: Codable {
        var maximumModelVocabularySize: Int?
        var specialTokens: [BPE.SpecialTokenDefinition]?
    }
    struct ModelSettings: Codable {
        var contextLength: Int?
        var embeddingSize: Int?
        var headCount: Int?
        var layerCount: Int?
        var dropoutProbability: Float?
        var qkvBias: Bool?
    }
    struct TrainingSettings: Codable {
        var batchSize: Int?
        var epochs: Int?
        var learningRate: Float?
        var evalCadence: Int?
        var patience: Int?
    }
    struct GenerationSettings: Codable {
        var newTokenCount: Int?
        var temperature: Float?
        var topK: Int?
        var topP: Float?
        var repetitionPenalty: Float?
    }

    let formatVersion: Int
    var data: DataSettings?
    var tokenizer: TokenizerSettings?
    var model: ModelSettings?
    var training: TrainingSettings?
    var generation: GenerationSettings?

    static func load(from url: URL) throws -> (RunConfiguration, URL) {
        let configuration = try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
        guard configuration.formatVersion == 1 else {
            throw RunConfigurationError.invalid("Unsupported configuration format version \(configuration.formatVersion); expected 1.")
        }
        return (configuration, url.deletingLastPathComponent())
    }
}

/// Values explicitly supplied on the command line.
///
/// They remain optional so the resolver can distinguish "not supplied" from a
/// real override and fall back to the JSON file or a default.
struct RunOverrides {
    var corpus: URL?
    var validation: URL?
    var vocabulary: URL?
    var vocabularyCorpora: [URL]
    var checkpoint: URL?
    var resumeFrom: URL?
    var maximumModelVocabularySize: Int?
    var contextLength: Int?
    var embeddingSize: Int?
    var headCount: Int?
    var layerCount: Int?
    var dropoutProbability: Float?
    var qkvBias: Bool?
    var batchSize: Int?
    var epochs: Int?
    var learningRate: Float?
    var evalCadence: Int?
    var patience: Int?
    var newTokenCount: Int?
    var temperature: Float?
    var topK: Int?
    var topP: Float?
    var repetitionPenalty: Float? = nil
}

/// The complete, validated settings used for one run.
///
/// Unlike `RunConfiguration`, every required value has now been chosen and
/// every path is absolute. Saving this beside a checkpoint records the model
/// shape and tokenizer identity needed to decide whether it is safe to reload.
struct ResolvedRunConfiguration: Codable, Equatable {
    struct DataSettings: Codable, Equatable {
        let corpus: String?
        let validation: String?
        let vocabularyCorpora: [String]
        let vocabulary: String
        let checkpoint: String
        let resumeFrom: String?
    }
    struct TokenizerSettings: Codable, Equatable {
        let maximumModelVocabularySize: Int
        let specialTokens: [BPE.SpecialToken]
    }
    struct TrainingSettings: Codable, Equatable {
        let batchSize: Int
        let epochs: Int
        let learningRate: Float
        let evalCadence: Int
        let patience: Int
    }
    struct GenerationSettings: Codable, Equatable {
        let newTokenCount: Int
        let temperature: Float
        let topK: Int
        let topP: Float?
        let repetitionPenalty: Float?
    }

    let formatVersion: Int
    let data: DataSettings
    let tokenizer: TokenizerSettings
    let model: ShakeGPT.Config
    let training: TrainingSettings
    let generation: GenerationSettings
    let tokenizerIdentity: String
    let compatibilityIdentity: String

    static func resolve(
        configurationURL: URL?,
        overrides: RunOverrides,
        isTraining: Bool,
        tokenizer: BPE? = nil
    ) throws -> ResolvedRunConfiguration {
        let loaded = try configurationURL.map { try RunConfiguration.load(from: $0) }
        let file = loaded?.0
        let base = loaded?.1
        func path(_ cli: URL?, _ json: String?) -> URL? {
            if let cli { return cli }
            guard let json else { return nil }
            return URL(fileURLWithPath: json, relativeTo: base).standardizedFileURL
        }
        func value<T>(_ cli: T?, _ json: T?, _ fallback: T) -> T { cli ?? json ?? fallback }

        let specialDefinitions = file?.tokenizer?.specialTokens ?? [.init(name: "endOfText", text: BPE.endOfTextMarker)]
        try validateSpecialTokens(specialDefinitions)
        let maximum = value(overrides.maximumModelVocabularySize, file?.tokenizer?.maximumModelVocabularySize, 1_025)
        let contextLength = value(overrides.contextLength, file?.model?.contextLength, 256)
        let embeddingSize = value(overrides.embeddingSize, file?.model?.embeddingSize, 384)
        let headCount = value(overrides.headCount, file?.model?.headCount, 6)
        let layerCount = value(overrides.layerCount, file?.model?.layerCount, 6)
        let dropout = value(overrides.dropoutProbability, file?.model?.dropoutProbability, 0.2)
        let qkvBias = value(overrides.qkvBias, file?.model?.qkvBias, false)
        let batchSize = value(overrides.batchSize, file?.training?.batchSize, 8)
        let epochs = value(overrides.epochs, file?.training?.epochs, 10)
        let learningRate = value(overrides.learningRate, file?.training?.learningRate, 0.0003)
        let evalCadence = value(overrides.evalCadence, file?.training?.evalCadence, 100)
        let patience = value(overrides.patience, file?.training?.patience, 20)
        let newTokenCount = value(overrides.newTokenCount, file?.generation?.newTokenCount, contextLength)
        let temperature = value(overrides.temperature, file?.generation?.temperature, 0.7)
        let topK = value(overrides.topK, file?.generation?.topK, 40)
        let topP = overrides.topP ?? file?.generation?.topP
        let repetitionPenalty = value(
            overrides.repetitionPenalty,
            file?.generation?.repetitionPenalty,
            1.0
        )
        guard let vocabularyURL = path(overrides.vocabulary, file?.data?.vocabulary) else {
            throw RunConfigurationError.invalid("--vocabulary is required unless supplied by --config.")
        }
        let vocabularyCorpora = overrides.vocabularyCorpora.isEmpty
            ? (file?.data?.vocabularyCorpora ?? []).map { URL(fileURLWithPath: $0, relativeTo: base).standardizedFileURL }
            : overrides.vocabularyCorpora
        let corpusURL = path(overrides.corpus, file?.data?.corpus)
        if isTraining && corpusURL == nil {
            throw RunConfigurationError.invalid("--corpus is required when using --train unless supplied by --config.")
        }
        guard maximum >= 256 + specialDefinitions.count,
              contextLength > 0, embeddingSize > 0, headCount > 0, layerCount > 0,
              embeddingSize.isMultiple(of: headCount), dropout.isFinite, dropout >= 0, dropout < 1,
              batchSize > 0, epochs > 0, evalCadence > 0, patience > 0,
              learningRate.isFinite, learningRate > 0, newTokenCount >= 0,
              temperature.isFinite, temperature >= 0, topK > 0,
              topP.map({ $0.isFinite && $0 > 0 && $0 <= 1 }) ?? true,
              repetitionPenalty.isFinite, repetitionPenalty >= 1 else {
            throw RunConfigurationError.invalid("Configuration has an invalid numeric value.")
        }

        let defaultCheckpoint = vocabularyURL.deletingLastPathComponent().appendingPathComponent(vocabularyURL.deletingPathExtension().lastPathComponent + "-best.safetensors")
        let checkpointURL = path(overrides.checkpoint, file?.data?.checkpoint) ?? defaultCheckpoint
        let resolvedSpecialTokens = tokenizer?.specialTokens ?? specialDefinitions.enumerated().map {
            .init(name: $0.element.name, text: $0.element.text, id: 256 + $0.offset)
        }
        let model = ShakeGPT.Config(vocabularySize: tokenizer?.modelVocabularySize ?? maximum, contextLength: contextLength, embeddingSize: embeddingSize, headCount: headCount, layerCount: layerCount, dropoutProbability: dropout, qkvBias: qkvBias)
        let tokenizerIdentity = tokenizer?.identityDigest ?? digest(resolvedSpecialTokens)
        let compatibilityIdentity = digest(model, tokenizerIdentity)
        return .init(
            formatVersion: 1,
            data: .init(corpus: corpusURL?.path, validation: path(overrides.validation, file?.data?.validation)?.path, vocabularyCorpora: vocabularyCorpora.map(\.path), vocabulary: vocabularyURL.path, checkpoint: checkpointURL.path, resumeFrom: path(overrides.resumeFrom, file?.data?.resumeFrom)?.path),
            tokenizer: .init(maximumModelVocabularySize: maximum, specialTokens: resolvedSpecialTokens),
            model: model,
            training: .init(batchSize: batchSize, epochs: epochs, learningRate: learningRate, evalCadence: evalCadence, patience: patience),
            generation: .init(
                newTokenCount: newTokenCount,
                temperature: temperature,
                topK: topK,
                topP: topP,
                repetitionPenalty: repetitionPenalty
            ),
            tokenizerIdentity: tokenizerIdentity,
            compatibilityIdentity: compatibilityIdentity
        )
    }

    static func checkpointSidecarURL(for checkpointURL: URL) -> URL {
        checkpointURL.deletingPathExtension().appendingPathExtension("config.json")
    }

    private static func validateSpecialTokens(_ tokens: [BPE.SpecialTokenDefinition]) throws {
        guard !tokens.isEmpty, tokens.filter({ $0.name == "endOfText" }).count == 1,
              Set(tokens.map(\.name)).count == tokens.count, Set(tokens.map(\.text)).count == tokens.count,
              tokens.allSatisfy({ !$0.name.isEmpty && !$0.text.isEmpty }) else {
            throw RunConfigurationError.invalid("Special tokens must have unique non-empty names and spellings and exactly one endOfText token.")
        }
    }

    private static func digest<T: Encodable>(_ value: T) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try! encoder.encode(value)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func digest(_ model: ShakeGPT.Config, _ tokenizerIdentity: String) -> String {
        struct Identity: Codable { let model: ShakeGPT.Config; let tokenizerIdentity: String }
        return digest(Identity(model: model, tokenizerIdentity: tokenizerIdentity))
    }
}
