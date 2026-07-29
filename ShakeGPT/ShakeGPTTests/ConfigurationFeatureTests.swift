import Foundation
import MLX
import MLXNN
import Testing

@Suite("Versioned run configuration", .serialized)
struct ConfigurationFeatureTests {
    @Test
    func test_givenPartialConfigAndCLIOverrides_whenResolved_thenCLIFileDefaultPrecedenceAndPathsAreCorrect() throws {
        try withTemporaryDirectory { root in
            let configURL = root.appendingPathComponent("config/run.json")
            try FileManager.default.createDirectory(at: configURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(
                """
                {
                  "formatVersion": 1,
                  "data": {"corpus": "../data/train.txt", "vocabulary": "../artifacts/vocab.json"},
                  "model": {"contextLength": 128},
                  "generation": {"temperature": 0.5, "topK": 12}
                }
                """.utf8
            ).write(to: configURL)

            let fileValues = try ResolvedRunConfiguration.resolve(
                configurationURL: configURL,
                overrides: overrides(),
                isTraining: true
            )
            #expect(fileValues.data.corpus == root.appendingPathComponent("data/train.txt").path)
            #expect(fileValues.data.vocabulary == root.appendingPathComponent("artifacts/vocab.json").path)
            #expect(fileValues.model.contextLength == 128)
            #expect(fileValues.model.embeddingSize == 384)
            #expect(fileValues.generation.temperature == 0.5)

            var cli = overrides(vocabulary: root.appendingPathComponent("cli-vocab.json"))
            cli.contextLength = 64
            cli.temperature = 0
            let cliValues = try ResolvedRunConfiguration.resolve(
                configurationURL: configURL,
                overrides: cli,
                isTraining: true
            )
            #expect(cliValues.data.vocabulary == root.appendingPathComponent("cli-vocab.json").path)
            #expect(cliValues.model.contextLength == 64)
            #expect(cliValues.generation.temperature == 0)
            #expect(cliValues.generation.topK == 12)

            let defaults = try ResolvedRunConfiguration.resolve(
                configurationURL: nil,
                overrides: overrides(vocabulary: root.appendingPathComponent("legacy.json")),
                isTraining: false
            )
            #expect(defaults.tokenizer.maximumModelVocabularySize == 1_025)
            #expect(defaults.model.contextLength == 256)
            #expect(defaults.model.embeddingSize == 384)
            #expect(defaults.generation.newTokenCount == 256)
        }
    }

    @Test
    func test_givenOrderedSpecialTokens_whenTokenizerRoundTrips_thenMaximumOrderAtomicityAndIDsArePreserved() throws {
        let tokenizer = BPE(
            trainOn: "abababab",
            maximumVocabularySize: 260,
            specialTokens: [
                .init(name: "endOfText", text: "<|end|>"),
                .init(name: "long", text: "<|end|>x")
            ]
        )

        #expect(tokenizer.modelVocabularySize <= 260)
        #expect(tokenizer.specialTokens.map(\.id) == [
            tokenizer.vocabularySize,
            tokenizer.vocabularySize + 1
        ])

        let text = "é<|end|>xB<|end|>é"
        let encoded = tokenizer.encode(text)
        #expect(encoded.contains(tokenizer.specialTokens[0].id))
        #expect(!encoded.contains(tokenizer.specialTokens[1].id))
        #expect(tokenizer.decode(encoded) == text)

        let json = try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(tokenizer)
        ) as! [String: Any]
        let saved = json["specialTokens"] as! [[String: Any]]
        #expect(saved.map { $0["id"] as! Int } == tokenizer.specialTokens.map(\.id))
    }

    @Test
    func test_givenLegacyVocabularyWithoutSpecialTokens_whenDecoded_thenHistoricalEOTIdentityIsPreserved() throws {
        let tokenizer = BPE(trainOn: "legacy legacy", maximumVocabularySize: 258)
        var json = try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(tokenizer)
        ) as! [String: Any]
        json.removeValue(forKey: "specialTokens")

        let legacy = try JSONDecoder().decode(
            BPE.self,
            from: JSONSerialization.data(withJSONObject: json)
        )
        let ordinaryCount = (json["idToToken"] as! [Any]).count

        #expect(legacy.endOfTextTokenID == ordinaryCount)
        #expect(legacy.modelVocabularySize == ordinaryCount + 1)
        #expect(legacy.encode(BPE.endOfTextMarker) == [ordinaryCount])
    }

    @Test
    func test_givenMatchingResolvedIdentity_whenCheckpointSavedAndLoaded_thenSidecarAndWeightsRoundTrip() throws {
        try withTemporaryDirectory { root in
            let tokenizer = tinyTokenizer("aaaaaaaa")
            let configuration = try tinyConfiguration(tokenizer: tokenizer, root: root)
            let checkpointURL = root.appendingPathComponent("model.safetensors")

            try saveCheckpoint(
                model: ShakeGPT(config: configuration.model),
                validationLoss: 1.25,
                to: checkpointURL,
                configuration: configuration
            )

            let sidecarURL = ResolvedRunConfiguration.checkpointSidecarURL(for: checkpointURL)
            #expect(FileManager.default.fileExists(atPath: sidecarURL.path))
            let sidecar = try JSONDecoder().decode(
                ResolvedRunConfiguration.self,
                from: Data(contentsOf: sidecarURL)
            )
            let (_, metadata) = try loadArraysAndMetadata(url: checkpointURL, stream: .cpu)
            #expect(metadata["compatibilityIdentity"] == configuration.compatibilityIdentity)
            #expect(sidecar.compatibilityIdentity == configuration.compatibilityIdentity)

            let loss = try loadCheckpoint(
                into: ShakeGPT(config: configuration.model),
                from: checkpointURL,
                configuration: configuration
            )
            #expect(loss == 1.25)
        }
    }

    @Test
    func test_givenMarkedCheckpoint_whenIdentityOrSidecarIsIncompatible_thenLoadRejectsBeforeMutation() throws {
        try withTemporaryDirectory { root in
            let tokenizer = tinyTokenizer("aaaaaaaa")
            let configuration = try tinyConfiguration(tokenizer: tokenizer, root: root)
            let checkpointURL = root.appendingPathComponent("model.safetensors")
            try saveCheckpoint(
                model: ShakeGPT(config: configuration.model),
                validationLoss: 1,
                to: checkpointURL,
                configuration: configuration
            )

            let changedGeometry = ShakeGPT.Config(
                vocabularySize: configuration.model.vocabularySize,
                contextLength: configuration.model.contextLength + 1,
                embeddingSize: configuration.model.embeddingSize,
                headCount: configuration.model.headCount,
                layerCount: configuration.model.layerCount,
                dropoutProbability: configuration.model.dropoutProbability,
                qkvBias: configuration.model.qkvBias
            )
            expectCheckpointError(
                "The checkpoint was created for a different model configuration."
            ) {
                try loadCheckpoint(
                    into: ShakeGPT(config: changedGeometry),
                    from: checkpointURL,
                    configuration: configuration
                )
            }

            let otherConfiguration = try tinyConfiguration(
                tokenizer: tinyTokenizer("bbbbbbbb"),
                root: root
            )
            expectCheckpointError(
                "The checkpoint was created for a different tokenizer or model identity."
            ) {
                try loadCheckpoint(
                    into: ShakeGPT(config: configuration.model),
                    from: checkpointURL,
                    configuration: otherConfiguration
                )
            }

            let sidecarURL = ResolvedRunConfiguration.checkpointSidecarURL(for: checkpointURL)
            let originalSidecar = try Data(contentsOf: sidecarURL)
            try FileManager.default.removeItem(at: sidecarURL)
            expectCheckpointError(
                "The checkpoint requires a readable resolved configuration sidecar."
            ) {
                try loadCheckpoint(
                    into: ShakeGPT(config: configuration.model),
                    from: checkpointURL,
                    configuration: configuration
                )
            }

            var sidecarJSON = try JSONSerialization.jsonObject(
                with: originalSidecar
            ) as! [String: Any]
            sidecarJSON["tokenizerIdentity"] = "mismatched"
            try JSONSerialization.data(withJSONObject: sidecarJSON).write(to: sidecarURL)
            expectCheckpointError(
                "The checkpoint was created for a different tokenizer or model identity."
            ) {
                try loadCheckpoint(
                    into: ShakeGPT(config: configuration.model),
                    from: checkpointURL,
                    configuration: configuration
                )
            }
        }
    }

    @Test
    func test_givenExistingLegacyArtifacts_whenBuiltCheckedAndGenerated_thenCLICompatibilityIsPreserved() throws {
        try withTemporaryDirectory { root in
            let currentTokenizer = tinyTokenizer("legacy legacy")
            var vocabularyJSON = try JSONSerialization.jsonObject(
                with: JSONEncoder().encode(currentTokenizer)
            ) as! [String: Any]
            vocabularyJSON.removeValue(forKey: "specialTokens")
            let legacyTokenizer = try JSONDecoder().decode(
                BPE.self,
                from: JSONSerialization.data(withJSONObject: vocabularyJSON)
            )
            let configuration = try tinyConfiguration(tokenizer: legacyTokenizer, root: root)
            let checkpointURL = root.appendingPathComponent("legacy.safetensors")
            let source = ShakeGPT(config: configuration.model)
            let configData = try JSONEncoder().encode(source.config)
            let checkpoint = try saveToData(
                arrays: Dictionary(uniqueKeysWithValues: source.parameters().flattened()),
                metadata: [
                    "config": String(decoding: configData, as: UTF8.self),
                    "validationLoss": "2.5"
                ]
            )
            try checkpoint.write(to: checkpointURL)

            let restored = ShakeGPT(config: configuration.model)
            let loss = try loadCheckpoint(
                into: restored,
                from: checkpointURL,
                configuration: configuration
            )
            #expect(loss == 2.5)
            #expect(generate(
                after: "Hi",
                newTokenCount: 0,
                using: restored,
                tokeniser: legacyTokenizer,
                contextLength: configuration.model.contextLength
            ) == "Hi")
        }
    }

    @Test
    func test_givenMalformedConfiguration_whenResolvedForTraining_thenFailurePrecedesSideEffects() throws {
        try withTemporaryDirectory { root in
            let cases = [
                "{\"data\":{\"vocabulary\":\"output.json\",\"corpus\":\"missing.txt\"}}",
                "{\"formatVersion\":2,\"data\":{\"vocabulary\":\"output.json\",\"corpus\":\"missing.txt\"}}",
                "{\"formatVersion\":1,\"data\":{\"vocabulary\":\"output.json\",\"corpus\":\"missing.txt\"},\"tokenizer\":{\"specialTokens\":[{\"name\":\"endOfText\",\"text\":\"x\"},{\"name\":\"other\",\"text\":\"x\"}]}}",
                "{\"formatVersion\":1,\"data\":{\"vocabulary\":\"output.json\",\"corpus\":\"missing.txt\"},\"tokenizer\":{\"maximumModelVocabularySize\":256}}",
                "{\"formatVersion\":1,\"data\":{\"vocabulary\":\"output.json\",\"corpus\":\"missing.txt\"},\"model\":{\"headCount\":0}}"
            ]

            for (index, json) in cases.enumerated() {
                let caseRoot = root.appendingPathComponent(String(index))
                try FileManager.default.createDirectory(at: caseRoot, withIntermediateDirectories: true)
                let configURL = caseRoot.appendingPathComponent("run.json")
                try Data(json.utf8).write(to: configURL)

                do {
                    _ = try ResolvedRunConfiguration.resolve(
                        configurationURL: configURL,
                        overrides: overrides(),
                        isTraining: true
                    )
                    Issue.record("Malformed configuration \(index) unexpectedly resolved")
                } catch {
                    #expect(!error.localizedDescription.isEmpty)
                }
                #expect(!FileManager.default.fileExists(
                    atPath: caseRoot.appendingPathComponent("output.json").path
                ))
            }
        }
    }
}

private func overrides(vocabulary: URL? = nil) -> RunOverrides {
    RunOverrides(
        corpus: nil,
        validation: nil,
        vocabulary: vocabulary,
        vocabularyCorpora: [],
        checkpoint: nil,
        resumeFrom: nil,
        maximumModelVocabularySize: nil,
        contextLength: nil,
        embeddingSize: nil,
        headCount: nil,
        layerCount: nil,
        dropoutProbability: nil,
        qkvBias: nil,
        batchSize: nil,
        epochs: nil,
        learningRate: nil,
        evalCadence: nil,
        patience: nil,
        newTokenCount: nil,
        temperature: nil,
        topK: nil,
        topP: nil
    )
}

private func tinyTokenizer(_ corpus: String) -> BPE {
    BPE(trainOn: corpus, maximumVocabularySize: 258)
}

private func tinyConfiguration(
    tokenizer: BPE,
    root: URL
) throws -> ResolvedRunConfiguration {
    var values = overrides(vocabulary: root.appendingPathComponent("vocabulary.json"))
    values.maximumModelVocabularySize = tokenizer.modelVocabularySize
    values.contextLength = 4
    values.embeddingSize = 8
    values.headCount = 2
    values.layerCount = 1
    values.dropoutProbability = 0
    values.batchSize = 1
    values.epochs = 1
    values.evalCadence = 1
    values.patience = 1
    values.newTokenCount = 0
    values.temperature = 0
    values.topK = 4
    return try ResolvedRunConfiguration.resolve(
        configurationURL: nil,
        overrides: values,
        isTraining: false,
        tokenizer: tokenizer
    )
}

private func expectCheckpointError(
    _ expectedDescription: String,
    operation: () throws -> Void
) {
    do {
        try operation()
        Issue.record("Checkpoint load unexpectedly succeeded")
    } catch {
        #expect(error.localizedDescription == expectedDescription)
    }
}

private func withTemporaryDirectory(
    _ body: (URL) throws -> Void
) throws {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: url) }
    try body(url)
}
