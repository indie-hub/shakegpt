//
//  ShakeGPT.swift
//  ShakeGPT
//
//  Created by Bruno O
//

import MLX
import MLXNN

/// A decoder-only transformer that predicts one vocabulary logit per position.
///
/// Token IDs `[B, C]` become embeddings `[B, C, D]`, pass through the
/// transformer blocks, and leave as logits `[B, C, V]`.
final class ShakeGPT: Module, UnaryLayer {
    /// Collects the architectural choices needed to construct a model.
    ///
    /// `Codable` will later let the same configuration be saved beside trained
    /// weights and reconstructed for generation.
    struct Config: Codable {
        let vocabularySize: Int
        let contextLength: Int
        let embeddingSize: Int
        let headCount: Int
        let layerCount: Int
        let dropoutProbability: Float
        let qkvBias: Bool
    }

    @ModuleInfo private var outputHead: Linear

    private let embeddings: InputEmbeddings
    private let normLayer: LayerNorm
    private let dropout: Dropout
    private let transformerBlocks: Sequential

    init(config: Config) {
        precondition(config.vocabularySize > 0, "Vocabulary size must be positive")
        precondition(config.contextLength > 0, "Context length must be positive")
        precondition(config.embeddingSize > 0, "Embedding size must be positive")
        precondition(config.headCount > 0, "Head count must be positive")
        precondition(config.layerCount > 0, "Layer count must be positive")
        precondition(
            config.embeddingSize.isMultiple(of: config.headCount),
            "Embedding size must be divisible by head count"
        )
        precondition(
            (0..<1).contains(config.dropoutProbability),
            "Dropout probability must be between zero and one"
        )

        embeddings = InputEmbeddings(
            vocabularySize: config.vocabularySize,
            maximumContextLength: config.contextLength,
            embeddingSize: config.embeddingSize
        )

        self._outputHead.wrappedValue = Linear(
            config.embeddingSize,
            config.vocabularySize,
            bias: false
        )

        normLayer = LayerNorm(dimensions: config.embeddingSize)
        dropout = Dropout(p: config.dropoutProbability)

        transformerBlocks = Sequential {
            for _ in 0..<config.layerCount {
                TransformerBlock(
                    embeddingSize: config.embeddingSize,
                    dropoutProbability: config.dropoutProbability,
                    headCount: config.headCount,
                    qkvBias: config.qkvBias
                )
            }
        }
    }

    /// Runs the complete forward pass without choosing tokens or probabilities.
    ///
    /// Returning raw logits keeps this method suitable for both cross-entropy
    /// training and generation.
    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        let embeddedTokens = embeddings(tokens)
        let droppedEmbeddings = dropout(embeddedTokens)

        let transformed = transformerBlocks(droppedEmbeddings)
        let finalNorm = normLayer(transformed)

        return outputHead(finalNorm)
    }
}

/// Lightweight model diagnostics used by the command-line tutorial.
extension ShakeGPT {
    /// Total number of scalar values that gradient descent can update.
    var parameterCount: Int {
        let values = trainableParameters().flattenedValues()

        return values.reduce(0) {
            $0 + $1.size
        }
    }

    /// Bytes occupied by trainable parameters, excluding runtime training state.
    var parameterBytes: Int {
        let values = trainableParameters().flattenedValues()

        return values.reduce(0) {
            $0 + ($1.size * $1.dtype.size)
        }
    }
}
