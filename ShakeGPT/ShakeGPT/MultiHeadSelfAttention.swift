//
//  MultiHeadSelfAttention.swift
//  ShakeGPT
//
//  Created by Bruno O
//


import MLX
import MLXNN

/// Multi-head self-attention lets several attention heads examine a sequence
/// in parallel.
///
/// Every embedded token is projected into three different representations:
/// - a query describing what the token is looking for;
/// - a key describing what the token offers;
/// - a value containing the information the token can contribute.
///
/// Each `[B, C, D]` projection is split into `[B, H, C, headSize]`, where
/// `D == H * headSize`. Every head then computes its own attention matrix.
/// The weights mix the value vectors before the heads are joined back into
/// `[B, C, D]`.
final class MultiHeadSelfAttention: Module, UnaryLayer {
    @ModuleInfo private var queryProjection: Linear
    @ModuleInfo private var keyProjection: Linear
    @ModuleInfo private var valueProjection: Linear
    @ModuleInfo private var outputProjection: Linear

    private let embeddingSize: Int
    private let headCount: Int
    private let headSize: Int

    private let dropout: Dropout

    init(
        embeddingSize: Int,
        headCount: Int,
        dropoutProbability: Float = 0.1,
        qkvBias: Bool = false
    ) {
        precondition(embeddingSize > 0, "Embedding size must be positive")
        precondition(headCount > 0, "Head count must be positive")
        precondition(
            embeddingSize.isMultiple(of: headCount),
            "Embedding size must be divisible by the number of heads"
        )

        self.embeddingSize = embeddingSize
        self.headCount = headCount
        self.headSize = embeddingSize / headCount

        self._queryProjection.wrappedValue = Linear(embeddingSize, embeddingSize, bias: qkvBias)
        self._keyProjection.wrappedValue = Linear(embeddingSize, embeddingSize, bias: qkvBias)
        self._valueProjection.wrappedValue = Linear(embeddingSize, embeddingSize, bias: qkvBias)
        self._outputProjection.wrappedValue = Linear(embeddingSize, embeddingSize, bias: false)

        dropout = Dropout(p: dropoutProbability)
    }

    /// Projects the input into Q, K and V, then divides each into heads.
    ///
    /// Input shape: `[B, C, D]`
    /// Returned shapes: `[B, H, C, headSize]`
    private func projections(of input: MLXArray) -> (
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray
    ) {
        precondition(input.shape.count == 3, "Expected input shaped [batch, context, embedding]")
        precondition(input.shape[2] == embeddingSize, "Input embedding size does not match the attention layer")

        return (
            queries: splitIntoHeads(queryProjection(input)),
            keys: splitIntoHeads(keyProjection(input)),
            values: splitIntoHeads(valueProjection(input))
        )
    }

    /// Compares every query with every key inside the same head.
    ///
    /// Transposing the keys from `[B, H, C, headSize]` to
    /// `[B, H, headSize, C]` makes matrix multiplication produce
    /// `[B, H, C, C]`: one attention matrix per batch item and head.
    private func scores(
        queries: MLXArray,
        keys: MLXArray
    ) -> MLXArray {
        precondition(
            queries.shape == keys.shape && queries.shape.count == 4,
            "Queries and keys must have matching [B, H, C, headSize] shapes"
        )

        let transposedKeys = keys.transposed(0, 1, 3, 2)

        // Each dot product spans one head, so scale by headSize rather than D.
        let scale = Float(headSize).squareRoot()

        return queries.matmul(transposedKeys) / scale
    }

    /// Masks future tokens and converts the remaining scores into weights.
    ///
    /// The `[C, C]` causal mask broadcasts across both batch and head dimensions.
    /// Softmax makes each row add up to one before dropout; dropout may change
    /// an individual row's sum during training while preserving it on average.
    private func weights(from scores: MLXArray) -> MLXArray {
        precondition(
            scores.shape.count == 4,
            "Expected scores shaped [batch, head, query, key]"
        )

        let contextLength = scores.shape[2]

        let futureMask = MLX.triu(
            MLX.ones([contextLength, contextLength]),
            k: 1
        )

        // Future positions become zero after softmax because exp(-infinity) = 0.
        let maskedScores = MLX.which(
            futureMask .> 0,
            -Float.infinity,
            scores
        )
        let weights = MLX.softmax(maskedScores, axis: -1)

        return dropout(weights)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        // Q, K and V: [B, H, C, headSize]
        let projected = projections(of: input)

        // Scores: [B, H, C, C]
        let attentionScores = scores(
            queries: projected.queries,
            keys: projected.keys
        )

        // Masked, normalized and dropped-out weights: [B, H, C, C]
        let attentionWeights = weights(
            from: attentionScores
        )

        // Each head uses its weights to mix the value vectors.
        // [B, H, C, C] × [B, H, C, headSize]
        // becomes [B, H, C, headSize].
        let context = attentionWeights.matmul(
            projected.values
        )

        let batchSize = input.shape[0]
        let contextLength = input.shape[1]

        // Put context before heads:
        // [B, H, C, headSize] → [B, C, H, headSize]
        //
        // Then join H × headSize back into D:
        // [B, C, H, headSize] → [B, C, D]
        let joinedHeads = context
            .transposed(0, 2, 1, 3)
            .reshaped([
                batchSize,
                contextLength,
                embeddingSize
            ])

        // Allow the model to mix information produced by different heads.
        return outputProjection(joinedHeads)
    }
}

private extension MultiHeadSelfAttention {
    /// Changes `[B, C, D]` into `[B, H, C, headSize]`.
    ///
    /// Reshape first exposes the head dimension. Transpose then places heads
    /// before the context dimension so every head can run attention in parallel.
    func splitIntoHeads(_ input: MLXArray) -> MLXArray {
        let batchSize = input.shape[0]
        let contextLength = input.shape[1]

        return input
            .reshaped([
                batchSize,
                contextLength,
                headCount,
                headSize
            ])
            .transposed(0, 2, 1, 3)
    }
}
