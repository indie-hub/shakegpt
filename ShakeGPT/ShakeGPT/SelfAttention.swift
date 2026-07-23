//
//  SelfAttention.swift
//  ShakeGPT
//
//  Created by Bruno O
//


import MLX
import MLXNN

/// The first, single-head version of self-attention used by ShakeGPT.
///
/// Every embedded token is projected into three different representations:
/// - a query describing what the token is looking for;
/// - a key describing what the token offers;
/// - a value containing the information the token can contribute.
///
/// Queries are compared with keys to produce attention scores. Softmax then
/// converts those scores into weights. A future step will use the weights to
/// combine the values into context vectors.
final class SelfAttention: Module {
    @ModuleInfo var queryProjection: Linear
    @ModuleInfo var keyProjection: Linear
    @ModuleInfo var valueProjection: Linear
    @ModuleInfo var dropout: Dropout

    private let embeddingSize: Int

    init(embeddingSize: Int, dropoutProbability: Float = 0.1) {
        precondition(embeddingSize > 0, "Embedding size must be positive")

        self.embeddingSize = embeddingSize

        self._queryProjection.wrappedValue = Linear(embeddingSize, embeddingSize, bias: false)
        self._keyProjection.wrappedValue = Linear(embeddingSize, embeddingSize, bias: false)
        self._valueProjection.wrappedValue = Linear(embeddingSize, embeddingSize, bias: false)
        self._dropout.wrappedValue = Dropout(p: dropoutProbability)
    }

    /// Projects the input `[batch, context, embedding]` tensor into Q, K and V.
    ///
    /// The three tensors keep the same shape as the input, but each is produced
    /// by a different learned linear transformation.
    func projections(of input: MLXArray) -> (
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray
    ) {
        precondition(input.shape.count == 3, "Expected input shaped [batch, context, embedding]")
        precondition(input.shape[2] == embeddingSize, "Input embedding size does not match the attention layer")

        return(
            queries: self.queryProjection(input),
            keys: self.keyProjection(input),
            values: self.valueProjection(input)
        )
    }

    /// Compares every query with every key.
    ///
    /// Transposing the keys changes their shape from `[B, C, D]` to
    /// `[B, D, C]`, so matrix multiplication produces `[B, C, C]`: one score
    /// for every query-key pair in each sequence.
    func scores(
        queries: MLXArray,
        keys: MLXArray
    ) -> MLXArray {
        precondition(
            queries.shape == keys.shape && queries.shape.count == 3,
            "Queries and keys must have matching [batch, context, embedding] shapes"
        )

        let transposedKeys = keys.transposed(0, 2, 1)

        // Scaling prevents large dot products from making softmax too extreme.
        let scale = Float(embeddingSize).squareRoot()

        return queries.matmul(transposedKeys) / scale
    }

    /// Converts each query's scores into weights that add up to one.
    ///
    /// `axis: -1` means softmax runs across the keys, the final dimension.
    /// Future positions are replaced with negative infinity before softmax, so
    /// their resulting attention weights are zero.
    func weights(from scores: MLXArray) -> MLXArray {
        precondition(
            scores.shape.count == 3,
            "Expected scores shaped [batch, query, key]"
        )

        let contextLength = scores.shape[2]

        let futureMask = MLX.triu(
            MLX.ones([contextLength, contextLength]),
            k: 1
        )

        let maskedScores = MLX.which(futureMask .> 0, -Float.infinity, scores)
        let weights = MLX.softmax(maskedScores, axis: -1)

        return dropout(weights)
    }
}
