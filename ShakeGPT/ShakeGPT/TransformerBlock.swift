//
//  TransformerBlock.swift
//  ShakeGPT
//
//  Created by Bruno O
//

import MLX
import MLXNN

/// One pre-normalised transformer block.
///
/// Attention exchanges information between token positions. The feed-forward
/// network then transforms each position independently. Both sublayers use a
/// residual connection so they learn changes to the existing representation.
final class TransformerBlock: Module, UnaryLayer {
    @ModuleInfo private var attention: MultiHeadSelfAttention
    @ModuleInfo private var feedForward: FeedForward

    private let norm1: LayerNorm
    private let norm2: LayerNorm
    private let dropout: Dropout

    init(
        embeddingSize: Int,
        dropoutProbability: Float,
        headCount: Int,
        qkvBias: Bool
    ) {
        self._attention.wrappedValue = MultiHeadSelfAttention(
            embeddingSize: embeddingSize,
            headCount: headCount,
            dropoutProbability: dropoutProbability,
            qkvBias: qkvBias
        )

        self._feedForward.wrappedValue = FeedForward(embeddingSize: embeddingSize)

        norm1 = LayerNorm(dimensions: embeddingSize)
        norm2 = LayerNorm(dimensions: embeddingSize)
        dropout = Dropout(p: dropoutProbability)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        // Pre-normalise before attention, then preserve the original input
        // through the first residual connection.
        var x = norm1(input)
        x = attention(x)
        x = dropout(x)
        x = x + input

        // The second residual begins after attention so the feed-forward
        // network adds its contribution to the already contextualised tokens.
        let residual = x
        x = norm2(x)
        x = feedForward(x)
        x = dropout(x)
        x = x + residual

        return x
    }
}
