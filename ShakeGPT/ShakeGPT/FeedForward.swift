//
//  FeedForward.swift
//  ShakeGPT
//
//  Created by Bruno O
//

import MLX
import MLXNN

/// Lets each token transform its own representation after attention has mixed
/// information between positions.
///
/// The first linear layer expands `D` to `4D`, GELU introduces non-linearity,
/// and the second layer projects the result back to `D` for the residual path.
final class FeedForward: Module, UnaryLayer {
    private let layers: Sequential

    init(embeddingSize: Int) {
        precondition(embeddingSize > 0, "Embedding size must be positive")

        layers = Sequential {
            Linear(embeddingSize, 4 * embeddingSize)
            GELU()
            Linear(4 * embeddingSize, embeddingSize)
        }
    }

    /// Preserves the input shape `[B, C, D]`.
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        layers(input)
    }
}
