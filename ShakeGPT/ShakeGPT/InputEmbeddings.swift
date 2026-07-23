//
//  InputEmbeddings.swift
//  ShakeGPT
//
//  Created by Bruno O
//

import MLX
import MLXNN

final class InputEmbeddings: Module, UnaryLayer {
    @ModuleInfo var tokenEmbeddings: Embedding
    @ModuleInfo var positionEmbedding: Embedding


    private let maximumContextLength: Int

    init(vocabularySize: Int,
         maximumContextLength: Int,
         embeddingSize: Int) {

        precondition(vocabularySize > 0)
        precondition(maximumContextLength > 0)
        precondition(embeddingSize > 0)

        self.maximumContextLength = maximumContextLength

        self._tokenEmbeddings.wrappedValue = Embedding(embeddingCount: vocabularySize, dimensions: embeddingSize)
        self._positionEmbedding.wrappedValue = Embedding(embeddingCount: maximumContextLength, dimensions: embeddingSize)
    }

    func callAsFunction(_ tokenIDs: MLXArray) -> MLXArray {
        precondition(tokenIDs.shape.count == 2,
                     "Expected token IDs shaped [batch, context]")

        let contextLength = tokenIDs.shape[1]

        precondition(
            contextLength <= maximumContextLength,
            "Input exceeds the maximum context length"
        )

        let positions = MLXArray(0..<contextLength)
        let tokens = tokenEmbeddings(tokenIDs)
        let positionsInSpace = positionEmbedding(positions)

        return tokens + positionsInSpace
    }
}
