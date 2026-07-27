//
//  Training.swift
//  ShakeGPT
//
//  Created by Bruno O
//

import Foundation

import MLX
import MLXNN
import MLXOptimizers

/// Explains why a safetensors checkpoint cannot be restored safely.
enum CheckpointError: LocalizedError {
    case missingMetadata
    case incompatibleConfiguration

    var errorDescription: String? {
        switch self {
        case .missingMetadata:
            "The checkpoint is missing its training metadata."
        case .incompatibleConfiguration:
            "The checkpoint was created for a different model configuration."
        }
    }
}

/// Saves the model weights and the information needed to identify them.
///
/// Safetensors stores the parameters without executable code. Writing the complete
/// file atomically prevents an interrupted save from destroying the previous winner.
func saveCheckpoint(
    model: ShakeGPT,
    validationLoss: Float,
    to url: URL
) throws {
    let parameters = Dictionary(
        uniqueKeysWithValues: model.parameters().flattened()
    )
    let encodedConfig = try JSONEncoder().encode(model.config)
    let metadata = [
        "config": String(decoding: encodedConfig, as: UTF8.self),
        "validationLoss": String(validationLoss)
    ]

    let checkpoint = try saveToData(
        arrays: parameters,
        metadata: metadata
    )

    try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try checkpoint.write(to: url, options: .atomic)
}

/// Loads weights into an already constructed model after verifying its architecture.
///
/// The BPE vocabulary must also be the same one used during training because the
/// checkpoint stores numerical token relationships, not the meaning of each token ID.
@discardableResult
func loadCheckpoint(
    into model: ShakeGPT,
    from url: URL
) throws -> Float {
    let (arrays, metadata) = try loadArraysAndMetadata(
        url: url,
        stream: .cpu
    )

    guard
        let configText = metadata["config"],
        let validationLossText = metadata["validationLoss"],
        let configData = configText.data(using: .utf8),
        let validationLoss = Float(validationLossText)
    else {
        throw CheckpointError.missingMetadata
    }

    let savedConfig = try JSONDecoder().decode(
        ShakeGPT.Config.self,
        from: configData
    )

    guard savedConfig == model.config else {
        throw CheckpointError.incompatibleConfiguration
    }

    let parameters = ModuleParameters.unflattened(arrays)
    try model.update(parameters: parameters, verify: .all)
    eval(model)
    model.train(false)

    return validationLoss
}

/// Creates one random mini-batch for next-token prediction.
///
/// Each input row contains `contextLength` consecutive token IDs. Its matching
/// target row contains the same sequence shifted one position forward, so every
/// input position is paired with the token that should follow it.
///
/// Both returned tensors have shape `[batchSize, contextLength]` and use
/// `Int32` values because token IDs are indices rather than continuous values.
func makeBatch(
    from tokenIDs: [Int],
    contextLength: Int,
    batchSize: Int
) -> (
    inputs: MLXArray,
    targets: MLXArray
) {
    precondition(contextLength > 0, "Context length must be positive")
    precondition(batchSize > 0, "Batch size must be positive")
    precondition(
        tokenIDs.count > contextLength,
        "The corpus must contain more tokens than the context length"
    )

    var inputs: [Int32] = []
    var targets: [Int32] = []

    for _ in 0..<batchSize {
        // Sampling a fresh start lets each training step see different text.
        let startIndex = Int.random(
            in: 0..<(tokenIDs.count - contextLength)
        )

        // Append complete rows in row-major order before giving them a 2D shape.
        inputs.append(
            contentsOf: tokenIDs[
                startIndex ..< startIndex + contextLength
            ].map(Int32.init)
        )
        targets.append(
            contentsOf: tokenIDs[
                startIndex + 1 ... startIndex + contextLength
            ].map(Int32.init)
        )
    }

    let shape = [batchSize, contextLength]

    return (
        inputs: MLXArray(inputs, shape),
        targets: MLXArray(targets, shape)
    )
}

/// Measures how far the model's predictions are from the expected next tokens.
///
/// The model produces one score for every vocabulary token at every position.
/// Cross-entropy rewards a high score for the correct next token, then `.mean`
/// combines all positions and batch rows into one number that training can minimise.
func loss(model: ShakeGPT, inputs: MLXArray, target: MLXArray) -> MLXArray {
    let logits = model(inputs)

    return crossEntropy(logits: logits, targets: target, reduction: .mean)
}

/// Estimates the model's loss by averaging several fixed validation batches.
///
/// Reusing the same batches makes every validation result comparable: the model
/// changes between evaluations, but the exam questions do not. The model temporarily
/// enters evaluation mode so dropout does not randomly change the score.
func estimateLoss(
    model: ShakeGPT,
    on batches: [(inputs: MLXArray, targets: MLXArray)]
) -> Float {
    precondition(!batches.isEmpty, "Validation batches cannot be empty")

    let wasTraining = model.training
    model.train(false)
    // Leave the model in exactly the mode in which we found it.
    defer { model.train(wasTraining) }

    var total: Float = 0

    for batch in batches {
        let batchLoss = loss(
            model: model,
            inputs: batch.inputs,
            target: batch.targets
        )

        eval(batchLoss)
        total += batchLoss.item(Float.self)
    }

    return total / Float(batches.count)
}

/// Trains the model to predict the next token and watches a held-out validation set.
///
/// Training updates the model's parameters. Validation never updates them: it only
/// checks whether the model is learning patterns that also work on unseen text.
/// If validation stops improving, training ends early and restores the best model
/// observed during this run.
func train(
    model: ShakeGPT,
    on tokenIDs: [Int],
    validatedWith: [Int]?,
    tokeniser: BPE,
    contextLength: Int,
    batchSize: Int,
    epochs: Int,
    validationPercentage: Float = 0.1,
    evalCadence: Int = 10,
    evaluationText: String = "Hello, world!",
    patience: Int = 20,
    minimumImprovement: Float = 0.005,
    validationBatchCount: Int = 20,
    checkpointURL: URL? = nil,
    isContinuation: Bool = false
) throws {
    precondition(epochs > 0, "Number of epochs must be positive")
    precondition(
        validationPercentage > 0 && validationPercentage < 1,
        "Validation percentage must be between zero and one"
    )
    precondition(evalCadence > 0, "Evaluation cadence must be positive")
    precondition(!evaluationText.isEmpty, "Evaluation text cannot be empty")
    precondition(patience > 0, "Patience must be positive")
    precondition(minimumImprovement >= 0, "Minimum improvement cannot be negative")
    precondition(validationBatchCount > 0, "Validation batch count must be positive")

    var trainingTokens: [Int] = []
    var validationTokens: [Int] = []

    if let validatedWith {
        trainingTokens = tokenIDs
        validationTokens = validatedWith
    } else {
        // Keep the final part of the corpus separate so training never samples from it.
        let splitIndex = Int(Double(tokenIDs.count) * (1 - Double(validationPercentage)))

        trainingTokens = Array(tokenIDs[..<splitIndex])
        validationTokens = Array(tokenIDs[splitIndex...])
    }

    precondition(
        trainingTokens.count > contextLength,
        "Training data must be longer than the context"
    )
    precondition(
        validationTokens.count > contextLength,
        "Validation data must be longer than the context"
    )

    // A random batch processes `batchSize * contextLength` next-token
    // predictions. Once that total is approximately the size of the training
    // corpus, we call it one effective epoch. It is approximate because random
    // sampling may revisit some windows while skipping others.
    let predictionsPerStep = batchSize * contextLength
    let stepsPerEpoch = Int(
        ceil(
            Double(trainingTokens.count - 1)
                / Double(predictionsPerStep)
        )
    )
    let totalSteps = epochs * stepsPerEpoch

    print("Training tokens:", trainingTokens.count)
    print("Steps per effective epoch:", stepsPerEpoch)
    print("Total training steps:", totalSteps)

    // AdamW decides how the gradients change each parameter.
    let optimiser = AdamW(learningRate: 0.0003, weightDecay: 0.1)

    // Automatic differentiation gives us both the loss and its gradients.
    let lossAndGrad = valueAndGrad(model: model, loss)
    var evaluationCount: Int = 0

    // Sample the validation exam once, then reuse it throughout this training run.
    let validationBatches = (0..<validationBatchCount).map { _ in
        makeBatch(
            from: validationTokens,
            contextLength: contextLength,
            batchSize: batchSize
        )
    }

    // These values remember the best-performing version of the model.
    var bestValidationLoss = Float.infinity
    var bestParameters: ModuleParameters?
    var evaluationsWithoutImprovement = 0

    if isContinuation {
        // Establish how well the loaded checkpoint performs on this run's fixed
        // validation exam before allowing new training to replace it on disk.
        bestValidationLoss = estimateLoss(
            model: model,
            on: validationBatches
        )
        bestParameters = model.parameters()
        print(
            "Loaded checkpoint validation loss:",
            bestValidationLoss
        )
    }

    for step in 1...totalSteps {
        model.train(true)
        let trainingBatch = makeBatch(from: trainingTokens, contextLength: contextLength, batchSize: batchSize)

        // Back-propagation computes the direction in which every parameter should move.
        let (batchLoss, gradients) = lossAndGrad(model, trainingBatch.inputs, trainingBatch.targets)
        optimiser.update(model: model, gradients: gradients)

        // Materialise MLX's lazy calculations before beginning the next step.
        eval(model, optimiser)

        // Validation is more expensive, so run it only at the chosen cadence.
        if step % evalCadence == 0 || step == totalSteps {
            model.train(false)

            let validationLoss = estimateLoss(model: model, on: validationBatches)

            // Generated text is not a scientific metric, but it is a useful human
            // glimpse of what the model currently believes plausible text looks like.
            let evaluation = generate(
                after: evaluationText,
                newTokenCount: 16,
                using: model,
                tokeniser: tokeniser,
                contextLength: contextLength
            ).replacingOccurrences(of: "\n", with: " ")
                .replacingOccurrences(of: "\r", with: " ")

            if evaluationCount == 0 {
                print("| Step | Epoch | Training Loss | Validation Loss | Generated |")
                print("|------|-------|---------------|-----------------|-----------|")
            }

            let completedEpochs = Double(step) / Double(stepsPerEpoch)
            let epochText = completedEpochs.formatted(
                .number.precision(.fractionLength(2))
            )

            print(
                "| \(step)/\(totalSteps) | \(epochText)/\(epochs) |",
                "\(batchLoss.item(Float.self)) | \(validationLoss) | \(evaluation) |"
            )
            evaluationCount += 1

            if validationLoss < bestValidationLoss - minimumImprovement {
                bestValidationLoss = validationLoss
                // Preserve the parameters from this step before training changes them.
                bestParameters = model.parameters()
                evaluationsWithoutImprovement = 0

                if let checkpointURL {
                    try saveCheckpoint(
                        model: model,
                        validationLoss: validationLoss,
                        to: checkpointURL
                    )
                }
            } else {
                // Patience counts validation checks, not individual training steps.
                evaluationsWithoutImprovement += 1
            }

            if evaluationsWithoutImprovement >= patience {
                print(
                    "Stopping early. Best validation loss",
                    bestValidationLoss
                )

                break
            }
        }
    }

    // The final step is not necessarily the best one, so return to the saved winner.
    if let bestParameters {
        model.update(parameters: bestParameters)
        print("Restored best model.")
    }
    model.train(false)
}
