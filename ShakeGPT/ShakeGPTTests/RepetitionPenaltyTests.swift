import MLX
import Testing

extension ConfigurationFeatureTests {
    @Test
    func repetitionPenaltyReducesRecentOrdinaryTokenLogits() {
        let adjusted = applyingRepetitionPenalty(
            to: MLXArray([Float(2), -2, 3, 4]),
            tokenIDs: [0, 1, 2],
            excluding: [2],
            penalty: 2
        )

        eval(adjusted)
        #expect(adjusted.asArray(Float.self) == [1, -4, 3, 4])
    }
}
