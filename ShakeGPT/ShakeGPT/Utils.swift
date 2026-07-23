//
//  Utils.swift
//  ShakeGPT
//
//  Created by Bruno O
//

/// Runs an operation, prints its elapsed wall-clock time, and returns its result.
///
/// `rethrows` means this helper only throws when the operation itself throws.
func timed<T>(
    _ label: String = "",
    operation: () throws -> T
) rethrows -> T {
    print("[\(label)]: Starting...")

    let clock = ContinuousClock()
    let start = clock.now

    let result = try operation()

    print("[\(label)] Finished. Time taken: \(start.duration(to: clock.now))")
    return result
}
