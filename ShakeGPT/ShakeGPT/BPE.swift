//
//  BPE.swift
//  ShakeGPT
//
//  Created by Bruno O
//

import CryptoKit
import Foundation

/// A small byte-level Byte Pair Encoding (BPE) tokenizer.
///
/// BPE begins with one token for every possible UTF-8 byte. During training it
/// repeatedly finds the most common adjacent pair and joins that pair into a
/// larger token. Encoding starts from bytes and replays those learned joins.
///
/// Configured special tokens are different: their spellings are recognized as
/// indivisible markers and receive reserved IDs after the ordinary vocabulary.
/// Starting from bytes means ordinary text never needs an "unknown" token.
struct BPE: Codable {
    /// The historical document-boundary spelling used by existing vocabularies.
    static let endOfTextMarker = "<|endoftext|>"

    /// One ordinary BPE token containing one or more UTF-8 bytes.
    typealias Token = [UInt8]

    /// Two adjacent ordinary tokens that may be joined.
    struct Pair: Hashable, Codable {
        let left: Token
        let right: Token
    }

    /// A spelling requested by a run configuration, before it has an ID.
    struct SpecialTokenDefinition: Codable, Equatable {
        let name: String
        let text: String
    }

    /// A durable reserved token. IDs are part of the vocabulary format.
    struct SpecialToken: Codable, Equatable {
        let name: String
        let text: String
        let id: Int
    }

    /// Only durable values are stored. `tokenToId` is rebuilt while loading.
    private enum CodingKeys: String, CodingKey {
        case merges, idToToken, specialTokens
    }

    /// Merge rules stay in learning order because encoding replays that order.
    private var merges: [Pair] = []

    /// The two directions of the ordinary-token lookup table.
    private var tokenToId: [Token: Int] = [:]
    private var idToToken: [Token] = []

    /// Reserved tokens stay in declaration order, which also defines priority.
    private(set) var specialTokens: [SpecialToken] = []

    /// Number of byte and learned BPE tokens, excluding reserved tokens.
    var vocabularySize: Int { idToToken.count }

    /// Number of logits the model predicts, including reserved tokens.
    var modelVocabularySize: Int { idToToken.count + specialTokens.count }

    /// ID of the configured document-boundary token.
    var endOfTextTokenID: Int {
        specialTokens.first { $0.name == "endOfText" }!.id
    }

    /// Trains up to the total model vocabulary limit, reserving special slots.
    init(
        trainOn text: String,
        maximumVocabularySize: Int,
        specialTokens definitions: [SpecialTokenDefinition] = [
            .init(name: "endOfText", text: Self.endOfTextMarker)
        ]
    ) {
        precondition(maximumVocabularySize >= 256 + definitions.count)
        precondition(Self.areValid(definitions))

        idToToken = (0..<256).map { [UInt8($0)] }
        tokenToId = Dictionary(uniqueKeysWithValues: idToToken.enumerated().map { ($0.element, $0.offset) })
        let ordinaryMaximum = maximumVocabularySize - definitions.count
        var tokens = byteTokens(from: text)

        while idToToken.count < ordinaryMaximum {
            updateLine("Token count: \(idToToken.count)/\(ordinaryMaximum)")
            guard let winner = mostFrequentPair(in: tokens) else { break }
            merges.append(winner)
            tokens = merge(winner, in: tokens)
            let newToken = winner.left + winner.right
            if tokenToId[newToken] == nil {
                tokenToId[newToken] = idToToken.count
                idToToken.append(newToken)
            }
        }
        specialTokens = definitions.enumerated().map {
            .init(name: $0.element.name, text: $0.element.text, id: idToToken.count + $0.offset)
        }
    }

    /// Loads both the new format and historical vocabularies without rewriting.
    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        merges = try container.decode([Pair].self, forKey: .merges)
        idToToken = try container.decode([Token].self, forKey: .idToToken)
        let baseTokens: [Token] = (0..<256).map { [UInt8($0)] }
        guard idToToken.count >= 256, idToToken.starts(with: baseTokens), Set(idToToken).count == idToToken.count else {
            throw DecodingError.dataCorruptedError(forKey: .idToToken, in: container, debugDescription: "Invalid BPE vocabulary.")
        }
        specialTokens = try container.decodeIfPresent([SpecialToken].self, forKey: .specialTokens)
            ?? [.init(name: "endOfText", text: Self.endOfTextMarker, id: idToToken.count)]
        guard Self.areValid(specialTokens.map { .init(name: $0.name, text: $0.text) }),
              specialTokens.enumerated().allSatisfy({ $0.element.id == idToToken.count + $0.offset }) else {
            throw DecodingError.dataCorruptedError(forKey: .specialTokens, in: container, debugDescription: "Invalid BPE special tokens.")
        }
        tokenToId = Dictionary(uniqueKeysWithValues: idToToken.enumerated().map { ($0.element, $0.offset) })
    }

    /// Persists the learned merges, ordinary tokens, and resolved special IDs.
    func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(merges, forKey: .merges)
        try container.encode(idToToken, forKey: .idToToken)
        try container.encode(specialTokens, forKey: .specialTokens)
    }

    /// Encodes ordinary text with learned merges and special spellings atomically.
    ///
    /// Reserved spellings are checked in declaration order at each position, so
    /// the first definition wins when two spellings overlap.
    func encode(_ text: String) -> [Int] {
        var result: [Int] = []
        var cursor = text.startIndex
        while cursor < text.endIndex {
            if let token = specialTokens.first(where: { text[cursor...].hasPrefix($0.text) }) {
                result.append(token.id)
                cursor = text.index(cursor, offsetBy: token.text.count)
                continue
            }
            let next = specialTokens.compactMap { text[cursor...].range(of: $0.text)?.lowerBound }.min() ?? text.endIndex
            result += encodeOrdinary(String(text[cursor..<next]))
            cursor = next
        }
        return result
    }

    /// Kept for source compatibility; all configured special tokens are now atomic.
    func encodeWithSpecialMarkers(_ text: String) -> [Int] { encode(text) }

    /// Reconstructs text from ordinary byte tokens and reserved spellings.
    func decode(_ ids: [Int]) -> String {
        let bytes = ids.flatMap { id -> [UInt8] in
            if let special = specialTokens.first(where: { $0.id == id }) {
                return Array(special.text.utf8)
            }
            precondition(idToToken.indices.contains(id), "Unknown token ID: \(id)")
            return idToToken[id]
        }

        // A Unicode character may span several byte tokens. Decode only after
        // reassembling the complete byte stream so those bytes stay together.
        return String(decoding: bytes, as: UTF8.self)
    }

    /// Stable tokenizer identity used to bind new checkpoints to token meanings.
    var identityDigest: String {
        struct Identity: Codable {
            let merges: [Pair]
            let idToToken: [Token]
            let specialTokens: [SpecialToken]
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try! encoder.encode(Identity(merges: merges, idToToken: idToToken, specialTokens: specialTokens))
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Atomically saves the tokenizer so interruption cannot corrupt the file.
    func save(to url: URL) throws {
        let data = try JSONEncoder().encode(self)
        try data.write(to: url, options: .atomic)
    }

    static func load(from url: URL) throws -> BPE {
        try JSONDecoder().decode(BPE.self, from: Data(contentsOf: url))
    }

    /// Applies learned merges only to ordinary text between special markers.
    private func encodeOrdinary(_ text: String) -> [Int] {
        merges.reduce(byteTokens(from: text)) { merge($1, in: $0) }.map { tokenToId[$0]! }
    }

    /// Creates the initial stream: one token for every UTF-8 byte.
    private func byteTokens(from text: String) -> [Token] { text.utf8.map { [$0] } }

    /// Selects the most common pair, keeping first-seen order for ties.
    private func mostFrequentPair(in tokens: [Token]) -> Pair? {
        var counts: [Pair: Int] = [:]
        var order: [Pair] = []
        for (left, right) in zip(tokens, tokens.dropFirst()) {
            let pair = Pair(left: left, right: right)
            if counts[pair] == nil { order.append(pair) }
            counts[pair, default: 0] += 1
        }
        var winner: Pair?
        var winningCount = 0
        for pair in order {
            let count = counts[pair, default: 0]
            if count > winningCount {
                winner = pair
                winningCount = count
            }
        }
        return winner
    }

    /// Replaces every non-overlapping occurrence with one larger token.
    private func merge(_ pair: Pair, in tokens: [Token]) -> [Token] {
        tokens.reduce(into: []) { result, symbol in
            if result.last == pair.left, symbol == pair.right { result[result.endIndex - 1] += symbol }
            else { result.append(symbol) }
        }
    }

    private static func areValid(_ tokens: [SpecialTokenDefinition]) -> Bool {
        !tokens.isEmpty && tokens.filter { $0.name == "endOfText" }.count == 1
            && Set(tokens.map(\.name)).count == tokens.count
            && Set(tokens.map(\.text)).count == tokens.count
            && tokens.allSatisfy { !$0.name.isEmpty && !$0.text.isEmpty }
    }
}

// MARK: - Debugging and inspection

extension BPE {
    /// Prints the IDs, raw bytes, and readable fragments produced for `text`.
    func inspect(_ text: String) {
        let ids = encode(text)
        print("Input: \(text.debugDescription)")
        print("Token count: \(ids.count)")

        for id in ids {
            let bytes: [UInt8]
            if let special = specialTokens.first(where: { $0.id == id }) {
                bytes = Array(special.text.utf8)
            } else {
                bytes = idToToken[id]
            }
            print(
                "ID \(id)",
                "bytes \(bytes)",
                "text \(String(decoding: bytes, as: UTF8.self).debugDescription)"
            )
        }
    }

    /// Prints a deterministic slice of learned ordinary vocabulary tokens.
    func inspectLearnedTokens(skip: Int = 256, limit: Int = 20) {
        precondition(
            skip >= 0 && skip <= idToToken.count,
            "Skip must be between 0 and \(idToToken.count)"
        )
        precondition(limit > 0, "Limit must be positive")

        for id in idToToken.indices.dropFirst(skip).prefix(limit) {
            let token = idToToken[id]
            print(
                "ID \(id)",
                "bytes \(token)",
                "text \(String(decoding: token, as: UTF8.self).debugDescription)"
            )
        }
    }

    /// Reports how many UTF-8 bytes each encoded token represents on average.
    func bytesPerToken(in text: String) -> Double {
        let tokenCount = encode(text).count
        guard tokenCount > 0 else { return 0 }
        return Double(text.utf8.count) / Double(tokenCount)
    }
}
