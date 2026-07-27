//
//  BPE.swift
//  ShakeGPT
//
//  Created by Bruno O
//

import Foundation

/// A small byte-level Byte Pair Encoding (BPE) tokenizer.
///
/// BPE begins with one token for every UTF-8 byte in the training text. During
/// training, it repeatedly finds the most common adjacent token pair and joins
/// that pair into a larger token. The learned merges and vocabulary form the
/// model: encoding new text means starting from its bytes, replaying the merges
/// in order, and replacing the resulting tokens with their integer IDs.
///
/// Using bytes guarantees that every Swift `String` can be represented without
/// needing an "unknown character" token.
struct BPE: Codable {
    /// A visible corpus marker that becomes one reserved token ID during encoding.
    static let endOfTextMarker = "<|endoftext|>"

    /// One BPE token containing one or more UTF-8 bytes.
    typealias Token = [UInt8]

    /// Two adjacent tokens that may be joined during training or encoding.
    struct Pair: Hashable, Codable {
        let left: Token
        let right: Token
    }


    /// Only the durable parts of the tokenizer belong in its JSON file.
    ///
    /// `tokenToId` is derived from `idToToken`, so loading can rebuild it rather
    /// than storing the same relationship twice.
    private enum CodingKeys: String, CodingKey {
        case merges
        case idToToken
    }

    /// Merge rules in the exact order in which they were learned.
    private var merges: [Pair] = []

    /// Looks up the integer ID assigned to a byte token during encoding.
    private var tokenToId: [Token: Int] = [:]

    /// Looks up the byte token represented by an integer ID during decoding.
    private var idToToken: [Token] = []

    /// The number of byte and learned tokens actually available for encoding.
    var vocabularySize: Int {
        idToToken.count
    }

    /// The first ID after the byte and learned BPE tokens marks a document boundary.
    ///
    /// This ID is never produced by ordinary text encoding. Callers insert it
    /// explicitly when a complete document ends.
    var endOfTextTokenID: Int {
        vocabularySize
    }

    /// The transformer predicts every BPE token plus the end-of-text token.
    var modelVocabularySize: Int {
        vocabularySize + 1
    }

    /// Trains until the vocabulary reaches `maximumVocabularySize` or the
    /// current token stream contains no adjacent pair left to merge.
    init(trainOn text: String, maximumVocabularySize: Int) {
        precondition(
            maximumVocabularySize >= 256,
            "Byte-level BPE requires all 256 byte tokens"
        )

        idToToken = (0..<256).map { byte in [UInt8(byte)] }
        tokenToId = Dictionary(
            uniqueKeysWithValues: idToToken.enumerated().map {
                id, token in (token, id)
            }
        )


        var tokens: [Token] = byteTokens(from: text)

        while idToToken.count < maximumVocabularySize {
            updateLine("Token count: \(idToToken.count)/\(maximumVocabularySize)")
            guard let winner = mostFrequentPair(in: tokens) else {
                break
            }

            self.merges.append(winner)
            tokens = merge(winner, in: tokens)

            let newToken = winner.left + winner.right

            if tokenToId[newToken] == nil {
                tokenToId[newToken] = idToToken.count
                idToToken.append(newToken)
            }
        }
    }


    /// Restores a saved vocabulary and rebuilds its reverse lookup table.
    ///
    /// The first 256 entries must still be the original byte tokens. This check
    /// rejects damaged or incompatible vocabulary files before encoding begins.
    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        merges = try container.decode([Pair].self, forKey: .merges)
        idToToken = try container.decode([Token].self, forKey: .idToToken)

        let baseTokens: [Token] = (0..<256).map {
            [UInt8($0)]
        }

        guard
            idToToken.count >= 256,
            idToToken.starts(with: baseTokens),
            Set(idToToken).count == idToToken.count
        else {
            throw DecodingError.dataCorruptedError(
                forKey: .idToToken,
                in: container,
                debugDescription: "Invalid BPE vocabulary."
            )
        }

        tokenToId = Dictionary(
            uniqueKeysWithValues: idToToken.enumerated().map {
                id, token in (token, id)
            }
        )
    }

    /// Saves the learned merge order and ID-to-token vocabulary through `Codable`.
    func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        try container.encode(merges, forKey: .merges)
        try container.encode(idToToken, forKey: .idToToken)
    }

    /// Encodes new text by converting it to bytes and replaying learned merges.
    func encode(_ text: String) -> [Int] {
        let tokens = self.merges.reduce(byteTokens(from: text)) { tokens, pair in
            merge(pair, in: tokens)
        }

        return tokens.map { token in
            tokenToId[token]!
        }
    }
    /// Encodes corpus text while turning every visible boundary marker into EOT.
    ///
    /// Ordinary text is encoded section by section so the marker's characters
    /// never enter the model as thirteen unrelated byte tokens.
    func encodeWithSpecialMarkers(_ text: String) -> [Int] {
        let sections = text.components(separatedBy: Self.endOfTextMarker)

        return sections.enumerated().flatMap { index, section in
            var tokens = encode(section)

            if index < sections.count - 1 {
                tokens.append(endOfTextTokenID)
            }

            return tokens
        }
    }

    /// Reconstructs text by resolving token IDs and joining their UTF-8 bytes.
    func decode(_ ids: [Int]) -> String {
        let bytes = ids.flatMap { id -> Token in
            precondition(idToToken.indices.contains(id), "Unknown token ID: \(id)")
            return idToToken[id]
        }

        return String(decoding: bytes, as: Unicode.UTF8.self)
    }
}

// MARK: - Persistence

extension BPE {
    /// Atomically writes the vocabulary so an interrupted save cannot corrupt it.
    func save(to url: URL) throws {
        let data = try JSONEncoder().encode(self)
        try data.write(to: url, options: .atomic)
    }

    /// Loads the merge rules and token IDs previously written by `save(to:)`.
    static func load(from url: URL) throws -> BPE {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(BPE.self, from: data)
    }
}

private extension BPE {
    /// Converts text into the initial stream: one token per UTF-8 byte.
    func byteTokens(from text: String) -> [Token] {
        text.utf8.map { byte in [byte] }
    }

    /// Selects the most frequent pair, keeping the first occurrence on a tie.
    func mostFrequentPair(in tokens: [Token]) -> Pair? {
        var counts: [Pair: Int] = [:]
        var firstSeen: [Pair] = []

        for (left, right) in zip(tokens, tokens.dropFirst()) {
            let pair = Pair(left: left, right: right)

            if counts[pair] == nil {
                firstSeen.append(pair)
            }

            counts[pair, default: 0] += 1
        }

        var winner: Pair?
        var winningCount: Int = 0

        for pair in firstSeen {
            let count = counts[pair, default: 0]

            if count > winningCount {
                winner = pair
                winningCount = count
            }
        }

        return winner
    }

    /// Replaces every non-overlapping occurrence of `pair` with one larger token.
    func merge(_ pair: Pair, in tokens: [Token]) -> [Token] {
        tokens.reduce(into: []) { result, symbol in
            if result.last == pair.left, symbol == pair.right {
                result[result.endIndex - 1] += symbol
            } else {
                result.append(symbol)
            }
        }
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
            let bytes = idToToken[id]
            let text = String(decoding: bytes, as: UTF8.self)

            print(
                "ID \(id)",
                "bytes \(bytes)",
                "text \(text.debugDescription)"
            )
        }
    }

    /// Prints a deterministic slice of the vocabulary for training inspection.
    /// The default skips the 256 single-byte tokens and starts with learned ones.
    func inspectLearnedTokens(skip: Int = 256, limit: Int = 20) {
        precondition(
            skip >= 0 && skip <= idToToken.count,
            "Skip must be between 0 and \(idToToken.count)"
        )
        precondition(limit > 0, "Limit must be positive")

        for id in idToToken.indices.dropFirst(skip).prefix(limit) {
            let token = idToToken[id]
            let text = String(decoding: token, as: UTF8.self)

            print(
                "ID \(id)",
                "bytes \(token)",
                "text \(text.debugDescription)"
            )
        }
    }

    /// Reports tokenizer compression for `text`.
    ///
    /// Empty text contains neither bytes nor tokens, so its ratio is defined as
    /// zero rather than producing `NaN` from `0 / 0`.
    func bytesPerToken(in text: String) -> Double {
        let tokenCount = encode(text).count

        guard tokenCount > 0 else {
            return 0
        }

        return Double(text.utf8.count) / Double(tokenCount)
    }
}
