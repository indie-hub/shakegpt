//
//  BPE.swift
//  ShakeGPT
//
//  Created by Bruno O
//
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
struct BPE {

    /// One BPE token containing one or more UTF-8 bytes.
    typealias Token = [UInt8]

    /// Two adjacent tokens that may be joined during training or encoding.
    struct Pair: Hashable {
        let left: Token
        let right: Token
    }

    /// Merge rules in the exact order in which they were learned.
    private var merges: [Pair] = []

    /// Looks up the integer ID assigned to a byte token during encoding.
    private var tokenToId: [Token: Int] = [:]

    /// Looks up the byte token represented by an integer ID during decoding.
    private var idToToken: [Token] = []

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

    /// Encodes new text by converting it to bytes and replaying learned merges.
    func encode(_ text: String) -> [Int] {
        let tokens = self.merges.reduce(byteTokens(from: text)) { tokens, pair in
            merge(pair, in: tokens)
        }

        return tokens.map { token in
            tokenToId[token]!
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


private extension BPE {
    /// Converts text into the initial stream: one token per UTF-8 byte.
    func byteTokens(from text: String) -> [Token] {
        text.utf8.map { byte in [byte] }
    }

    /// Counts how often each adjacent pair occurs in the current token stream.
    func countPairs(in tokens: [Token]) -> [Pair:Int] {
        zip(tokens, tokens.dropFirst())
            .reduce(into: [Pair:Int]()) { counts, neighbours in
                let pair = Pair(left: neighbours.0, right: neighbours.1)

                counts[pair, default: 0] += 1
            }
    }

    /// Selects the most frequent pair, keeping the first occurrence on a tie.
    func mostFrequentPair(in tokens: [Token]) -> Pair? {
        let counts = countPairs(in: tokens)

        var winner: Pair?
        var winningCount: Int = 0

        for (left, right) in zip(tokens, tokens.dropFirst()) {
            let pair: Pair = Pair(left: left, right: right)
            let count = counts[pair, default: 0]

            if count > winningCount {
                winner = pair
                winningCount = count
            }
        }

        return winner
    }

    /// Replaces every non-overlapping occurrence of `pair` with one larger token.
    func merge(_ pair: Pair, in word: [Token]) -> [Token] {
        word.reduce(into: []) { result, symbol in
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
}
