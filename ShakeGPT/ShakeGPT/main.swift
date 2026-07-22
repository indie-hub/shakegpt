//
//  main.swift
//  ShakeGPT
//
//  Created by Bruno on 21/07/2026.
//


let corpus = "banana banana banana"
let model = BPE(trainOn: corpus, maximumVocabularySize: 1_024)

let toEncode = "hello"
let encoded = model.encode(toEncode)

print(encoded)
print("decoded: \(model.decode(encoded))")
