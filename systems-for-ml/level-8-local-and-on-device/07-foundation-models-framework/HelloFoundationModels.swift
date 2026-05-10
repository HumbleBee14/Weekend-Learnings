// HelloFoundationModels.swift
//
// Minimal sample for Apple's Foundation Models framework.
// Requires macOS 26+ (or iOS 26+) and Apple Intelligence enabled.
//
// Two demos:
//   1. Streaming chat against the on-device ~3B model.
//   2. Schema-constrained structured output via @Generable.
//
// Build inside an Xcode 26 SwiftUI app, or run directly with `swift` on
// macOS 26+ where the FoundationModels framework is in the SDK.

import Foundation
import FoundationModels

@Generable
struct Recipe {
    let title: String
    let ingredients: [String]
    let steps: [String]
}

@main
struct HelloFoundationModels {
    static func main() async {
        // Always check availability first.
        let availability = SystemLanguageModel.default.availability
        switch availability {
        case .available:
            print("[on-device]")
        case .unavailable(let reason):
            print("Foundation Models unavailable: \(reason)")
            return
        }

        let session = LanguageModelSession(
            instructions: "You are concise. Three sentences max."
        )

        // 1. Streaming.
        print("Streaming chunks:")
        do {
            for try await chunk in session.streamResponse(
                to: "Explain unified memory on Apple Silicon."
            ) {
                print(chunk, terminator: "")
            }
            print("")
        } catch {
            print("Stream error: \(error)")
        }

        // 2. Structured output.
        do {
            let recipe = try await session.respond(
                to: "A simple pasta carbonara.",
                generating: Recipe.self
            )
            print("\nStructured output (Recipe):")
            print(recipe)
        } catch {
            print("Structured error: \(error)")
        }
    }
}
