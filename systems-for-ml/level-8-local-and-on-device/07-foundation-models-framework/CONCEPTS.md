# 07 — Foundation Models Framework

## What it is

A Swift-native API to Apple's on-device ~3B parameter model — the same model that powers Apple Intelligence (writing tools, Genmoji, Siri summaries). Announced at WWDC25, matured through 2026. Available on macOS 26 / iOS 26 and later.

Key facts:

- Mixed 2/4-bit weights, **quantization-aware-trained from scratch** (not a post-hoc quantization of a bigger model).
- Runs fully offline on Apple Silicon. ANE-resident hot path; GPU/CPU for spillover.
- Zero per-token cost, no network, no key.
- The same binary on the device. Apps share it; no per-app weight footprint.
- Auto-routes to **Private Cloud Compute** for queries the on-device model can't handle (Topic 16).

## API surface

```swift
import FoundationModels

// 1. A stateful chat session.
let session = LanguageModelSession()

// 2. Plain prompt -> string.
let response = try await session.respond(to: "Summarize: ...")

// 3. Streaming.
for try await chunk in session.streamResponse(to: prompt) {
    print(chunk, terminator: "")
}

// 4. Guided structured output via @Generable.
@Generable struct Recipe {
    let title: String
    let ingredients: [String]
    let steps: [String]
}
let recipe = try await session.respond(to: "Pasta carbonara", generating: Recipe.self)

// 5. Tools.
struct WeatherTool: Tool {
    static let name = "weather"
    func call(arguments: WeatherArgs) async throws -> WeatherResult { ... }
}
let session2 = LanguageModelSession(tools: [WeatherTool()])
```

The `@Generable` macro is doing what `outlines` / XGrammar do in the Python ecosystem — schema-constrained generation. The constraint is enforced at logit-mask time, not by post-hoc parsing, so well-formed output is guaranteed.

## What the 3B model is good at

Use it for the slice where small models have always worked:

- Summarization
- Rewriting / tone change
- Extraction (entities, structured fields)
- Classification
- Short-form generation
- Function-calling for app-internal tools

What it is *not*: a GPT-4 / Claude / Gemini replacement. Code generation, multi-step reasoning, hard math, long-form drafting — fall back to a bigger local model (Topic 08) or a server.

## Languages supported (2026)

English, French, German, Italian, Spanish, Portuguese (Brazil), Chinese (simplified), Japanese, Korean. Apple has expanded the list each release; check `LanguageModelAvailability` at runtime rather than hard-coding.

## Custom adapters

Apple ships a Python toolchain to train LoRA-style adapters for the on-device base. The flow:

```
  base model (frozen, on-device)
       +
  your adapter (small, ~50-200 MB)
       =
  specialized model
```

Steps:

1. Install Apple's adapter training toolkit (Python). It pulls the base model's frozen weights for offline fine-tuning.
2. Prepare a JSONL dataset (instruction/response).
3. Train. Toolchain wraps a standard LoRA loop — frozen base, trainable low-rank adapters at attention/MLP layers.
4. Export `.fmadapter` and embed in your app bundle.
5. At runtime: `LanguageModelSession(adapter: myAdapter)`.

The adapter has access to the same on-device base across app launches and devices. You ship megabytes, not gigabytes.

```
  +-----------------------------+
  |  on-device 3B base (shared) |   <-- one copy on the OS
  +-----------------------------+
            ^
            | merge at load
            |
  +-----------------------------+
  |  your .fmadapter (~100 MB)  |   <-- shipped with app
  +-----------------------------+
```

## Availability and fallbacks

The framework expresses availability as a stack:

```
  isAvailable     -> use on-device
  needsPCC        -> route to Private Cloud Compute (Apple-attested)
  unavailable     -> fall back to your own remote model or refuse
```

Always check `SystemLanguageModel.default.availability` before calling. Reasons for unavailable: device not Apple-Intelligence-eligible, region unsupported, Apple Intelligence disabled, low storage, low battery.

## On-device + PCC threat model preview

The framework will route sensitive queries to PCC if on-device cannot satisfy them. PCC nodes are Apple Silicon servers with attested boot, no persistent storage, public binary transparency. Topic 16 covers this in depth — for now, know the routing exists and that it is not an arbitrary cloud call.

## What this is not

- Not a way to run *your own* arbitrary 3B model. The framework is bound to Apple's specific base. For your own weights, use MLX (Topic 02) or Core ML (Topic 06).
- Not training infrastructure. The Python toolchain is for adapter fine-tuning of *Apple's base*, not arbitrary models.
- Not reachable from the Mac CLI in a meaningful way. It is a framework-level API designed to be called from a SwiftUI / AppKit / UIKit app.

## Common pitfalls

1. **Treating the 3B as a foundation chat model.** It is a domain-shaped tool. Use it for the slice it nails; route harder things elsewhere.
2. **Not handling unavailability.** First-run on a fresh device may show `unavailable` while Apple Intelligence finishes downloading. Always have a non-FM path.
3. **Adapter overfitting.** LoRA on a few hundred examples will catastrophically forget. Keep a held-out general benchmark check (Topic 12 covers this).
4. **Schema drift via @Generable.** Renaming a field breaks model-side generation in subtle ways — keep schemas stable, version them in your code review.

## References

- Foundation Models docs: https://developer.apple.com/documentation/FoundationModels
- WWDC25 session "Meet the Foundation Models framework": https://developer.apple.com/videos/play/wwdc2025/286/
- Adapter training toolkit: https://developer.apple.com/apple-intelligence/foundation-models-adapter/
- Apple ML — On-device language model: https://machinelearning.apple.com/research/introducing-apple-foundation-models
- Private Cloud Compute (security overview): https://security.apple.com/blog/private-cloud-compute/
