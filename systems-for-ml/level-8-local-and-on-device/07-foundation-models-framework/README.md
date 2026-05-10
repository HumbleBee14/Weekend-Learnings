# 07 — Foundation Models Framework

## Files

- `CONCEPTS.md` — what the framework is, the API surface (`LanguageModelSession`, `@Generable`, `Tool`), what the 3B model is good at, custom adapter flow, availability/PCC routing.
- `HelloFoundationModels.swift` — minimal Swift sample showing a streaming chat call and a `@Generable` structured-output call. Runs as a SwiftUI app on macOS 26+ with Xcode 26.
- `availability_check.swift` — small helper showing the right way to check `SystemLanguageModel.default.availability` before calling.

## Quickstart

You need macOS 26+, Xcode 26+, and Apple Intelligence enabled.

```bash
# Create a new SwiftUI macOS app in Xcode.
# Drop HelloFoundationModels.swift into the project as the App entry.
# Build & run.
```

Or, for the quick CLI smoke test using `swift`:

```bash
swift HelloFoundationModels.swift
```

This requires the `FoundationModels` framework to be in the SDK, which it is on macOS 26.

## Expected output

```
[on-device]
Streaming chunks:
The unified memory architecture on Apple Silicon...
...

Structured output (Recipe):
Recipe(title: "Pasta Carbonara",
       ingredients: ["spaghetti", "guanciale", "eggs", "pecorino", "black pepper"],
       steps: ["boil pasta", ...])
```

If Apple Intelligence is still downloading or unavailable in your region, `availability_check.swift` will print the precise reason rather than failing silently.

## Try

- Add a tool conforming to `Tool` that returns the current time. Pass it to `LanguageModelSession(tools: [...])` and watch the model decide to call it.
- Define a richer `@Generable` struct (nested types, optionals) and confirm the model never produces malformed JSON — the framework masks logits to the schema.
- Train a tiny adapter with Apple's adapter training Python toolkit on a small instruction dataset (https://developer.apple.com/apple-intelligence/foundation-models-adapter/), embed `.fmadapter`, and pass it to the session.
- Test the unavailability path: disable Apple Intelligence in System Settings and re-run `availability_check.swift`.

## Where this goes

Topic 08 is the cross-stack serving picture for **your own** local LLMs (Ollama / LM Studio / vLLM-MLX). Foundation Models is the Apple-blessed path; the rest of this level is what you do when you want bigger models or arbitrary weights. Topic 16 covers the privacy story including PCC routing.
