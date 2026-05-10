// availability_check.swift
//
// The right way to gate Foundation Models usage.
// Print the exact reason if the on-device model is not currently usable.

import Foundation
import FoundationModels

let model = SystemLanguageModel.default

switch model.availability {
case .available:
    print("Foundation Models: available on-device.")
case .unavailable(let reason):
    // .deviceNotEligible, .appleIntelligenceNotEnabled,
    // .modelNotReady (still downloading), .insufficientStorage, etc.
    print("Foundation Models: unavailable. Reason: \(reason)")
    print("Fall back to a bigger local model (Topic 08) or remote API.")
}
