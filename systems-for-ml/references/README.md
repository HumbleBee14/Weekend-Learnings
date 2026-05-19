# References

External books and source code that pair with this curriculum. The repo is the main guide; these are the authoritative companions.

## Books in this folder

### `Inference-Engineering-Kiely-2025.pdf`

- **Title:** *Inference Engineering*
- **Author:** Philip Kiely (Head of Developer Relations, [Baseten](https://baseten.co))
- **Published:** December 2025, revised April 2026
- **Length:** 259 pages, 8 chapters + 2 appendices (47-page glossary + 26-page reading list)
- **Flavor:** Practitioner / production-grade. The book that explains how 2026 inference infrastructure is actually built and run.

**Where it pairs in this repo:**

| Chapter | This repo |
|---|---|
| Ch 0 Inference + Ch 1 Prerequisites | [Level 1](../level-1-inference-serving/) — read **before** starting |
| Ch 2 Models | python-pytorch/ prereq + Levels 2/3 |
| Ch 3 Hardware | [Level 2](../level-2-cuda-and-gpu-programming/) + [Level 8 §3.5](../level-8-local-and-on-device/) |
| Ch 4 Software (vLLM/SGLang/TRT-LLM/Dynamo + benchmarking) | [Level 5](../level-5-production-engines/) — strongest fit in the book |
| Ch 5 Techniques (quant / spec-decode / caching / parallelism / disaggregation) | [Level 4](../level-4-llm-optimization/) + [Level 5 Topics 08–09](../level-5-production-engines/) |
| Ch 6 Modalities (VLM / embedding / ASR / TTS / image-gen / video-gen) | [Level 5 Topics 13–14](../level-5-production-engines/) — Kiely covers ASR/TTS/image/video that this repo doesn't |
| Ch 7 Production (containers / autoscaling / multi-cloud / GPU procurement / observability) | [Level 7](../level-7-ml-platform/) — closest match to this level in any book published |
| App A Glossary | Cross-cutting reference |
| App B Recommended Reading | Curated 2025–2026 papers/posts |

**Caveat:** Section 7.6 is "Production Inference with Baseten" — read it as a vendor case study (the author works there), not neutral comparison.

## Books referenced externally

### *Machine Learning Systems* — Reddi et al. (Harvard CS249r, MIT Press 2026)

- Free online: [mlsysbook.ai](https://mlsysbook.ai/)
- Source: [harvard-edge/cs249r_book](https://github.com/harvard-edge/cs249r_book)
- Flavor: Academic, textbook-grade. Two volumes spanning the full ML systems lifecycle.

Maps to all 9 levels — see the table in the top-level [README.md](../README.md#mapping--both-books--this-repo).

## How to use both books

1. **Reddi** for the *concept* — citation-grade framing of what a thing is and why it matters
2. **Kiely** for the *production view* — what an inference engineer actually does with the thing in 2026
3. **This repo** for the *lab* — build it, break it, measure it, ship a report

If a level's topic is in both books, read Reddi first for the foundations, Kiely second for the production angle, then build in the repo.
