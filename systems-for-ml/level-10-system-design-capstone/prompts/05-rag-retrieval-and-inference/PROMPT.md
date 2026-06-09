# Prompt 05 — RAG-Shaped Retrieval + Inference Service

Internal-search product for a Fortune 500 company. Employees ask natural-language questions; the system retrieves from internal docs and synthesizes an answer.

Components needed:
- **Embedding model** (BGE-large or similar) to embed both queries and docs
- **Vector store** (assume already exists — Pinecone / pgvector / Weaviate)
- **Reranker** (cross-encoder, e.g. bge-reranker-v2)
- **LLM** (Llama-3-70B for the synthesis step)
- **Optional vision encoder** for screenshots in queries

Workload: 50 QPS sustained, p95 < 3s end-to-end (the full retrieve → rerank → synthesize loop).

Design the inference platform. The vector store is given; design everything else.
