# Changelog

All notable changes to `Glacier.Rag` are documented here.

## v1.4.3

- Cached CSR graph compilation (`InMemoryCsrGraph`) for HippoRAG Personalized PageRank reranking (`HybridGraphScorer.ScoreAndRerankHippoRag`).
- Tightened numeric acronym regex in `EntityExtractor` to prevent false-positive entity nodes.
- Added chunk overlap floor handling in `DocumentChunker` when natural breaks shorten trailing slices.
- Aligned `Glacier.Graph`, `Glacier.Inference`, and `Glacier.Vector` dependencies.

## v1.2.0

- Added native `OpenAiEmbeddingModel` with `ProbeDimensionsAsync`, `CreateAutoAsync`, and batched `GenerateEmbeddingsAsync` for OpenAI-compatible and NPU sidecar `/v1/embeddings` servers.
- Upgraded `HybridGraphScorer` with full reciprocal rank fusion (RRF) and multi-hop graph candidate expansion.
- Hardened `GraphRagEngine` disposal using lock-free `_activeOps` draining and non-blocking async I/O.

## v1.0.2

- Initial public release unifying `Glacier.Vector` HNSW search, `Glacier.Graph` CSR traversal, `EntityExtractor`, and `Glacier.Inference` streaming generation.
