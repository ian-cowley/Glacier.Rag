# Glacier.Rag User Manual

This manual documents the public API in `Glacier.Rag` as implemented in `src/Glacier.Rag`.

---

## 1. Engine Lifecycle (`GraphRagEngine` & `RagOptions`)

`Glacier.Rag.Engine.GraphRagEngine` is thread-safe for concurrent reads and serialized writes, implementing `IDisposable`:

```csharp
using Glacier.Rag.Embeddings;
using Glacier.Rag.Engine;

using var rag = new GraphRagEngine(new FastHashEmbeddingModel(384));
```

If no `IEmbeddingModel` is passed to `new GraphRagEngine()`, it defaults to `new FastHashEmbeddingModel(384)`.

Inspection properties:

- `rag.IndexedChunksCount` — Number of vector-indexed document chunks.
- `rag.GraphNodeCount` — Number of nodes currently stored in `GraphStore`.
- `rag.GraphEdgeCount` — Number of relational edges stored in `GraphStore`.
- `rag.EmbeddingModel` — Active `IEmbeddingModel` instance (disposed when `GraphRagEngine` disposes).

### Configuring Retrieval (`RagOptions`)

| Property | Default | Purpose |
| :--- | :--- | :--- |
| `TopK` | `3` | Maximum dense vector matches to retrieve before fusion |
| `MaxGraphHops` | `2` | Maximum knowledge-graph traversal depth from query entities |
| `MaxDegreePerNode` | `25` | Fan-out cap per graph node during traversal |
| `MaxTriplets` | `100` | Cap on expanded graph relationships returned in `GraphRelations` |
| `MinSimilarity` | `0.0f` | Minimum similarity score threshold for vector matches |
| `SynthesizeWithLlm` | `false` | Optional hint when integrating downstream synthesis |
| `MaxTokensToGenerate` | `256` | Token limit used by `AskAsync` |

---

## 2. Document Chunking & Ingestion

### `IndexDocument` and `IndexDocumentAsync`

```csharp
int chunkCount = rag.IndexDocument("DOC-001", text, metadata: new Dictionary<string, string>
{
    ["source"] = "architecture-spec"
});

// Asynchronous ingestion (batches HTTP calls for OpenAiEmbeddingModel before taking write lock)
int asyncChunks = await rag.IndexDocumentAsync("DOC-002", largeText, cancellationToken: ct);
```

Internally, ingestion calls `DocumentChunker.ChunkText(docId, text, maxChunkLength: 512, overlap: 64)`:

- Prefers paragraph boundaries (`\r\n\r\n`, `\n\n`), single newlines, sentence periods, and whitespace rather than cutting words mid-token.
- Assigns deterministic chunk IDs (`{documentId}_{chunkIndex}`) and records `StartChar` and `EndChar`.
- Runs `EntityExtractor` over each chunk to extract PascalCase symbols, acronyms, quoted phrases, and directed predicate triplets (`EntityTriplet(Subject, Predicate, Object)` such as `IMPLEMENTS`, `DEPENDS_ON`, `STREAMS_FROM`, `VALIDATES`, `ROUTES_TO`).

---

## 3. Retrieval & End-to-End Synthesis

### Synchronous & Asynchronous Retrieval

```csharp
HybridRetrievalResult result = rag.Retrieve("How does CustomerRecord stream data?", new RagOptions
{
    TopK = 4,
    MaxGraphHops = 2
});

HybridRetrievalResult asyncResult = await rag.RetrieveAsync(query, options, cancellationToken: ct);
```

`HybridRetrievalResult` returns:

- `VectorMatches` — Top matching `SearchResult` chunks from `HnswVectorIndex`.
- `DiscoveredEntities` — Entities matched or reached during graph expansion.
- `GraphRelations` — `GraphRelation` entries (`Source --[Relation]--> Target`).
- `ScoredChunks` — `HybridGraphScorer.ScoredChunk` list with `DenseScore`, `GraphScore`, `HybridScore`, and `RelevantRelations`.
- `SynthesizedContext` — Combined textual context assembled from top scored chunks and relationships.
- `VectorSearchLatencyMs`, `GraphTraversalLatencyMs`, `TotalRetrievalLatencyMs`.

### Prompt Synthesis & `AskAsync`

Build an injection-escaped prompt manually:

```csharp
string chatMlPrompt = rag.BuildAugmentedPrompt(query, result);
string sessionPrompt = rag.BuildAugmentedPrompt(query, result, inferenceSession);
```

Or run retrieval + prompt formatting + streaming generation in one call using `Glacier.Inference`:

```csharp
using Glacier.Inference.Engine;

using var session = new InferenceSession("model.gguf", maxSeqLen: 2048);
string answer = await rag.AskAsync(
    query: "Summarize the ledger persistence model.",
    session: session,
    options: new RagOptions { TopK = 3, MaxTokensToGenerate = 256 },
    onToken: token => Console.Write(token),
    ct: cancellationToken);
```

Both `SynthesizedContext` and the user `query` pass through `PromptSanitizer.Sanitize(...)` so model control tokens (`<|im_start|>`, `<|eot_id|>`, `<think>`, `[INST]`, etc.) in indexed documents cannot hijack chat roles.

---

## 4. Embedding Models (`IEmbeddingModel`)

All embedding backends implement `Glacier.Rag.Embeddings.IEmbeddingModel` (`int Dimensions`, `GenerateEmbedding(ReadOnlySpan<char>, Span<float>)`, `IDisposable`).

### `FastHashEmbeddingModel`

Deterministic lexical feature-hashing model over word tokens and character trigrams:

```csharp
var hashModel = new FastHashEmbeddingModel(dimensions: 384); // dimensions must be a positive multiple of 8
```

### `GlacierInferenceEmbeddingModel`

Runs a local encoder or decoder GGUF model directly in-process via `Glacier.Inference.Engine.InferenceSession`:

```csharp
using var model = new GlacierInferenceEmbeddingModel(
    ggufModelPath: "nomic-embed.gguf",
    device: "auto",
    poolingStrategy: PoolingStrategy.Mean);
```

### `LinearProjectionEmbeddingModel`

Projects an upstream `IEmbeddingModel` into a lower-dimensional target space using AVX-512 / AVX2 / `Vector128` GEMV followed by SIMD L2 normalization:

```csharp
using var projected = new LinearProjectionEmbeddingModel(
    upstreamModel: upstream,
    outputDim: 384,
    seed: 1337); // or pass explicit float[] weights and optional float[]? bias
```

### `OpenAiEmbeddingModel`

NativeAOT-compatible client targeting any OpenAI-compatible `POST /v1/embeddings` endpoint (local NPU servers such as FastFlowLM / Lemonade or remote REST endpoints):

```csharp
// Auto-probe embedding dimensionality from the endpoint
using var embedder = await OpenAiEmbeddingModel.CreateAutoAsync(
    baseUrl: "http://127.0.0.1:52625",
    modelId: "embeddinggemma",
    apiKey: null,
    timeout: TimeSpan.FromSeconds(30));

// Or inspect dimensions without instantiating
int dims = await OpenAiEmbeddingModel.ProbeDimensionsAsync("http://127.0.0.1:52625", "embeddinggemma");

// Or supply known dimensions directly
var direct = new OpenAiEmbeddingModel("http://127.0.0.1:8000", "embeddinggemma", dimensions: 768);
float[][] batch = await direct.GenerateEmbeddingsAsync(inputs, maxBatchSize: 64);
```
