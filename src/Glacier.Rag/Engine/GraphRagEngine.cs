namespace Glacier.Rag.Engine;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Graph.Storage;
using Glacier.Graph.Traversal;
using Glacier.Inference.Engine;
using Glacier.Inference.Sampling;
using Glacier.Rag.Chunking;
using Glacier.Rag.Embeddings;
using Glacier.Rag.Extraction;
using Glacier.Vector.Index;
using Glacier.Vector.Storage;

public sealed class RagOptions
{
    public int TopK { get; set; } = 3;
    public int MaxGraphHops { get; set; } = 2;
    public int MaxDegreePerNode { get; set; } = 25;
    public int MaxTriplets { get; set; } = 100;
    public float MinSimilarity { get; set; } = 0.0f;
    public bool SynthesizeWithLlm { get; set; } = false;
    public int MaxTokensToGenerate { get; set; } = 256;
}

/// <summary>
/// High-throughput in-process GraphRAG engine integrating CSR Graph (Glacier.Graph), Vector Search (Glacier.Vector),
/// and streaming native inference (Glacier.Inference) in the exact same memory space.
/// Synchronized with reader-writer locking to allow unbounded concurrent retrieval queries.
/// </summary>
public sealed class GraphRagEngine : IDisposable
{
    private readonly IEmbeddingModel _embeddingModel;
    private readonly IVectorStorage _vectorStorage;
    private readonly VectorIndex _vectorIndex;
    private readonly GraphStore _graphStore;
    private readonly GraphSearch _graphSearch;
    private readonly Dictionary<string, DocumentChunk> _chunksById = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, string> _chunkContentById = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, List<string>> _docEntities = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, List<string>> _entityToChunkIds = new(StringComparer.OrdinalIgnoreCase);
    private readonly ReaderWriterLockSlim _rwLock = new(LockRecursionPolicy.SupportsRecursion);
    private bool _disposed;

    public int IndexedChunksCount
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _vectorIndex.Count;
        }
    }

    public int GraphNodeCount
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _graphStore.NodeCount;
        }
    }

    public int GraphEdgeCount
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _graphStore.EdgeCount;
        }
    }

    public IEmbeddingModel EmbeddingModel => _embeddingModel;

    public GraphRagEngine(IEmbeddingModel? embeddingModel = null)
    {
        _embeddingModel = embeddingModel ?? new FastHashEmbeddingModel(384);
        _vectorStorage = new InMemoryVectorStorage(_embeddingModel.Dimensions);
        _vectorIndex = new VectorIndex(_vectorStorage, leaveStorageOpen: true);
        _graphStore = new GraphStore(initialNodeCapacity: 50_000, initialEdgeCapacity: 200_000);
        _graphSearch = new GraphSearch(_graphStore);
    }

    /// <summary>
    /// Ingests and indexes an entire document into both VectorIndex and GraphStore simultaneously.
    /// Zero-copy, zero external processes.
    /// </summary>
    public int IndexDocument(string docId, string text, IDictionary<string, string>? metadata = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var chunks = DocumentChunker.ChunkText(docId, text);
        float[] embBuffer = new float[_embeddingModel.Dimensions];

        _rwLock.EnterWriteLock();
        try
        {
            foreach (var chunk in chunks)
            {
                if (metadata != null)
                {
                    foreach (var kv in metadata) chunk.Metadata[kv.Key] = kv.Value;
                }

                // 1. Vector Indexing keyed by unique ChunkId (prevents duplicate-boilerplate overwrites)
                _embeddingModel.GenerateEmbedding(chunk.Content, embBuffer);
                _vectorIndex.Add(embBuffer, chunk.ChunkId);
                IndexChunkGraphAndMetadata(docId, chunk);
            }
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }

        return chunks.Count;
    }

    /// <summary>
    /// Asynchronously ingests and indexes an entire document, forwarding <paramref name="cancellationToken"/>
    /// to asynchronous <see cref="OpenAiEmbeddingModel"/> calls without holding the write lock across I/O.
    /// </summary>
    public async Task<int> IndexDocumentAsync(
        string docId,
        string text,
        IDictionary<string, string>? metadata = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        cancellationToken.ThrowIfCancellationRequested();

        var chunks = DocumentChunker.ChunkText(docId, text);
        var vectors = new float[chunks.Count][];

        for (int i = 0; i < chunks.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (_embeddingModel is OpenAiEmbeddingModel openAi)
            {
                vectors[i] = await openAi.GenerateEmbeddingAsync(chunks[i].Content, cancellationToken).ConfigureAwait(false);
            }
            else
            {
                var buf = new float[_embeddingModel.Dimensions];
                _embeddingModel.GenerateEmbedding(chunks[i].Content, buf);
                vectors[i] = buf;
            }
        }

        ObjectDisposedException.ThrowIf(_disposed, this);
        _rwLock.EnterWriteLock();
        try
        {
            for (int i = 0; i < chunks.Count; i++)
            {
                var chunk = chunks[i];
                if (metadata != null)
                {
                    foreach (var kv in metadata) chunk.Metadata[kv.Key] = kv.Value;
                }

                _vectorIndex.Add(vectors[i], chunk.ChunkId);
                IndexChunkGraphAndMetadata(docId, chunk);
            }
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }

        return chunks.Count;
    }

    private void IndexChunkGraphAndMetadata(string docId, DocumentChunk chunk)
    {
        _chunksById[chunk.ChunkId] = chunk;
        _chunkContentById[chunk.ChunkId] = chunk.Content;

        // 2. Entity & Relation Graph Indexing + Inverted Entity -> ChunkId map
        var (entities, triplets) = EntityExtractor.Extract(chunk.Content);
        _docEntities[chunk.ChunkId] = entities;

        foreach (var entity in entities)
        {
            _graphStore.AddNode(entity, docId);
            if (!_entityToChunkIds.TryGetValue(entity, out var chunkList))
            {
                chunkList = new List<string>(4);
                _entityToChunkIds[entity] = chunkList;
            }
            if (!chunkList.Contains(chunk.ChunkId))
            {
                chunkList.Add(chunk.ChunkId);
            }
        }

        foreach (var t in triplets)
        {
            _graphStore.AddEdge(t.Subject, t.Object, t.Predicate);
        }
    }

    /// <summary>
    /// Executes hybrid retrieval combining dense vector search and multi-hop graph hops with authentic RRF scoring.
    /// </summary>
    public HybridRetrievalResult Retrieve(string query, RagOptions? options = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        int topK = options?.TopK ?? 3;
        float[]? queryEmb = null;
        if (topK > 0)
        {
            queryEmb = new float[_embeddingModel.Dimensions];
            _embeddingModel.GenerateEmbedding(query, queryEmb);
        }

        return RetrieveWithQueryEmbedding(query, queryEmb, options);
    }

    /// <summary>
    /// Asynchronously executes hybrid retrieval, forwarding <paramref name="cancellationToken"/>
    /// to asynchronous <see cref="OpenAiEmbeddingModel"/> calls without holding the read lock across I/O.
    /// </summary>
    public async Task<HybridRetrievalResult> RetrieveAsync(
        string query,
        RagOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        cancellationToken.ThrowIfCancellationRequested();

        int topK = options?.TopK ?? 3;
        float[]? queryEmb = null;
        if (topK > 0)
        {
            if (_embeddingModel is OpenAiEmbeddingModel openAi)
            {
                queryEmb = await openAi.GenerateEmbeddingAsync(query, cancellationToken).ConfigureAwait(false);
            }
            else
            {
                queryEmb = new float[_embeddingModel.Dimensions];
                _embeddingModel.GenerateEmbedding(query, queryEmb);
            }
        }

        ObjectDisposedException.ThrowIf(_disposed, this);
        return RetrieveWithQueryEmbedding(query, queryEmb, options);
    }

    private HybridRetrievalResult RetrieveWithQueryEmbedding(string query, float[]? queryEmb, RagOptions? options)
    {
        int topK = options?.TopK ?? 3;
        int maxHops = options?.MaxGraphHops ?? 2;
        float minSimilarity = options?.MinSimilarity ?? 0.0f;

        _rwLock.EnterReadLock();
        try
        {
            // 1. Dense Vector Search + MinSimilarity threshold filtering
            var swVec = Stopwatch.StartNew();
            SearchResult[] rawVectorResults;
            if (topK <= 0 || queryEmb == null)
            {
                rawVectorResults = [];
            }
            else
            {
                var candidates = _vectorIndex.Search(queryEmb, topK: topK);
                if (minSimilarity > 0.0f)
                {
                    var filtered = new List<SearchResult>(candidates.Length);
                    for (int i = 0; i < candidates.Length; i++)
                    {
                        if (candidates[i].Score >= minSimilarity)
                        {
                            filtered.Add(candidates[i]);
                        }
                    }
                    rawVectorResults = filtered.ToArray();
                }
                else
                {
                    rawVectorResults = candidates;
                }
            }
            swVec.Stop();

            // 2. Extract query entities & seed from vector matches
            var swGraph = Stopwatch.StartNew();
            var (queryEntities, _) = EntityExtractor.Extract(query);
            var queryEntitySet = new HashSet<string>(queryEntities, StringComparer.OrdinalIgnoreCase);
            var allEntities = new HashSet<string>(queryEntities, StringComparer.OrdinalIgnoreCase);

            for (int r = 0; r < rawVectorResults.Length; r++)
            {
                string? chunkId = rawVectorResults[r].Metadata;
                if (chunkId != null && _docEntities.TryGetValue(chunkId, out var ents))
                {
                    for (int i = 0; i < ents.Count; i++)
                    {
                        allEntities.Add(ents[i]);
                    }
                }
            }

            // 3. Multi-hop Graph Traversal (Forward Star CSR) with Hub Pruning & Predicate Retention
            var relations = new List<GraphRelation>();
            var seenTriplets = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var graphExpandedEntities = new HashSet<string>(queryEntitySet, StringComparer.OrdinalIgnoreCase);

            int maxDegree = options?.MaxDegreePerNode ?? 25;
            int maxTriplets = options?.MaxTriplets ?? 100;

            foreach (var entity in allEntities)
            {
                if (relations.Count >= maxTriplets) break;
                var triplets = _graphSearch.FindNeighborhoodTriplets(entity, maxHops, maxDegree, maxTriplets - relations.Count);
                bool seedFromQuery = queryEntitySet.Contains(entity);

                foreach (var t in triplets)
                {
                    string key = $"{t.Source}|{t.Predicate}|{t.Target}";
                    if (seenTriplets.Add(key))
                    {
                        relations.Add(new GraphRelation
                        {
                            Source = t.Source,
                            Target = t.Target,
                            Relation = t.Predicate
                        });
                    }

                    if (seedFromQuery)
                    {
                        graphExpandedEntities.Add(t.Source);
                        graphExpandedEntities.Add(t.Target);
                    }
                }
            }

            // Expand candidate chunk IDs linked to query entities and their graph neighborhood
            List<string>? graphCandidateIds = null;
            if (topK > 0 && graphExpandedEntities.Count > 0)
            {
                int maxGraphCandidates = Math.Max(topK * 4, 16);
                graphCandidateIds = new List<string>(maxGraphCandidates);
                var seenChunkIds = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                foreach (var ent in graphExpandedEntities)
                {
                    if (_entityToChunkIds.TryGetValue(ent, out var linkedChunks))
                    {
                        for (int i = 0; i < linkedChunks.Count; i++)
                        {
                            if (seenChunkIds.Add(linkedChunks[i]))
                            {
                                graphCandidateIds.Add(linkedChunks[i]);
                                if (graphCandidateIds.Count >= maxGraphCandidates) break;
                            }
                        }
                    }
                    if (graphCandidateIds.Count >= maxGraphCandidates) break;
                }
            }
            swGraph.Stop();

            // 4. Hybrid Graph-Dense Joint Scoring via Reciprocal Rank Fusion (RRF) with zero O(N) map copy
            var scoredChunks = topK > 0
                ? HybridGraphScorer.ScoreAndRerank(
                    rawVectorResults,
                    queryEntitySet,
                    _docEntities,
                    _chunkContentById,
                    _graphStore,
                    _graphSearch,
                    denseWeight: 0.5f,
                    kSmoothing: 60,
                    graphCandidateIds: graphCandidateIds,
                    maxResults: topK)
                : [];

            // Map VectorMatches metadata to human-readable chunk content for callers
            var externalVectorMatches = new SearchResult[rawVectorResults.Length];
            for (int i = 0; i < rawVectorResults.Length; i++)
            {
                string chunkId = rawVectorResults[i].Metadata ?? string.Empty;
                string content = _chunkContentById.TryGetValue(chunkId, out var c) ? c : chunkId;
                externalVectorMatches[i] = new SearchResult(rawVectorResults[i].Id, rawVectorResults[i].Score, content);
            }

            // 5. Synthesize Context Prompt
            var sb = new StringBuilder();
            sb.AppendLine("=== KNOWLEDGE GRAPH STRUCTURAL RELATIONSHIPS ===");
            if (relations.Count > 0)
            {
                foreach (var r in relations)
                {
                    sb.AppendLine($"* {r.Source} -> {r.Relation} -> {r.Target}");
                }
            }
            else
            {
                sb.AppendLine("(Direct entities active in context)");
            }

            sb.AppendLine("\n=== RETRIEVED SEMANTIC DOCUMENT EXCERPTS ===");
            if (scoredChunks.Count > 0)
            {
                for (int i = 0; i < scoredChunks.Count; i++)
                {
                    var sc = scoredChunks[i];
                    sb.AppendLine($"[Source {i + 1}] (Relevance: {sc.HybridScore:F4})");
                    sb.AppendLine(sc.Content);
                    sb.AppendLine();
                }
            }
            else
            {
                for (int i = 0; i < externalVectorMatches.Length; i++)
                {
                    sb.AppendLine($"[Source {i + 1}] (Relevance: {externalVectorMatches[i].Score:F4})");
                    sb.AppendLine(externalVectorMatches[i].Metadata);
                    sb.AppendLine();
                }
            }

            return new HybridRetrievalResult
            {
                Query = query,
                VectorMatches = externalVectorMatches,
                DiscoveredEntities = new List<string>(allEntities),
                GraphRelations = relations,
                SynthesizedContext = sb.ToString(),
                ScoredChunks = scoredChunks,
                VectorSearchLatencyMs = swVec.Elapsed.TotalMilliseconds,
                GraphTraversalLatencyMs = swGraph.Elapsed.TotalMilliseconds
            };
        }
        finally
        {
            _rwLock.ExitReadLock();
        }
    }

    /// <summary>
    /// Executes end-to-end GraphRAG: retrieves hybrid context, synthesizes augmented prompt,
    /// and streams answer tokens directly from Glacier.Inference in the exact same memory space.
    /// </summary>
    public async Task<string> AskAsync(
        string query,
        InferenceSession session,
        RagOptions? options = null,
        Action<string>? onToken = null,
        CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(session);

        var retrieval = await RetrieveAsync(query, options, ct).ConfigureAwait(false);
        string prompt = BuildAugmentedPrompt(query, retrieval);

        var samplingOptions = new SamplingOptions
        {
            MaxTokens = options?.MaxTokensToGenerate ?? 256,
            Temperature = 0.7f,
            TopP = 0.9f
        };

        var result = await session.GenerateAsync(
            prompt,
            samplingOptions,
            formatChat: false,
            onToken: onToken,
            ct: ct);

        return result.Text;
    }

    /// <summary>
    /// Synthesizes the full augmented prompt with ChatML formatting ready for LLM generation.
    /// </summary>
    public string BuildAugmentedPrompt(string query, HybridRetrievalResult retrieval)
    {
        return $"<|im_start|>system\n" +
               $"You are an expert enterprise AI assistant. Answer the user prompt accurately using ONLY the provided verified Knowledge Graph relationships and semantic document excerpts.\n\n" +
               $"{retrieval.SynthesizedContext}<|im_end|>\n" +
               $"<|im_start|>user\n" +
               $"{query}<|im_end|>\n" +
               $"<|im_start|>assistant\n";
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _rwLock.Dispose();
            _embeddingModel.Dispose();
            _vectorIndex.Dispose();
            _vectorStorage.Dispose();
            _graphStore.Dispose();
            _chunksById.Clear();
            _chunkContentById.Clear();
            _docEntities.Clear();
            _entityToChunkIds.Clear();
        }
    }
}
