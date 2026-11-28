namespace Glacier.Rag.Engine;

using System;
using System.Buffers;
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
/// High-throughput in-process GraphRAG engine integrating in-memory contiguous CSR Graph (Glacier.Graph),
/// HNSW Vector Search (Glacier.Vector), and streaming native inference (Glacier.Inference) in the same memory space.
/// Synchronized with reader-writer locking to allow unbounded concurrent retrieval queries.
/// </summary>
public sealed class GraphRagEngine : IDisposable
{
    private readonly IEmbeddingModel _embeddingModel;
    private readonly IVectorStorage _vectorStorage;
    private readonly HnswVectorIndex _vectorIndex;
    private readonly GraphStore _graphStore;
    private readonly GraphSearch _graphSearch;
    private readonly Dictionary<string, DocumentChunk> _chunksById = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, string> _chunkContentById = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, List<string>> _docEntities = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, List<string>> _entityToChunkIds = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, HashSet<string>> _canonicalEntitiesByLower = new(StringComparer.OrdinalIgnoreCase);
    private readonly ReaderWriterLockSlim _rwLock = new(LockRecursionPolicy.SupportsRecursion);
    private readonly object _csrSync = new();
    private InMemoryCsrGraph? _csrSnapshot;
    private bool _disposed;
    private int _disposeFlag;
    private int _activeOps;

    public int IndexedChunksCount
    {
        get
        {
            EnterReadLockSafe();
            try
            {
                return _vectorIndex.Count;
            }
            finally
            {
                ExitReadLockSafe();
            }
        }
    }

    public int GraphNodeCount
    {
        get
        {
            EnterReadLockSafe();
            try
            {
                return _graphStore.NodeCount;
            }
            finally
            {
                ExitReadLockSafe();
            }
        }
    }

    public int GraphEdgeCount
    {
        get
        {
            EnterReadLockSafe();
            try
            {
                return _graphStore.EdgeCount;
            }
            finally
            {
                ExitReadLockSafe();
            }
        }
    }

    public IEmbeddingModel EmbeddingModel => _embeddingModel;

    public GraphRagEngine(IEmbeddingModel? embeddingModel = null)
    {
        _embeddingModel = embeddingModel ?? new FastHashEmbeddingModel(384);
        _vectorStorage = new InMemoryVectorStorage(_embeddingModel.Dimensions);
        _vectorIndex = new HnswVectorIndex(_vectorStorage, m: 16, efConstruction: 100, initialCapacity: 4096);
        _graphStore = new GraphStore(initialNodeCapacity: 50_000, initialEdgeCapacity: 200_000);
        _graphSearch = new GraphSearch(_graphStore);
    }

    private void BeginOperation()
    {
        while (true)
        {
            int cur = Volatile.Read(ref _activeOps);
            if (cur < 0)
                throw new ObjectDisposedException(nameof(GraphRagEngine));

            if (Interlocked.CompareExchange(ref _activeOps, cur + 1, cur) == cur)
                return;
        }
    }

    private void EndOperation()
    {
        Interlocked.Decrement(ref _activeOps);
    }

    private void EnterReadLockSafe()
    {
        BeginOperation();
        try
        {
            _rwLock.EnterReadLock();
        }
        catch
        {
            EndOperation();
            throw;
        }
    }

    private void ExitReadLockSafe()
    {
        try
        {
            _rwLock.ExitReadLock();
        }
        finally
        {
            EndOperation();
        }
    }

    private void EnterWriteLockSafe()
    {
        BeginOperation();
        try
        {
            _rwLock.EnterWriteLock();
        }
        catch
        {
            EndOperation();
            throw;
        }
    }

    private void ExitWriteLockSafe()
    {
        try
        {
            _rwLock.ExitWriteLock();
        }
        finally
        {
            EndOperation();
        }
    }

    /// <summary>
    /// Ingests and indexes an entire document into both HnswVectorIndex and GraphStore simultaneously.
    /// Zero external processes.
    /// </summary>
    public int IndexDocument(string docId, string text, IDictionary<string, string>? metadata = null)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed), this);

        var chunks = DocumentChunker.ChunkText(docId, text);
        if (chunks.Count == 0)
            return 0;

        int dims = _embeddingModel.Dimensions;
        float[] embBuffer = ArrayPool<float>.Shared.Rent(dims);
        try
        {
            Span<float> embSpan = embBuffer.AsSpan(0, dims);

            EnterWriteLockSafe();
            try
            {
                foreach (var chunk in chunks)
                {
                    if (metadata != null)
                    {
                        foreach (var kv in metadata) chunk.Metadata[kv.Key] = kv.Value;
                    }

                    // 1. Vector Indexing keyed by unique ChunkId (prevents duplicate-boilerplate overwrites)
                    _embeddingModel.GenerateEmbedding(chunk.Content, embSpan);
                    _vectorIndex.Add(embSpan, chunk.ChunkId);
                    IndexChunkGraphAndMetadata(docId, chunk);
                }

                Volatile.Write(ref _csrSnapshot, null);
            }
            finally
            {
                ExitWriteLockSafe();
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(embBuffer);
        }

        return chunks.Count;
    }

    /// <summary>
    /// Asynchronously ingests and indexes an entire document, batching OpenAI embedding requests
    /// via <see cref="OpenAiEmbeddingModel.GenerateEmbeddingsAsync"/> and forwarding
    /// <paramref name="cancellationToken"/> without holding the write lock across I/O.
    /// </summary>
    public async Task<int> IndexDocumentAsync(
        string docId,
        string text,
        IDictionary<string, string>? metadata = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed), this);
        cancellationToken.ThrowIfCancellationRequested();

        var chunks = DocumentChunker.ChunkText(docId, text);
        if (chunks.Count == 0)
            return 0;

        float[][] vectors;
        if (_embeddingModel is OpenAiEmbeddingModel openAi)
        {
            string[] contents = new string[chunks.Count];
            for (int i = 0; i < chunks.Count; i++)
            {
                contents[i] = chunks[i].Content;
            }

            vectors = await openAi.GenerateEmbeddingsAsync(contents, cancellationToken: cancellationToken).ConfigureAwait(false);
        }
        else
        {
            vectors = new float[chunks.Count][];
            for (int i = 0; i < chunks.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var buf = new float[_embeddingModel.Dimensions];
                _embeddingModel.GenerateEmbedding(chunks[i].Content, buf);
                vectors[i] = buf;
            }
        }

        EnterWriteLockSafe();
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

            Volatile.Write(ref _csrSnapshot, null);
        }
        finally
        {
            ExitWriteLockSafe();
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
            AddCanonicalEntity(entity);
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
            AddCanonicalEntity(t.Subject);
            AddCanonicalEntity(t.Object);
        }
    }

    private void AddCanonicalEntity(string entity)
    {
        if (string.IsNullOrEmpty(entity))
        {
            return;
        }

        if (!_canonicalEntitiesByLower.TryGetValue(entity, out var canonicals))
        {
            canonicals = new HashSet<string>(StringComparer.Ordinal);
            _canonicalEntitiesByLower[entity] = canonicals;
        }
        canonicals.Add(entity);
    }

    /// <summary>
    /// Executes hybrid retrieval combining dense vector search and multi-hop graph hops with authentic RRF scoring.
    /// </summary>
    public HybridRetrievalResult Retrieve(string query, RagOptions? options = null)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed), this);
        ArgumentNullException.ThrowIfNull(query);

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
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed), this);
        ArgumentNullException.ThrowIfNull(query);
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

        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed), this);
        return RetrieveWithQueryEmbedding(query, queryEmb, options);
    }

    private HybridRetrievalResult RetrieveWithQueryEmbedding(string query, float[]? queryEmb, RagOptions? options)
    {
        int topK = options?.TopK ?? 3;
        int maxHops = options?.MaxGraphHops ?? 2;
        float minSimilarity = options?.MinSimilarity ?? 0.0f;

        EnterReadLockSafe();
        try
        {
            // 1. Dense HNSW Vector Search (oversampled 4x before RRF) + MinSimilarity filtering
            var swVec = Stopwatch.StartNew();
            SearchResult[] rawVectorResults;
            int indexedCount = _vectorIndex.Count;
            if (topK <= 0 || queryEmb == null || indexedCount == 0)
            {
                rawVectorResults = [];
            }
            else
            {
                int denseFetchK = (int)Math.Min((long)topK * 4L, indexedCount);
                int efSearch = (int)Math.Min(Math.Max(64L, (long)denseFetchK * 2L), int.MaxValue);
                var candidates = _vectorIndex.Search(queryEmb, denseFetchK, efSearch);
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

            // 2. Extract query entities (chunk-seeded entities and relation extraction are deferred until after RRF TopK selection)
            var swGraph = Stopwatch.StartNew();
            var (queryEntities, _) = EntityExtractor.Extract(query);
            var queryEntitySet = new HashSet<string>(queryEntities, StringComparer.OrdinalIgnoreCase);

            // Resolve case-variant query entities to canonical Ordinal graph node IDs for GraphStore / CSR lookups.
            var resolvedQueryEntitySet = new HashSet<string>(StringComparer.Ordinal);
            foreach (var q in queryEntitySet)
            {
                if (_canonicalEntitiesByLower.TryGetValue(q, out var canonicals) && canonicals.Count > 0)
                {
                    foreach (var canonical in canonicals)
                    {
                        resolvedQueryEntitySet.Add(canonical);
                    }
                }
                else
                {
                    resolvedQueryEntitySet.Add(q);
                }
            }

            // 3. Pre-RRF multi-hop expansion (strictly when topK > 0) for HybridGraphScorer graphCandidateIds;
            //    do NOT populate output GraphRelations here so dropped chunks/query-only edges cannot starve selected TopK chunks.
            var relations = new List<GraphRelation>();
            List<string>? graphCandidateIds = null;

            int maxDegree = options?.MaxDegreePerNode ?? 25;
            int maxTriplets = options?.MaxTriplets ?? 100;

            if (topK > 0 && resolvedQueryEntitySet.Count > 0)
            {
                var graphExpandedEntities = new HashSet<string>(resolvedQueryEntitySet, StringComparer.OrdinalIgnoreCase);
                if (maxTriplets > 0)
                {
                    foreach (var entity in resolvedQueryEntitySet)
                    {
                        var triplets = _graphSearch.FindNeighborhoodTriplets(entity, maxHops, maxDegree, maxTriplets);
                        for (int i = 0; i < triplets.Count; i++)
                        {
                            graphExpandedEntities.Add(triplets[i].Source);
                            graphExpandedEntities.Add(triplets[i].Target);
                        }
                    }
                }

                // Expand candidate chunk IDs linked to query entities and their graph neighborhood
                if (graphExpandedEntities.Count > 0)
                {
                    int maxGraphCandidates = (int)Math.Min(Math.Max((long)topK * 4L, 16L), Math.Max(_chunksById.Count, 16));
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
            }
            swGraph.Stop();

            // 4. Hybrid Graph-Dense Joint Scoring via HippoRAG Personalized PageRank over in-memory contiguous CSR spans
            List<HybridGraphScorer.ScoredChunk> scoredChunks;
            if (topK > 0)
            {
                var csr = GetOrCompileInMemoryCsr();
                scoredChunks = csr != null && csr.NodeCount > 0 && _graphStore.EdgeCount > 0 && resolvedQueryEntitySet.Count > 0
                    ? HybridGraphScorer.ScoreAndRerankHippoRag(
                        rawVectorResults,
                        resolvedQueryEntitySet,
                        _docEntities,
                        _chunkContentById,
                        csr,
                        denseWeight: 0.5f,
                        kSmoothing: 60,
                        graphCandidateIds: graphCandidateIds,
                        maxResults: topK)
                    : HybridGraphScorer.ScoreAndRerank(
                        rawVectorResults,
                        resolvedQueryEntitySet,
                        _docEntities,
                        _chunkContentById,
                        _graphStore,
                        _graphSearch,
                        denseWeight: 0.5f,
                        kSmoothing: 60,
                        graphCandidateIds: graphCandidateIds,
                        maxResults: topK);
            }
            else
            {
                scoredChunks = [];
            }

            // Map VectorMatches metadata to human-readable chunk content, capped at TopK
            int matchCount = topK > 0 ? Math.Min(topK, rawVectorResults.Length) : 0;
            var externalVectorMatches = new SearchResult[matchCount];
            for (int i = 0; i < matchCount; i++)
            {
                string chunkId = rawVectorResults[i].Metadata ?? string.Empty;
                string content = _chunkContentById.TryGetValue(chunkId, out var c) ? c : chunkId;
                externalVectorMatches[i] = new SearchResult(rawVectorResults[i].Id, rawVectorResults[i].Score, content);
            }

            // 4b. Seed DiscoveredEntities and GraphRelations from final post-RRF TopK chunks first, then remaining query entities
            swGraph.Start();
            var allEntities = new HashSet<string>(queryEntities, StringComparer.OrdinalIgnoreCase);
            if (topK > 0)
            {
                // Track exact Ordinal casing from indexed chunks so case-insensitive query entities
                // (e.g. "alphaservice") do not suppress GraphStore lookups for "AlphaService".
                var chunkExactEntities = new List<string>();
                var seenChunkExactEntities = new HashSet<string>(StringComparer.Ordinal);
                if (scoredChunks.Count > 0)
                {
                    for (int i = 0; i < scoredChunks.Count; i++)
                    {
                        string chunkId = scoredChunks[i].ChunkId;
                        if (_docEntities.TryGetValue(chunkId, out var ents))
                        {
                            for (int j = 0; j < ents.Count; j++)
                            {
                                allEntities.Add(ents[j]);
                                if (seenChunkExactEntities.Add(ents[j]))
                                {
                                    chunkExactEntities.Add(ents[j]);
                                }
                            }
                        }
                    }
                }
                else
                {
                    for (int r = 0; r < matchCount; r++)
                    {
                        string? chunkId = rawVectorResults[r].Metadata;
                        if (chunkId != null && _docEntities.TryGetValue(chunkId, out var ents))
                        {
                            for (int j = 0; j < ents.Count; j++)
                            {
                                allEntities.Add(ents[j]);
                                if (seenChunkExactEntities.Add(ents[j]))
                                {
                                    chunkExactEntities.Add(ents[j]);
                                }
                            }
                        }
                    }
                }

                if (maxTriplets > 0)
                {
                    var seenTriplets = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

                    for (int i = 0; i < chunkExactEntities.Count; i++)
                    {
                        if (relations.Count >= maxTriplets) break;
                        var triplets = _graphSearch.FindNeighborhoodTriplets(
                            chunkExactEntities[i], maxHops, maxDegree, maxTriplets - relations.Count);

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
                                if (relations.Count >= maxTriplets) break;
                            }
                        }
                    }

                    if (relations.Count < maxTriplets && maxHops > 0)
                    {
                        int queryFallbackHops = Math.Min(maxHops, 1);
                        foreach (var entity in resolvedQueryEntitySet)
                        {
                            if (relations.Count >= maxTriplets) break;
                            if (seenChunkExactEntities.Contains(entity)) continue;

                            var triplets = _graphSearch.FindNeighborhoodTriplets(
                                entity, queryFallbackHops, maxDegree, maxTriplets - relations.Count);

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
                                    if (relations.Count >= maxTriplets) break;
                                }
                            }
                        }
                    }
                }
            }
            swGraph.Stop();

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
            ExitReadLockSafe();
        }
    }

    /// <summary>
    /// Executes end-to-end GraphRAG: retrieves hybrid context, synthesizes augmented prompt
    /// using the session's native chat template with prompt-injection escaping,
    /// and streams answer tokens directly from Glacier.Inference in the exact same memory space.
    /// </summary>
    public async Task<string> AskAsync(
        string query,
        InferenceSession session,
        RagOptions? options = null,
        Action<string>? onToken = null,
        CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed), this);
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(session);

        var retrieval = await RetrieveAsync(query, options, ct).ConfigureAwait(false);
        string prompt = BuildAugmentedPrompt(query, retrieval, session);

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
            ct: ct).ConfigureAwait(false);

        return result.Text;
    }

    /// <summary>
    /// Synthesizes the full augmented prompt with sanitized ChatML formatting ready for LLM generation.
    /// Escapes control tokens in both retrieved context and user query to prevent prompt injection.
    /// </summary>
    public string BuildAugmentedPrompt(string query, HybridRetrievalResult retrieval)
    {
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(retrieval);

        string safeContext = PromptSanitizer.Sanitize(retrieval.SynthesizedContext);
        string safeQuery = PromptSanitizer.Sanitize(query);

        return $"<|im_start|>system\n" +
               $"You are an expert enterprise AI assistant. Answer the user prompt accurately using ONLY the provided verified Knowledge Graph relationships and semantic document excerpts.\n\n" +
               $"{safeContext}<|im_end|>\n" +
               $"<|im_start|>user\n" +
               $"{safeQuery}<|im_end|>\n" +
               $"<|im_start|>assistant\n";
    }

    /// <summary>
    /// Synthesizes the full augmented prompt using the target <see cref="InferenceSession"/>'s
    /// native chat template (<see cref="Glacier.Inference.Tokenizer.BpeTokenizer.FormatChat"/>)
    /// after escaping control tokens in both retrieved context and user query.
    /// </summary>
    public string BuildAugmentedPrompt(string query, HybridRetrievalResult retrieval, InferenceSession session)
    {
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(retrieval);
        ArgumentNullException.ThrowIfNull(session);

        string safeContext = PromptSanitizer.Sanitize(retrieval.SynthesizedContext);
        string safeQuery = PromptSanitizer.Sanitize(query);

        string systemPrompt =
            "You are an expert enterprise AI assistant. Answer the user prompt accurately using ONLY the provided verified Knowledge Graph relationships and semantic document excerpts.\n\n" +
            safeContext;

        return session.Tokenizer.FormatChat([
            ("system", systemPrompt),
            ("user", safeQuery)
        ]);
    }

    private InMemoryCsrGraph? GetOrCompileInMemoryCsr()
    {
        if (_graphStore.NodeCount == 0)
            return null;

        var cached = Volatile.Read(ref _csrSnapshot);
        if (cached != null)
            return cached;

        lock (_csrSync)
        {
            cached = Volatile.Read(ref _csrSnapshot);
            if (cached != null)
                return cached;

            if (_graphStore.NodeCount == 0)
                return null;

            var compiled = InMemoryCsrGraph.CompileFromForwardStar(_graphStore);
            Volatile.Write(ref _csrSnapshot, compiled);
            return compiled;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposeFlag, 1) != 0)
            return;

        // Flip sign bit so all future BeginOperation() calls fail immediately before touching _rwLock
        while (true)
        {
            int cur = Volatile.Read(ref _activeOps);
            if (cur < 0)
                break;

            if (Interlocked.CompareExchange(ref _activeOps, cur | int.MinValue, cur) == cur)
                break;
        }

        // Wait until all in-flight operations have exited _rwLock and called EndOperation()
        var spin = new SpinWait();
        while ((Volatile.Read(ref _activeOps) & int.MaxValue) != 0)
        {
            spin.SpinOnce();
        }

        if (_disposed)
            return;

        Volatile.Write(ref _disposed, true);
        Volatile.Write(ref _csrSnapshot, null);

        _embeddingModel.Dispose();
        _vectorIndex.Dispose();
        _vectorStorage.Dispose();
        _graphStore.Dispose();
        _chunksById.Clear();
        _chunkContentById.Clear();
        _docEntities.Clear();
        _entityToChunkIds.Clear();
        _canonicalEntitiesByLower.Clear();
        _rwLock.Dispose();
    }
}
