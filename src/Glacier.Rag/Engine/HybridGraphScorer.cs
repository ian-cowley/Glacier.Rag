namespace Glacier.Rag.Engine;

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Glacier.Graph.Kernels;
using Glacier.Graph.Storage;
using Glacier.Graph.Traversal;
using Glacier.Vector.Index;

/// <summary>
/// Authentic hybrid retrieval reranker evaluating dense vector similarity,
/// knowledge graph shortest-path hop proximity, and Reciprocal Rank Fusion (RRF)
/// across the union of dense vector matches and graph-expanded candidate chunks.
/// </summary>
public static class HybridGraphScorer
{
    public readonly record struct ScoredChunk(
        string ChunkId,
        string Content,
        float DenseScore,
        float GraphScore,
        float HybridScore,
        IReadOnlyList<GraphRelation> RelevantRelations);

    public static List<ScoredChunk> ScoreAndRerank(
        SearchResult[] vectorMatches,
        IReadOnlySet<string> queryEntities,
        IReadOnlyDictionary<string, List<string>> chunkEntities,
        IReadOnlyDictionary<string, string> chunkContentById,
        GraphStore graphStore,
        GraphSearch graphSearch,
        float denseWeight = 0.5f,
        int kSmoothing = 60,
        IReadOnlyCollection<string>? graphCandidateIds = null,
        int maxResults = 0)
    {
        int extraCount = graphCandidateIds?.Count ?? 0;
        if (vectorMatches.Length == 0 && extraCount == 0)
            return [];

        var candidateIds = new List<string>(vectorMatches.Length + extraCount);
        var seenCandidates = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        // 1. Compute Dense Ranks & Scores
        var denseRanks = new Dictionary<string, int>(vectorMatches.Length, StringComparer.OrdinalIgnoreCase);
        var denseScores = new Dictionary<string, float>(vectorMatches.Length, StringComparer.OrdinalIgnoreCase);

        for (int r = 0; r < vectorMatches.Length; r++)
        {
            string id = vectorMatches[r].Metadata ?? string.Empty;
            if (string.IsNullOrEmpty(id)) continue;

            if (seenCandidates.Add(id))
            {
                candidateIds.Add(id);
                denseRanks[id] = r + 1;
                denseScores[id] = vectorMatches[r].Score;
            }
        }

        if (graphCandidateIds != null)
        {
            foreach (var gId in graphCandidateIds)
            {
                if (!string.IsNullOrEmpty(gId) && seenCandidates.Add(gId))
                {
                    candidateIds.Add(gId);
                }
            }
        }

        if (candidateIds.Count == 0)
            return [];

        // 2. Resolve Query Entities to exact GraphStore node IDs (Ordinal first, OrdinalIgnoreCase fallback)
        var resolvedQueryEntities = new HashSet<string>(StringComparer.Ordinal);
        if (queryEntities.Count > 0)
        {
            List<string>? unresolvedQueryEntities = null;
            foreach (var qEnt in queryEntities)
            {
                if (graphStore.TryGetInternalId(qEnt, out _))
                {
                    resolvedQueryEntities.Add(qEnt);
                }
                else
                {
                    unresolvedQueryEntities ??= new List<string>();
                    unresolvedQueryEntities.Add(qEnt);
                }
            }

            if (unresolvedQueryEntities != null && graphStore.NodeCount > 0)
            {
                bool[] matched = new bool[unresolvedQueryEntities.Count];
                using (graphStore.AcquireReadLock())
                {
                    int nodeCount = graphStore.NodeCount;
                    for (int i = 1; i <= nodeCount; i++)
                    {
                        string extId = graphStore.GetExternalId(i);
                        if (string.IsNullOrEmpty(extId)) continue;
                        for (int u = 0; u < unresolvedQueryEntities.Count; u++)
                        {
                            if (extId.Equals(unresolvedQueryEntities[u], StringComparison.OrdinalIgnoreCase))
                            {
                                resolvedQueryEntities.Add(extId);
                                matched[u] = true;
                            }
                        }
                    }
                }

                for (int u = 0; u < unresolvedQueryEntities.Count; u++)
                {
                    if (!matched[u])
                    {
                        resolvedQueryEntities.Add(unresolvedQueryEntities[u]);
                    }
                }
            }
            else if (unresolvedQueryEntities != null)
            {
                for (int u = 0; u < unresolvedQueryEntities.Count; u++)
                {
                    resolvedQueryEntities.Add(unresolvedQueryEntities[u]);
                }
            }
        }

        // 2b. Compute Graph Relevance Scores for every candidate chunk (dense union graph-expanded)
        var graphScores = new Dictionary<string, (float Score, List<GraphRelation> Relations)>(candidateIds.Count, StringComparer.OrdinalIgnoreCase);
        foreach (var candidateId in candidateIds)
        {
            if (!chunkEntities.TryGetValue(candidateId, out var ents) || resolvedQueryEntities.Count == 0)
            {
                graphScores[candidateId] = (0f, []);
                continue;
            }

            float gScore = 0f;
            var rels = new List<GraphRelation>();

            foreach (var qEnt in resolvedQueryEntities)
            {
                foreach (var dEnt in ents)
                {
                    if (qEnt.Equals(dEnt, StringComparison.OrdinalIgnoreCase))
                    {
                        gScore += 1.0f;
                        continue;
                    }

                    var path = graphSearch.FindShortestPath(qEnt, dEnt);
                    if (path.Count <= 1)
                    {
                        path = graphSearch.FindShortestPath(dEnt, qEnt);
                    }

                    if (path.Count > 1 && path.Count <= 4)
                    {
                        int hops = path.Count - 1;
                        gScore += 1.0f / (1.0f + hops);
                        rels.Add(new GraphRelation
                        {
                            Source = qEnt,
                            Target = dEnt,
                            Relation = $"PATH_{hops}_HOPS"
                        });
                    }
                }
            }

            graphScores[candidateId] = (gScore, rels);
        }

        // 3. Compute Graph Ranks (chunks with 0 graph score share worst rank = candidateIds.Count + 1)
        var sortedByGraph = new List<KeyValuePair<string, (float Score, List<GraphRelation> Relations)>>(graphScores);
        sortedByGraph.Sort(static (a, b) => b.Value.Score.CompareTo(a.Value.Score));

        int missingRank = candidateIds.Count + 1;
        var graphRanks = new Dictionary<string, int>(sortedByGraph.Count, StringComparer.OrdinalIgnoreCase);
        for (int r = 0; r < sortedByGraph.Count; r++)
        {
            if (sortedByGraph[r].Value.Score > 0f)
            {
                graphRanks[sortedByGraph[r].Key] = r + 1;
            }
        }

        // 4. Reciprocal Rank Fusion (RRF)
        float graphWeight = 1.0f - denseWeight;
        var scoredList = new List<ScoredChunk>(candidateIds.Count);

        foreach (var candidateId in candidateIds)
        {
            int rDense = denseRanks.GetValueOrDefault(candidateId, missingRank);
            int rGraph = graphRanks.GetValueOrDefault(candidateId, missingRank);

            float rrfScore = (denseWeight / (kSmoothing + rDense)) + (graphWeight / (kSmoothing + rGraph));
            float dScore = denseScores.GetValueOrDefault(candidateId, 0f);
            var (gScore, rels) = graphScores[candidateId];

            string content = chunkContentById.TryGetValue(candidateId, out var c) ? c : candidateId;
            scoredList.Add(new ScoredChunk(candidateId, content, dScore, gScore, rrfScore, rels));
        }

        scoredList.Sort(static (a, b) =>
        {
            int cmp = b.HybridScore.CompareTo(a.HybridScore);
            if (cmp != 0) return cmp;
            cmp = b.GraphScore.CompareTo(a.GraphScore);
            return cmp != 0 ? cmp : b.DenseScore.CompareTo(a.DenseScore);
        });

        if (maxResults > 0 && scoredList.Count > maxResults)
        {
            scoredList.RemoveRange(maxResults, scoredList.Count - maxResults);
        }

        return scoredList;
    }

    /// <summary>
    /// High-performance HippoRAG hybrid retrieval reranker.
    /// Executes a single Personalized PageRank pass over query seed entities,
    /// scoring all candidate document chunks directly via constituent entity PPR scores,
    /// followed by Reciprocal Rank Fusion (RRF) with dense vector scores.
    /// </summary>
    public static List<ScoredChunk> ScoreAndRerankHippoRag(
        SearchResult[] vectorMatches,
        IReadOnlySet<string> queryEntities,
        IReadOnlyDictionary<string, List<string>> chunkEntities,
        IReadOnlyDictionary<string, string> chunkContentById,
        CsrGraphStore csrStore,
        float denseWeight = 0.5f,
        int kSmoothing = 60,
        IReadOnlyCollection<string>? graphCandidateIds = null,
        int maxResults = 0,
        float alpha = 0.85f,
        int pprIterations = 20)
    {
        ArgumentNullException.ThrowIfNull(vectorMatches);
        ArgumentNullException.ThrowIfNull(queryEntities);
        ArgumentNullException.ThrowIfNull(chunkEntities);
        ArgumentNullException.ThrowIfNull(chunkContentById);
        ArgumentNullException.ThrowIfNull(csrStore);

        int extraCount = graphCandidateIds?.Count ?? 0;
        if (vectorMatches.Length == 0 && extraCount == 0)
            return [];

        var candidateIds = new List<string>(vectorMatches.Length + extraCount);
        var seenCandidates = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        // 1. Compute Dense Ranks & Scores
        var denseRanks = new Dictionary<string, int>(vectorMatches.Length, StringComparer.OrdinalIgnoreCase);
        var denseScores = new Dictionary<string, float>(vectorMatches.Length, StringComparer.OrdinalIgnoreCase);

        for (int r = 0; r < vectorMatches.Length; r++)
        {
            string id = vectorMatches[r].Metadata ?? string.Empty;
            if (string.IsNullOrEmpty(id)) continue;

            if (seenCandidates.Add(id))
            {
                candidateIds.Add(id);
                denseRanks[id] = r + 1;
                denseScores[id] = vectorMatches[r].Score;
            }
        }

        if (graphCandidateIds != null)
        {
            foreach (var gId in graphCandidateIds)
            {
                if (!string.IsNullOrEmpty(gId) && seenCandidates.Add(gId))
                {
                    candidateIds.Add(gId);
                }
            }
        }

        if (candidateIds.Count == 0)
            return [];

        // 2. Personalized PageRank across Query Seed Entities (Ordinal first, OrdinalIgnoreCase fallback)
        var seedEntities = new List<int>(queryEntities.Count);
        var seenSeedIds = new HashSet<int>();
        List<string>? unresolvedQueryEntities = null;
        foreach (var qEnt in queryEntities)
        {
            if (csrStore.TryGetInternalId(qEnt, out int internalId))
            {
                if (seenSeedIds.Add(internalId))
                {
                    seedEntities.Add(internalId);
                }
            }
            else
            {
                unresolvedQueryEntities ??= new List<string>();
                unresolvedQueryEntities.Add(qEnt);
            }
        }

        if (unresolvedQueryEntities != null && csrStore.NodeCount > 0)
        {
            int csrNodeCount = csrStore.NodeCount;
            for (int i = 1; i <= csrNodeCount; i++)
            {
                string extId = csrStore.GetExternalId(i);
                if (string.IsNullOrEmpty(extId)) continue;
                for (int u = 0; u < unresolvedQueryEntities.Count; u++)
                {
                    if (extId.Equals(unresolvedQueryEntities[u], StringComparison.OrdinalIgnoreCase) &&
                        seenSeedIds.Add(i))
                    {
                        seedEntities.Add(i);
                    }
                }
            }
        }

        float[]? rentedPpr = null;
        int bufSize = csrStore.NodeCount + 1;
        Span<float> pprScores = bufSize <= 1024
            ? stackalloc float[bufSize]
            : (rentedPpr = ArrayPool<float>.Shared.Rent(bufSize)).AsSpan(0, bufSize);

        try
        {
            if (seedEntities.Count > 0 && csrStore.NodeCount > 0)
            {
                CsrKernels.PersonalizedPageRank(
                    csrStore,
                    CollectionsMarshal.AsSpan(seedEntities),
                    ReadOnlySpan<float>.Empty,
                    pprScores,
                    alpha: alpha,
                    maxIterations: pprIterations);
            }
            else
            {
                pprScores.Clear();
            }

            // 3. Compute Graph Relevance Scores for candidate chunks via constituent entity PPR scores
            var graphScores = new Dictionary<string, (float Score, List<GraphRelation> Relations)>(candidateIds.Count, StringComparer.OrdinalIgnoreCase);
            foreach (var candidateId in candidateIds)
            {
                if (!chunkEntities.TryGetValue(candidateId, out var ents) || ents.Count == 0 || seedEntities.Count == 0)
                {
                    graphScores[candidateId] = (0f, []);
                    continue;
                }

                float gScore = 0f;
                var rels = new List<GraphRelation>();

                foreach (var dEnt in ents)
                {
                    if (csrStore.TryGetInternalId(dEnt, out int dInternalId) && (uint)dInternalId < (uint)pprScores.Length)
                    {
                        float p = pprScores[dInternalId];
                        if (p > 0f)
                        {
                            gScore += p;
                            rels.Add(new GraphRelation
                            {
                                Source = "QUERY",
                                Target = dEnt,
                                Relation = $"PPR_{p:F5}"
                            });
                        }
                    }
                }

                graphScores[candidateId] = (gScore, rels);
            }

            return BuildHippoRagRerankedResults(
                candidateIds,
                denseRanks,
                denseScores,
                graphScores,
                chunkContentById,
                denseWeight,
                kSmoothing,
                maxResults);
        }
        finally
        {
            if (rentedPpr != null)
            {
                ArrayPool<float>.Shared.Return(rentedPpr);
            }
        }
    }

    internal static List<ScoredChunk> ScoreAndRerankHippoRag(
        SearchResult[] vectorMatches,
        IReadOnlySet<string> queryEntities,
        IReadOnlyDictionary<string, List<string>> chunkEntities,
        IReadOnlyDictionary<string, string> chunkContentById,
        InMemoryCsrGraph csrGraph,
        float denseWeight = 0.5f,
        int kSmoothing = 60,
        IReadOnlyCollection<string>? graphCandidateIds = null,
        int maxResults = 0,
        float alpha = 0.85f,
        int pprIterations = 20)
    {
        ArgumentNullException.ThrowIfNull(csrGraph);

        return ScoreAndRerankHippoRagCore(
            vectorMatches,
            queryEntities,
            chunkEntities,
            chunkContentById,
            csrGraph.RowOffsets,
            csrGraph.ColumnIndices,
            csrGraph.EdgeWeights,
            csrGraph.NodeCount,
            csrGraph.ExternalToInternalId,
            csrGraph.ExternalToInternalIdsIgnoreCase,
            denseWeight,
            kSmoothing,
            graphCandidateIds,
            maxResults,
            alpha,
            pprIterations);
    }

    /// <summary>
    /// High-performance HippoRAG hybrid retrieval reranker operating directly over contiguous
    /// in-memory CSR spans without disk temp files or paged cache overhead.
    /// </summary>
    public static List<ScoredChunk> ScoreAndRerankHippoRag(
        SearchResult[] vectorMatches,
        IReadOnlySet<string> queryEntities,
        IReadOnlyDictionary<string, List<string>> chunkEntities,
        IReadOnlyDictionary<string, string> chunkContentById,
        ReadOnlySpan<int> rowOffsets,
        ReadOnlySpan<int> columnIndices,
        ReadOnlySpan<float> edgeWeights,
        int nodeCount,
        IReadOnlyDictionary<string, int> externalToInternalId,
        float denseWeight = 0.5f,
        int kSmoothing = 60,
        IReadOnlyCollection<string>? graphCandidateIds = null,
        int maxResults = 0,
        float alpha = 0.85f,
        int pprIterations = 20)
    {
        return ScoreAndRerankHippoRagCore(
            vectorMatches,
            queryEntities,
            chunkEntities,
            chunkContentById,
            rowOffsets,
            columnIndices,
            edgeWeights,
            nodeCount,
            externalToInternalId,
            externalToInternalIdsIgnoreCase: null,
            denseWeight,
            kSmoothing,
            graphCandidateIds,
            maxResults,
            alpha,
            pprIterations);
    }

    private static List<ScoredChunk> ScoreAndRerankHippoRagCore(
        SearchResult[] vectorMatches,
        IReadOnlySet<string> queryEntities,
        IReadOnlyDictionary<string, List<string>> chunkEntities,
        IReadOnlyDictionary<string, string> chunkContentById,
        ReadOnlySpan<int> rowOffsets,
        ReadOnlySpan<int> columnIndices,
        ReadOnlySpan<float> edgeWeights,
        int nodeCount,
        IReadOnlyDictionary<string, int> externalToInternalId,
        IReadOnlyDictionary<string, List<int>>? externalToInternalIdsIgnoreCase,
        float denseWeight,
        int kSmoothing,
        IReadOnlyCollection<string>? graphCandidateIds,
        int maxResults,
        float alpha,
        int pprIterations)
    {
        ArgumentNullException.ThrowIfNull(vectorMatches);
        ArgumentNullException.ThrowIfNull(queryEntities);
        ArgumentNullException.ThrowIfNull(chunkEntities);
        ArgumentNullException.ThrowIfNull(chunkContentById);
        ArgumentNullException.ThrowIfNull(externalToInternalId);
        ArgumentOutOfRangeException.ThrowIfNegative(nodeCount);
        if (nodeCount > 0 && rowOffsets.Length < nodeCount + 2)
            throw new ArgumentException("rowOffsets length must be >= nodeCount + 2.", nameof(rowOffsets));

        int extraCount = graphCandidateIds?.Count ?? 0;
        if (vectorMatches.Length == 0 && extraCount == 0)
            return [];

        var candidateIds = new List<string>(vectorMatches.Length + extraCount);
        var seenCandidates = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        // 1. Compute Dense Ranks & Scores
        var denseRanks = new Dictionary<string, int>(vectorMatches.Length, StringComparer.OrdinalIgnoreCase);
        var denseScores = new Dictionary<string, float>(vectorMatches.Length, StringComparer.OrdinalIgnoreCase);

        for (int r = 0; r < vectorMatches.Length; r++)
        {
            string id = vectorMatches[r].Metadata ?? string.Empty;
            if (string.IsNullOrEmpty(id)) continue;

            if (seenCandidates.Add(id))
            {
                candidateIds.Add(id);
                denseRanks[id] = r + 1;
                denseScores[id] = vectorMatches[r].Score;
            }
        }

        if (graphCandidateIds != null)
        {
            foreach (var gId in graphCandidateIds)
            {
                if (!string.IsNullOrEmpty(gId) && seenCandidates.Add(gId))
                {
                    candidateIds.Add(gId);
                }
            }
        }

        if (candidateIds.Count == 0)
            return [];

        // 2. Personalized PageRank across Query Seed Entities (Ordinal first, OrdinalIgnoreCase fallback)
        var seedEntities = new List<int>(queryEntities.Count);
        var seenSeedIds = new HashSet<int>();
        List<string>? unresolvedQueryEntities = null;
        foreach (var qEnt in queryEntities)
        {
            if (externalToInternalId.TryGetValue(qEnt, out int internalId))
            {
                if (seenSeedIds.Add(internalId))
                {
                    seedEntities.Add(internalId);
                }
            }
            else if (externalToInternalIdsIgnoreCase != null)
            {
                if (externalToInternalIdsIgnoreCase.TryGetValue(qEnt, out var ids))
                {
                    for (int i = 0; i < ids.Count; i++)
                    {
                        if (seenSeedIds.Add(ids[i]))
                        {
                            seedEntities.Add(ids[i]);
                        }
                    }
                }
            }
            else
            {
                unresolvedQueryEntities ??= new List<string>();
                unresolvedQueryEntities.Add(qEnt);
            }
        }

        if (unresolvedQueryEntities != null && externalToInternalId.Count > 0)
        {
            foreach (var kv in externalToInternalId)
            {
                for (int i = 0; i < unresolvedQueryEntities.Count; i++)
                {
                    if (kv.Key.Equals(unresolvedQueryEntities[i], StringComparison.OrdinalIgnoreCase) &&
                        seenSeedIds.Add(kv.Value))
                    {
                        seedEntities.Add(kv.Value);
                    }
                }
            }
        }

        float[]? rentedPpr = null;
        int bufSize = nodeCount + 1;
        Span<float> pprScores = bufSize <= 1024
            ? stackalloc float[bufSize]
            : (rentedPpr = ArrayPool<float>.Shared.Rent(bufSize)).AsSpan(0, bufSize);

        try
        {
            if (seedEntities.Count > 0 && nodeCount > 0)
            {
                CsrKernels.PersonalizedPageRank(
                    rowOffsets,
                    columnIndices,
                    edgeWeights,
                    nodeCount,
                    CollectionsMarshal.AsSpan(seedEntities),
                    ReadOnlySpan<float>.Empty,
                    pprScores,
                    alpha: alpha,
                    maxIterations: pprIterations);
            }
            else
            {
                pprScores.Clear();
            }

            // 3. Compute Graph Relevance Scores for candidate chunks via constituent entity PPR scores
            var graphScores = new Dictionary<string, (float Score, List<GraphRelation> Relations)>(candidateIds.Count, StringComparer.OrdinalIgnoreCase);
            foreach (var candidateId in candidateIds)
            {
                if (!chunkEntities.TryGetValue(candidateId, out var ents) || ents.Count == 0 || seedEntities.Count == 0)
                {
                    graphScores[candidateId] = (0f, []);
                    continue;
                }

                float gScore = 0f;
                var rels = new List<GraphRelation>();

                foreach (var dEnt in ents)
                {
                    bool found = externalToInternalId.TryGetValue(dEnt, out int dInternalId);
                    if (!found && externalToInternalIdsIgnoreCase != null &&
                        externalToInternalIdsIgnoreCase.TryGetValue(dEnt, out var fallbackIds) &&
                        fallbackIds.Count > 0)
                    {
                        dInternalId = fallbackIds[0];
                        found = true;
                    }

                    if (found && (uint)dInternalId < (uint)pprScores.Length)
                    {
                        float p = pprScores[dInternalId];
                        if (p > 0f)
                        {
                            gScore += p;
                            rels.Add(new GraphRelation
                            {
                                Source = "QUERY",
                                Target = dEnt,
                                Relation = $"PPR_{p:F5}"
                            });
                        }
                    }
                }

                graphScores[candidateId] = (gScore, rels);
            }

            return BuildHippoRagRerankedResults(
                candidateIds,
                denseRanks,
                denseScores,
                graphScores,
                chunkContentById,
                denseWeight,
                kSmoothing,
                maxResults);
        }
        finally
        {
            if (rentedPpr != null)
            {
                ArrayPool<float>.Shared.Return(rentedPpr);
            }
        }
    }

    private static List<ScoredChunk> BuildHippoRagRerankedResults(
        List<string> candidateIds,
        Dictionary<string, int> denseRanks,
        Dictionary<string, float> denseScores,
        Dictionary<string, (float Score, List<GraphRelation> Relations)> graphScores,
        IReadOnlyDictionary<string, string> chunkContentById,
        float denseWeight,
        int kSmoothing,
        int maxResults)
    {
        // 4. Compute Graph Ranks
        var sortedByGraph = new List<KeyValuePair<string, (float Score, List<GraphRelation> Relations)>>(graphScores);
        sortedByGraph.Sort(static (a, b) => b.Value.Score.CompareTo(a.Value.Score));

        int missingRank = candidateIds.Count + 1;
        var graphRanks = new Dictionary<string, int>(sortedByGraph.Count, StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < sortedByGraph.Count; i++)
        {
            var kv = sortedByGraph[i];
            graphRanks[kv.Key] = kv.Value.Score > 0f ? (i + 1) : missingRank;
        }

        // 5. Reciprocal Rank Fusion (RRF)
        float graphWeight = 1.0f - denseWeight;
        var scoredList = new List<ScoredChunk>(candidateIds.Count);

        foreach (var candidateId in candidateIds)
        {
            int rDense = denseRanks.GetValueOrDefault(candidateId, missingRank);
            int rGraph = graphRanks.GetValueOrDefault(candidateId, missingRank);

            float rrfScore = (denseWeight / (kSmoothing + rDense)) + (graphWeight / (kSmoothing + rGraph));
            float dScore = denseScores.GetValueOrDefault(candidateId, 0f);
            var (gScore, rels) = graphScores[candidateId];

            string content = chunkContentById.TryGetValue(candidateId, out var c) ? c : candidateId;
            scoredList.Add(new ScoredChunk(candidateId, content, dScore, gScore, rrfScore, rels));
        }

        scoredList.Sort(static (a, b) =>
        {
            int cmp = b.HybridScore.CompareTo(a.HybridScore);
            if (cmp != 0) return cmp;
            cmp = b.GraphScore.CompareTo(a.GraphScore);
            return cmp != 0 ? cmp : b.DenseScore.CompareTo(a.DenseScore);
        });

        if (maxResults > 0 && scoredList.Count > maxResults)
        {
            scoredList.RemoveRange(maxResults, scoredList.Count - maxResults);
        }

        return scoredList;
    }
}
