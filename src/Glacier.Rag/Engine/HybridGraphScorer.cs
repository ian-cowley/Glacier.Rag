namespace Glacier.Rag.Engine;

using System;
using System.Collections.Generic;
using Glacier.Graph.Storage;
using Glacier.Graph.Traversal;
using Glacier.Vector.Index;

/// <summary>
/// Authentic hybrid retrieval reranker evaluating dense vector similarity,
/// knowledge graph shortest-path hop proximity, and Reciprocal Rank Fusion (RRF).
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
        int kSmoothing = 60)
    {
        if (vectorMatches.Length == 0) return new List<ScoredChunk>();

        var scoredList = new List<ScoredChunk>(vectorMatches.Length);

        // 1. Compute Dense Ranks
        var denseRanks = new Dictionary<string, int>(vectorMatches.Length, StringComparer.Ordinal);
        for (int r = 0; r < vectorMatches.Length; r++)
        {
            if (!string.IsNullOrEmpty(vectorMatches[r].Metadata))
            {
                denseRanks[vectorMatches[r].Metadata] = r + 1;
            }
        }

        // 2. Compute Graph Relevance Scores for each chunk
        var graphScores = new Dictionary<string, (float Score, List<GraphRelation> Relations)>(vectorMatches.Length, StringComparer.Ordinal);
        foreach (var match in vectorMatches)
        {
            string contentOrId = match.Metadata ?? string.Empty;
            if (!chunkEntities.TryGetValue(contentOrId, out var ents))
            {
                graphScores[contentOrId] = (0f, new List<GraphRelation>());
                continue;
            }

            float gScore = 0f;
            var rels = new List<GraphRelation>();

            foreach (var qEnt in queryEntities)
            {
                foreach (var dEnt in ents)
                {
                    if (qEnt.Equals(dEnt, StringComparison.OrdinalIgnoreCase))
                    {
                        gScore += 1.0f; // Direct entity hit
                        continue;
                    }

                    var path = graphSearch.FindShortestPath(qEnt, dEnt);
                    if (path.Count <= 1)
                    {
                        path = graphSearch.FindShortestPath(dEnt, qEnt);
                    }

                    if (path.Count > 1 && path.Count <= 4) // within 3 hops
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
            graphScores[contentOrId] = (gScore, rels);
        }

        // 3. Compute Graph Ranks
        var sortedByGraph = new List<KeyValuePair<string, (float Score, List<GraphRelation> Relations)>>(graphScores);
        sortedByGraph.Sort((a, b) => b.Value.Score.CompareTo(a.Value.Score));
        var graphRanks = new Dictionary<string, int>(sortedByGraph.Count, StringComparer.Ordinal);
        for (int r = 0; r < sortedByGraph.Count; r++)
        {
            graphRanks[sortedByGraph[r].Key] = r + 1;
        }

        // 4. Reciprocal Rank Fusion (RRF)
        float graphWeight = 1.0f - denseWeight;
        foreach (var match in vectorMatches)
        {
            string contentOrId = match.Metadata ?? string.Empty;
            int rDense = denseRanks.GetValueOrDefault(contentOrId, vectorMatches.Length + 1);
            int rGraph = graphRanks.GetValueOrDefault(contentOrId, vectorMatches.Length + 1);

            float rrfScore = (denseWeight / (kSmoothing + rDense)) + (graphWeight / (kSmoothing + rGraph));
            var (gScore, rels) = graphScores.GetValueOrDefault(contentOrId, (0f, new List<GraphRelation>()));

            string content = chunkContentById.TryGetValue(contentOrId, out var c) ? c : contentOrId;
            scoredList.Add(new ScoredChunk(contentOrId, content, match.Score, gScore, rrfScore, rels));
        }

        // Sort descending by hybrid RRF score
        scoredList.Sort((a, b) => b.HybridScore.CompareTo(a.HybridScore));
        return scoredList;
    }
}
