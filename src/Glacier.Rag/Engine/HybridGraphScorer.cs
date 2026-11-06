namespace Glacier.Rag.Engine;

using System;
using System.Collections.Generic;
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

        // 2. Compute Graph Relevance Scores for every candidate chunk (dense union graph-expanded)
        var graphScores = new Dictionary<string, (float Score, List<GraphRelation> Relations)>(candidateIds.Count, StringComparer.OrdinalIgnoreCase);
        foreach (var candidateId in candidateIds)
        {
            if (!chunkEntities.TryGetValue(candidateId, out var ents) || queryEntities.Count == 0)
            {
                graphScores[candidateId] = (0f, []);
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
}
