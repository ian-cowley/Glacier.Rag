namespace Glacier.Rag.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using Glacier.Graph.Storage;
using Glacier.Rag.Engine;
using Glacier.Vector.Index;
using Xunit;

/// <summary>
/// Tier-5 White-Box Adversarial Coverage Hardening for Glacier.Rag.
/// Stress-tests HippoRAG retrieval integration, multi-entity PPR scoring,
/// dense vs graph weight dynamics, and reciprocal rank fusion monotonicity.
/// </summary>
public class Tier5RagHippoRagAdversarialTests
{
    [Fact]
    public void HippoRag_MissingOrEmptyEntities_ScoresZeroOnGraphAndPreservesDenseOrdering()
    {
        string tempFile = Path.Combine(Path.GetTempPath(), $"rag_tier5_empty_{Guid.NewGuid():N}.gcsr");
        try
        {
            using (var store = new GraphStore())
            {
                store.AddNode("NodeA");
                store.AddNode("NodeB");
                store.AddEdge("NodeA", "NodeB", "LINKS", 1.0f);
                CsrGraphCompiler.CompileFromForwardStar(store, tempFile);
            }

            using var csr = new CsrGraphStore(tempFile);

            var vectorMatches = new SearchResult[]
            {
                new(0, 0.90f, "chunk-1"),
                new(1, 0.70f, "chunk-2"),
                new(2, 0.50f, "chunk-3")
            };

            var emptyQueryEntities = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var chunkEntities = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase)
            {
                ["chunk-1"] = ["NonExistentEntityX"],
                ["chunk-2"] = ["NonExistentEntityY"],
                ["chunk-3"] = ["NonExistentEntityZ"]
            };

            var contentMap = new Dictionary<string, string>
            {
                ["chunk-1"] = "Content 1",
                ["chunk-2"] = "Content 2",
                ["chunk-3"] = "Content 3"
            };

            var results = HybridGraphScorer.ScoreAndRerankHippoRag(
                vectorMatches,
                emptyQueryEntities,
                chunkEntities,
                contentMap,
                csr,
                denseWeight: 0.5f);

            Assert.Equal(3, results.Count);
            // All graph scores must be zero
            foreach (var r in results)
            {
                Assert.Equal(0.0f, r.GraphScore);
                Assert.Empty(r.RelevantRelations);
            }

            // Ordering should follow dense ranks 1, 2, 3
            Assert.Equal("chunk-1", results[0].ChunkId);
            Assert.Equal("chunk-2", results[1].ChunkId);
            Assert.Equal("chunk-3", results[2].ChunkId);
        }
        finally
        {
            if (File.Exists(tempFile)) try { File.Delete(tempFile); } catch { }
        }
    }

    [Fact]
    public void HippoRag_MultiEntityPprAccumulation_MatchesConstituentSum()
    {
        string tempFile = Path.Combine(Path.GetTempPath(), $"rag_tier5_multi_{Guid.NewGuid():N}.gcsr");
        try
        {
            using (var store = new GraphStore())
            {
                store.AddNode("SeedEntity");
                store.AddNode("EntityA");
                store.AddNode("EntityB");

                store.AddEdge("SeedEntity", "EntityA", "LINKS", 1.0f);
                store.AddEdge("SeedEntity", "EntityB", "LINKS", 1.0f);
                CsrGraphCompiler.CompileFromForwardStar(store, tempFile);
            }

            using var csr = new CsrGraphStore(tempFile);

            var vectorMatches = new SearchResult[]
            {
                new(0, 0.50f, "chunk-single-A"),
                new(1, 0.50f, "chunk-single-B"),
                new(2, 0.50f, "chunk-multi-AB")
            };

            var queryEntities = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "SeedEntity" };
            var chunkEntities = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase)
            {
                ["chunk-single-A"] = ["EntityA"],
                ["chunk-single-B"] = ["EntityB"],
                ["chunk-multi-AB"] = ["EntityA", "EntityB"]
            };

            var contentMap = new Dictionary<string, string>
            {
                ["chunk-single-A"] = "A only",
                ["chunk-single-B"] = "B only",
                ["chunk-multi-AB"] = "Both A and B"
            };

            var results = HybridGraphScorer.ScoreAndRerankHippoRag(
                vectorMatches,
                queryEntities,
                chunkEntities,
                contentMap,
                csr,
                denseWeight: 0.5f);

            var resA = results.Find(c => c.ChunkId == "chunk-single-A");
            var resB = results.Find(c => c.ChunkId == "chunk-single-B");
            var resAB = results.Find(c => c.ChunkId == "chunk-multi-AB");

            Assert.Equal("chunk-single-A", resA.ChunkId);
            Assert.Equal("chunk-single-B", resB.ChunkId);
            Assert.Equal("chunk-multi-AB", resAB.ChunkId);

            // A and B have symmetric topology from seed
            Assert.Equal(resA.GraphScore, resB.GraphScore, precision: 4);

            // Multi chunk must have GraphScore equal to sum of A + B
            Assert.Equal(resA.GraphScore + resB.GraphScore, resAB.GraphScore, precision: 4);
            Assert.Equal(2, resAB.RelevantRelations.Count);

            // Multi chunk must rank first because of higher graph relevance
            Assert.Equal("chunk-multi-AB", results[0].ChunkId);
        }
        finally
        {
            if (File.Exists(tempFile)) try { File.Delete(tempFile); } catch { }
        }
    }

    [Fact]
    public void HippoRag_WeightSpectrum_DenseOnlyVsGraphOnly()
    {
        string tempFile = Path.Combine(Path.GetTempPath(), $"rag_tier5_weights_{Guid.NewGuid():N}.gcsr");
        try
        {
            using (var store = new GraphStore())
            {
                store.AddNode("QueryTarget");
                store.AddNode("HighGraphNode");
                store.AddEdge("QueryTarget", "HighGraphNode", "STRONGLY_COUPLED", 10.0f);
                CsrGraphCompiler.CompileFromForwardStar(store, tempFile);
            }

            using var csr = new CsrGraphStore(tempFile);

            // chunk-dense has high dense score but low graph relevance
            // chunk-graph has low dense score but high graph relevance
            var vectorMatches = new SearchResult[]
            {
                new(0, 0.99f, "chunk-dense"),
                new(1, 0.10f, "chunk-graph")
            };

            var queryEntities = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "QueryTarget" };
            var chunkEntities = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase)
            {
                ["chunk-dense"] = ["UnrelatedNode"],
                ["chunk-graph"] = ["HighGraphNode"]
            };

            var contentMap = new Dictionary<string, string>
            {
                ["chunk-dense"] = "Dense content",
                ["chunk-graph"] = "Graph content"
            };

            // 1. Pure dense mode (denseWeight = 1.0f)
            var pureDense = HybridGraphScorer.ScoreAndRerankHippoRag(
                vectorMatches, queryEntities, chunkEntities, contentMap, csr, denseWeight: 1.0f);
            Assert.Equal("chunk-dense", pureDense[0].ChunkId);

            // 2. Pure graph mode (denseWeight = 0.0f)
            var pureGraph = HybridGraphScorer.ScoreAndRerankHippoRag(
                vectorMatches, queryEntities, chunkEntities, contentMap, csr, denseWeight: 0.0f);
            Assert.Equal("chunk-graph", pureGraph[0].ChunkId);
        }
        finally
        {
            if (File.Exists(tempFile)) try { File.Delete(tempFile); } catch { }
        }
    }

    [Fact]
    public void HippoRag_ReciprocalRankFusion_MonotonicityAndGraphCandidateExpansion()
    {
        string tempFile = Path.Combine(Path.GetTempPath(), $"rag_tier5_rrf_{Guid.NewGuid():N}.gcsr");
        try
        {
            using (var store = new GraphStore())
            {
                store.AddNode("Seed1");
                store.AddNode("ExpandedEntity");
                store.AddEdge("Seed1", "ExpandedEntity", "CONNECTED", 5.0f);
                CsrGraphCompiler.CompileFromForwardStar(store, tempFile);
            }

            using var csr = new CsrGraphStore(tempFile);

            var vectorMatches = new SearchResult[]
            {
                new(0, 0.85f, "dense-doc-1"),
                new(1, 0.65f, "dense-doc-2")
            };

            // "graph-expanded-doc" was NOT retrieved by dense search, but injected via graphCandidateIds!
            string[] graphCandidates = ["graph-expanded-doc"];

            var queryEntities = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "Seed1" };
            var chunkEntities = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase)
            {
                ["dense-doc-1"] = ["Seed1"],
                ["dense-doc-2"] = [],
                ["graph-expanded-doc"] = ["ExpandedEntity"]
            };

            var contentMap = new Dictionary<string, string>
            {
                ["dense-doc-1"] = "Dense Doc 1",
                ["dense-doc-2"] = "Dense Doc 2",
                ["graph-expanded-doc"] = "Graph Expanded Doc"
            };

            var results = HybridGraphScorer.ScoreAndRerankHippoRag(
                vectorMatches,
                queryEntities,
                chunkEntities,
                contentMap,
                csr,
                denseWeight: 0.4f,
                graphCandidateIds: graphCandidates);

            Assert.Equal(3, results.Count);

            // Verify graph-expanded-doc is present
            Assert.Contains(results, r => r.ChunkId == "graph-expanded-doc");

            // Verify strict descending monotonicity of HybridScore
            for (int i = 0; i < results.Count - 1; i++)
            {
                Assert.True(results[i].HybridScore >= results[i + 1].HybridScore,
                    $"RRF monotonicity violation at rank {i}: {results[i].HybridScore} < {results[i + 1].HybridScore}");
            }
        }
        finally
        {
            if (File.Exists(tempFile)) try { File.Delete(tempFile); } catch { }
        }
    }
}
