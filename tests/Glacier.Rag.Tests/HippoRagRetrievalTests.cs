namespace Glacier.Rag.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using Glacier.Graph.Storage;
using Glacier.Rag.Engine;
using Glacier.Vector.Index;
using Xunit;

public class HippoRagRetrievalTests
{
    [Fact]
    public void ScoreAndRerankHippoRag_ReplacesPairwiseBfs_ReturnsOptimalChunks()
    {
        string tempFile = Path.Combine(Path.GetTempPath(), $"hipporag_test_{Guid.NewGuid():N}.gcsr");
        try
        {
            // 1. Build Forward Star Graph and compile to CSR
            using (var store = new GraphStore())
            {
                store.AddNode("OrderService", "Service");
                store.AddNode("PaymentGateway", "Service");
                store.AddNode("FraudDetection", "Service");
                store.AddNode("Database", "Storage");

                store.AddEdge("OrderService", "PaymentGateway", "CALLS", 1.0f);
                store.AddEdge("PaymentGateway", "FraudDetection", "CALLS", 1.0f);
                store.AddEdge("PaymentGateway", "Database", "PERSISTS", 0.5f);

                CsrGraphCompiler.CompileFromForwardStar(store, tempFile);
            }

            // 2. Load CSR Graph Store
            using (var csr = new CsrGraphStore(tempFile))
            {
                var vectorMatches = new SearchResult[]
                {
                    new(0, 0.95f, "chunk-1"),
                    new(1, 0.80f, "chunk-2"),
                    new(2, 0.40f, "chunk-3")
                };

                var queryEntities = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "OrderService" };
                var chunkEntities = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase)
                {
                    ["chunk-1"] = ["OrderService", "PaymentGateway"],
                    ["chunk-2"] = ["FraudDetection"],
                    ["chunk-3"] = ["UnrelatedService"]
                };

                var contentMap = new Dictionary<string, string>
                {
                    ["chunk-1"] = "OrderService dispatches checkout to PaymentGateway.",
                    ["chunk-2"] = "FraudDetection evaluates high-risk credit transactions.",
                    ["chunk-3"] = "Unrelated background monitoring daemon."
                };

                // 3. Execute HippoRAG PPR Hybrid Scoring
                var results = HybridGraphScorer.ScoreAndRerankHippoRag(
                    vectorMatches,
                    queryEntities,
                    chunkEntities,
                    contentMap,
                    csr,
                    denseWeight: 0.5f,
                    kSmoothing: 60);

                Assert.Equal(3, results.Count);

                // Chunk 1 has seed and 1-hop neighbor -> highest graph score
                Assert.Equal("chunk-1", results[0].ChunkId);
                Assert.True(results[0].GraphScore > 0f);

                // Chunk 2 has 2-hop neighbor -> positive graph score from stationary distribution
                Assert.True(results[1].GraphScore > 0f);

                // Chunk 3 has no connection to query entities -> zero graph score
                var chunk3 = results.Find(c => c.ChunkId == "chunk-3");
                Assert.Equal(0f, chunk3.GraphScore);
            }
        }
        finally
        {
            if (File.Exists(tempFile))
            {
                try { File.Delete(tempFile); } catch { }
            }
        }
    }
}
