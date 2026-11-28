namespace Glacier.Rag.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Graph.Storage;
using Glacier.Rag.Chunking;
using Glacier.Rag.Embeddings;
using Glacier.Rag.Engine;
using Glacier.Rag.Extraction;
using Glacier.Vector.Index;
using Xunit;

public sealed class RagHardeningTests
{
    private sealed class DeterministicLookupEmbeddingModel : IEmbeddingModel
    {
        private readonly Dictionary<string, float[]> _vectors;
        public int Dimensions { get; }

        public DeterministicLookupEmbeddingModel(int dimensions, Dictionary<string, float[]> vectors)
        {
            Dimensions = dimensions;
            _vectors = vectors;
        }

        public void GenerateEmbedding(ReadOnlySpan<char> text, Span<float> destination)
        {
            destination.Slice(0, Dimensions).Clear();
            string s = text.ToString();
            foreach (var kv in _vectors)
            {
                if (s.Contains(kv.Key, StringComparison.OrdinalIgnoreCase))
                {
                    kv.Value.AsSpan(0, Dimensions).CopyTo(destination);
                    return;
                }
            }
        }

        public void Dispose() { }
    }

    private sealed class BatchStubHandler : HttpMessageHandler
    {
        private readonly Func<string, int, CancellationToken, Task<HttpResponseMessage>> _responder;
        private int _callCount;
        public List<string> RequestBodies { get; } = [];

        public BatchStubHandler(Func<string, int, CancellationToken, Task<HttpResponseMessage>> responder)
        {
            _responder = responder;
        }

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            string body = request.Content != null
                ? await request.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false)
                : string.Empty;
            lock (RequestBodies)
            {
                RequestBodies.Add(body);
            }
            int call = Interlocked.Increment(ref _callCount);
            return await _responder(body, call, cancellationToken).ConfigureAwait(false);
        }
    }

    [Fact]
    public async Task GraphRagEngine_ConcurrentReadWriteDispose_ZeroUafAndZeroTempGcsrFiles()
    {
        string tempDir = Path.GetTempPath();
        var beforeGcsr = Directory.GetFiles(tempDir, "glacier_rag_*.gcsr").ToHashSet(StringComparer.OrdinalIgnoreCase);

        for (int round = 0; round < 5; round++)
        {
            var rag = new GraphRagEngine(new FastHashEmbeddingModel(128));
            for (int i = 0; i < 8; i++)
            {
                rag.IndexDocument($"INIT_{i}", $"OrderService_{i} routes to PaymentGateway_{i % 3}.");
            }

            using var startGate = new ManualResetEventSlim(false);
            var exceptions = new System.Collections.Concurrent.ConcurrentBag<Exception>();

            var tasks = new Task[16];
            for (int t = 0; t < tasks.Length; t++)
            {
                int tid = t;
                tasks[t] = Task.Run(() =>
                {
                    startGate.Wait();
                    try
                    {
                        if (tid == 0)
                        {
                            Thread.Sleep(2);
                            rag.Dispose();
                        }
                        else if (tid % 4 == 0)
                        {
                            for (int j = 0; j < 10; j++)
                            {
                                rag.IndexDocument($"W_{tid}_{j}", $"WorkerNode_{tid}_{j} connects to OrderService_0.");
                            }
                        }
                        else
                        {
                            for (int j = 0; j < 20; j++)
                            {
                                var res = rag.Retrieve("OrderService_0", new RagOptions { TopK = 3, MaxGraphHops = 2 });
                                Assert.NotNull(res);
                            }
                        }
                    }
                    catch (ObjectDisposedException)
                    {
                        // Expected once Dispose() completes
                    }
                    catch (Exception ex)
                    {
                        exceptions.Add(ex);
                    }
                });
            }

            startGate.Set();
            await Task.WhenAll(tasks);

            Assert.Empty(exceptions);
            Assert.Throws<ObjectDisposedException>(() => rag.Retrieve("OrderService_0"));
            Assert.Throws<ObjectDisposedException>(() => rag.IndexDocument("AFTER", "AfterNode uses Cache."));
            Assert.Throws<ObjectDisposedException>(() => _ = rag.IndexedChunksCount);
            Assert.Throws<ObjectDisposedException>(() => _ = rag.GraphNodeCount);
            Assert.Throws<ObjectDisposedException>(() => _ = rag.GraphEdgeCount);
        }

        var afterGcsr = Directory.GetFiles(tempDir, "glacier_rag_*.gcsr");
        foreach (string f in afterGcsr)
        {
            Assert.Contains(f, beforeGcsr);
        }
    }

    [Fact]
    public void HybridGraphScorer_InMemoryCsrSpanOverload_MatchesCsrGraphStoreResults()
    {
        string tempFile = Path.Combine(Path.GetTempPath(), $"rag_csr_parity_{Guid.NewGuid():N}.gcsr");
        try
        {
            using var store = new GraphStore();
            store.AddEdge("OrderService", "PaymentGateway", "CALLS", 2.0f);
            store.AddEdge("PaymentGateway", "FraudDetector", "VALIDATES", 1.5f);
            store.AddEdge("OrderService", "AuditLog", "WRITES_TO", 0.5f);
            CsrGraphCompiler.CompileFromForwardStar(store, tempFile);

            var inMem = InMemoryCsrGraph.CompileFromForwardStar(store);
            using var csrStore = new CsrGraphStore(tempFile);

            var vectorMatches = new SearchResult[]
            {
                new(0, 0.90f, "chunk-audit"),
                new(1, 0.85f, "chunk-fraud"),
                new(2, 0.40f, "chunk-payment")
            };

            var queryEntities = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "OrderService" };
            var chunkEntities = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase)
            {
                ["chunk-audit"] = ["AuditLog"],
                ["chunk-fraud"] = ["FraudDetector"],
                ["chunk-payment"] = ["PaymentGateway"]
            };
            var contentMap = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["chunk-audit"] = "AuditLog",
                ["chunk-fraud"] = "FraudDetector",
                ["chunk-payment"] = "PaymentGateway"
            };

            var fromFile = HybridGraphScorer.ScoreAndRerankHippoRag(
                vectorMatches, queryEntities, chunkEntities, contentMap, csrStore);
            var fromMem = HybridGraphScorer.ScoreAndRerankHippoRag(
                vectorMatches,
                queryEntities,
                chunkEntities,
                contentMap,
                inMem.RowOffsets,
                inMem.ColumnIndices,
                inMem.EdgeWeights,
                inMem.NodeCount,
                inMem.ExternalToInternalId);

            Assert.Equal(fromFile.Count, fromMem.Count);
            for (int i = 0; i < fromFile.Count; i++)
            {
                Assert.Equal(fromFile[i].ChunkId, fromMem[i].ChunkId);
                Assert.Equal(fromFile[i].GraphScore, fromMem[i].GraphScore, precision: 5);
                Assert.Equal(fromFile[i].HybridScore, fromMem[i].HybridScore, precision: 5);
            }
        }
        finally
        {
            if (File.Exists(tempFile))
            {
                try { File.Delete(tempFile); }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
            }
        }
    }

    [Fact]
    public void GraphRagEngine_HnswOversamplingTopK4_RescuesGraphStrongCandidateAndCapsVectorMatches()
    {
        var vectors = new Dictionary<string, float[]>(StringComparer.Ordinal)
        {
            ["QUERY_KEY"] = [1.0f, 0.0f, 0.0f, 0.0f],
            ["DENSE_1"] = [0.99f, 0.141f, 0.0f, 0.0f],
            ["DENSE_2"] = [0.95f, 0.312f, 0.0f, 0.0f],
            ["DENSE_3"] = [0.90f, 0.436f, 0.0f, 0.0f],
            // Rank 4 in dense search (outside TopK = 2, inside TopK * 4 = 8), but directly tied to QueryAnchor in graph!
            ["RESCUE_DOC"] = [0.75f, 0.661f, 0.0f, 0.0f],
            ["FAR_DOC"] = [0.10f, 0.995f, 0.0f, 0.0f]
        };

        using var rag = new GraphRagEngine(new DeterministicLookupEmbeddingModel(4, vectors));

        rag.IndexDocument("D1", "DENSE_1 UnrelatedAlpha logs metrics.");
        rag.IndexDocument("D2", "DENSE_2 UnrelatedBeta caches tokens.");
        rag.IndexDocument("D3", "DENSE_3 UnrelatedGamma parses files.");
        rag.IndexDocument("D4", "RESCUE_DOC QueryAnchor connects to TargetVault.");
        rag.IndexDocument("D5", "FAR_DOC DistantNode reads disk.");

        var res = rag.Retrieve("QUERY_KEY QueryAnchor", new RagOptions { TopK = 2, MaxGraphHops = 2 });

        // VectorMatches must be capped back to TopK (2), while RRF oversampled TopK * 4 and rescued D4 into ScoredChunks!
        Assert.True(res.VectorMatches.Count <= 2);
        Assert.NotNull(res.ScoredChunks);
        Assert.True(res.ScoredChunks!.Count <= 2);
        Assert.Contains(res.ScoredChunks, sc => sc.ChunkId == "D4_0");

        // Post-RRF DiscoveredEntities must include rescued D4_0's TargetVault and exclude dropped rank-2/3/5 entities!
        Assert.Contains("TargetVault", res.DiscoveredEntities, StringComparer.OrdinalIgnoreCase);
        Assert.Contains("UnrelatedAlpha", res.DiscoveredEntities, StringComparer.OrdinalIgnoreCase);
        Assert.DoesNotContain("UnrelatedBeta", res.DiscoveredEntities, StringComparer.OrdinalIgnoreCase);
        Assert.DoesNotContain("UnrelatedGamma", res.DiscoveredEntities, StringComparer.OrdinalIgnoreCase);
        Assert.DoesNotContain("DistantNode", res.DiscoveredEntities, StringComparer.OrdinalIgnoreCase);

        // Overflow guard on huge TopK must not throw or overflow
        var largeTopK = rag.Retrieve("QUERY_KEY", new RagOptions { TopK = int.MaxValue / 2, MaxGraphHops = 1 });
        Assert.Equal(5, largeTopK.VectorMatches.Count);

        // TopK <= 0 guard: returns only query entities and empty matches
        var zeroTopK = rag.Retrieve("QUERY_KEY QueryAnchor", new RagOptions { TopK = 0 });
        Assert.Empty(zeroTopK.VectorMatches);
        Assert.Empty(zeroTopK.ScoredChunks!);
        Assert.Contains("QueryAnchor", zeroTopK.DiscoveredEntities, StringComparer.OrdinalIgnoreCase);
        Assert.DoesNotContain("TargetVault", zeroTopK.DiscoveredEntities, StringComparer.OrdinalIgnoreCase);
        Assert.DoesNotContain("UnrelatedAlpha", zeroTopK.DiscoveredEntities, StringComparer.OrdinalIgnoreCase);
        Assert.DoesNotContain("UnrelatedBeta", zeroTopK.DiscoveredEntities, StringComparer.OrdinalIgnoreCase);
    }

    [Fact]
    public void GraphRagEngine_TopK1_MultiDocumentIsolation_DoesNotLeakOversampledEntitiesOrRelations()
    {
        var vectors = new Dictionary<string, float[]>(StringComparer.OrdinalIgnoreCase)
        {
            ["cosmos"] = [1.0f, 0.0f, 0.0f, 0.0f],
            ["Observatory"] = [1.0f, 0.0f, 0.0f, 0.0f],
            ["Pasta"] = [0.2f, 0.9798f, 0.0f, 0.0f]
        };

        using var rag = new GraphRagEngine(new DeterministicLookupEmbeddingModel(4, vectors));

        rag.IndexDocument("astro.txt", "Observatory dome housing wide-field optics for deep galaxy nebula imaging.");
        rag.IndexDocument("kitchen.txt", "Pasta cooking simmering culinary recipe in a warm kitchen.");

        var res = rag.Retrieve("cosmos planet star", new RagOptions { TopK = 1, MaxGraphHops = 2 });

        Assert.Single(res.VectorMatches);
        Assert.NotNull(res.ScoredChunks);
        Assert.Single(res.ScoredChunks!);
        Assert.Equal("astro.txt_0", res.ScoredChunks[0].ChunkId);

        Assert.Contains("Observatory", res.DiscoveredEntities, StringComparer.OrdinalIgnoreCase);
        Assert.DoesNotContain("Pasta", res.DiscoveredEntities, StringComparer.OrdinalIgnoreCase);
        Assert.Empty(res.GraphRelations);
        Assert.Contains("Observatory", res.SynthesizedContext, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("Pasta", res.SynthesizedContext, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void GraphRagEngine_TopK1_DroppedRank2WithGraphTriplet_DoesNotLeakEntitiesOrRelations()
    {
        var vectors = new Dictionary<string, float[]>(StringComparer.OrdinalIgnoreCase)
        {
            ["cosmos"] = [1.0f, 0.0f, 0.0f, 0.0f],
            ["Observatory"] = [1.0f, 0.0f, 0.0f, 0.0f],
            ["Pasta"] = [0.2f, 0.9798f, 0.0f, 0.0f]
        };

        using var rag = new GraphRagEngine(new DeterministicLookupEmbeddingModel(4, vectors));

        rag.IndexDocument("astro.txt", "Observatory dome housing wide-field optics for deep galaxy nebula imaging.");
        rag.IndexDocument("kitchen.txt", "Pasta uses KitchenPot.");

        var res = rag.Retrieve("cosmos planet star", new RagOptions { TopK = 1, MaxGraphHops = 2 });

        Assert.Single(res.VectorMatches);
        Assert.NotNull(res.ScoredChunks);
        Assert.Single(res.ScoredChunks!);
        Assert.Equal("astro.txt_0", res.ScoredChunks[0].ChunkId);

        Assert.Contains("Observatory", res.DiscoveredEntities, StringComparer.OrdinalIgnoreCase);
        Assert.DoesNotContain("Pasta", res.DiscoveredEntities, StringComparer.OrdinalIgnoreCase);
        Assert.DoesNotContain("KitchenPot", res.DiscoveredEntities, StringComparer.OrdinalIgnoreCase);

        Assert.DoesNotContain(res.GraphRelations, r =>
            r.Source.Equals("Pasta", StringComparison.OrdinalIgnoreCase) ||
            r.Target.Equals("KitchenPot", StringComparison.OrdinalIgnoreCase) ||
            r.Relation.Equals("USES", StringComparison.OrdinalIgnoreCase));
        Assert.Empty(res.GraphRelations);

        Assert.DoesNotContain("Pasta", res.SynthesizedContext, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("KitchenPot", res.SynthesizedContext, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("USES", res.SynthesizedContext, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void GraphRagEngine_NonPositiveTopK_WithQueryEntity_DoesNotLeakGraphRelationsOrTargetEntities(int topK)
    {
        using var rag = new GraphRagEngine(new FastHashEmbeddingModel(64));
        rag.IndexDocument("D1", "AlphaService uses BetaCache.");

        var res = rag.Retrieve("AlphaService", new RagOptions { TopK = topK, MaxGraphHops = 2 });

        Assert.Empty(res.VectorMatches);
        Assert.NotNull(res.ScoredChunks);
        Assert.Empty(res.ScoredChunks!);
        Assert.Empty(res.GraphRelations);
        Assert.Single(res.DiscoveredEntities);
        Assert.Contains("AlphaService", res.DiscoveredEntities, StringComparer.OrdinalIgnoreCase);
        Assert.DoesNotContain("BetaCache", res.DiscoveredEntities, StringComparer.OrdinalIgnoreCase);
        Assert.DoesNotContain("BetaCache", res.SynthesizedContext, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void GraphRagEngine_CaseInsensitiveQueryEntity_PreservesPostRrfExactCasedGraphRelations()
    {
        using var rag = new GraphRagEngine(new FastHashEmbeddingModel(64));
        rag.IndexDocument("D1", "AlphaService uses BetaCache.");

        var res = rag.Retrieve("\"alphaservice\"", new RagOptions { TopK = 1, MaxGraphHops = 1 });

        Assert.NotNull(res.ScoredChunks);
        Assert.Single(res.ScoredChunks!);
        Assert.Equal("D1_0", res.ScoredChunks[0].ChunkId);
        Assert.NotEmpty(res.GraphRelations);
        Assert.Contains(res.GraphRelations, r =>
            r.Source == "AlphaService" &&
            r.Relation == "USES" &&
            r.Target == "BetaCache");
        Assert.Contains("BetaCache", res.DiscoveredEntities, StringComparer.OrdinalIgnoreCase);
        Assert.Equal(2, res.DiscoveredEntities.Count);
    }

    [Fact]
    public void GraphRagEngine_MultiEntityQuery_PreRrfDoesNotStarveSelectedTopKChunkMaxTripletsBudget()
    {
        var vectors = new Dictionary<string, float[]>(StringComparer.OrdinalIgnoreCase)
        {
            ["QueryAnchor DecoyHub"] = [1.0f, 0.0f, 0.0f, 0.0f],
            ["DecoyHub"] = [0.99f, 0.141f, 0.0f, 0.0f],
            ["QueryAnchor"] = [0.10f, 0.995f, 0.0f, 0.0f]
        };

        using var rag = new GraphRagEngine(new DeterministicLookupEmbeddingModel(4, vectors));
        rag.IndexDocument("R1", "QueryAnchor AlphaAnchor uses AlphaTarget.");
        rag.IndexDocument("R2", "DecoyHub calls DecoyA. DecoyHub uses DecoyB. DecoyHub loads DecoyC.");

        var res = rag.Retrieve("QueryAnchor DecoyHub", new RagOptions { TopK = 1, MaxGraphHops = 2, MaxTriplets = 3 });

        Assert.NotNull(res.ScoredChunks);
        Assert.Single(res.ScoredChunks!);
        Assert.Equal("R2_0", res.ScoredChunks[0].ChunkId);

        Assert.InRange(res.GraphRelations.Count, 1, 3);
        Assert.Contains(res.GraphRelations, r => r.Source == "DecoyHub");
        Assert.DoesNotContain(res.GraphRelations, r =>
            r.Source.Equals("QueryAnchor", StringComparison.OrdinalIgnoreCase) ||
            r.Source.Equals("AlphaAnchor", StringComparison.OrdinalIgnoreCase) ||
            r.Target.Equals("AlphaTarget", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void GraphRagEngine_CaseInsensitiveQueryEntity_RrfRank4Rescue_RescuesTargetAndExcludesDroppedNeighbors()
    {
        var vectors = new Dictionary<string, float[]>(StringComparer.Ordinal)
        {
            ["QUERY_KEY"] = [1.0f, 0.0f, 0.0f, 0.0f],
            ["DENSE_1"] = [0.99f, 0.141f, 0.0f, 0.0f],
            ["DENSE_2"] = [0.95f, 0.312f, 0.0f, 0.0f],
            ["DENSE_3"] = [0.90f, 0.436f, 0.0f, 0.0f],
            ["RESCUE_DOC"] = [0.75f, 0.661f, 0.0f, 0.0f],
            ["FAR_DOC"] = [0.10f, 0.995f, 0.0f, 0.0f]
        };

        using var rag = new GraphRagEngine(new DeterministicLookupEmbeddingModel(4, vectors));

        rag.IndexDocument("D1", "DENSE_1 UnrelatedAlpha logs metrics.");
        rag.IndexDocument("D2", "DENSE_2 UnrelatedBeta caches tokens.");
        rag.IndexDocument("D3", "DENSE_3 UnrelatedGamma parses files.");
        rag.IndexDocument("D4", "RESCUE_DOC QueryAnchor connects to TargetVault.");
        rag.IndexDocument("D5", "FAR_DOC DistantNode reads disk.");

        var res = rag.Retrieve("QUERY_KEY \"queryanchor\"", new RagOptions { TopK = 2, MaxGraphHops = 2 });

        Assert.True(res.VectorMatches.Count <= 2);
        Assert.NotNull(res.ScoredChunks);
        Assert.True(res.ScoredChunks!.Count <= 2);
        Assert.Contains(res.ScoredChunks, sc => sc.ChunkId == "D4_0" && sc.GraphScore > 0f);

        Assert.Contains("TargetVault", res.DiscoveredEntities, StringComparer.OrdinalIgnoreCase);
        Assert.DoesNotContain("UnrelatedBeta", res.DiscoveredEntities, StringComparer.OrdinalIgnoreCase);
        Assert.DoesNotContain("UnrelatedGamma", res.DiscoveredEntities, StringComparer.OrdinalIgnoreCase);
    }

    [Fact]
    public void GraphRagEngine_CaseInsensitiveQueryEntity_MinSimilarity099_RescuesMultiHopWithGraphScoreParity()
    {
        using var rag = new GraphRagEngine(new FastHashEmbeddingModel(64));
        rag.IndexDocument("D1", "AlphaService connects to BetaCache.");
        rag.IndexDocument("D2", "BetaCache loads GammaQueue.");

        var options = new RagOptions { TopK = 2, MaxGraphHops = 2, MinSimilarity = 0.99f };
        var upper = rag.Retrieve("AlphaService", options);
        var lower = rag.Retrieve("\"alphaservice\"", options);

        Assert.Empty(upper.VectorMatches);
        Assert.Empty(lower.VectorMatches);
        Assert.NotNull(upper.ScoredChunks);
        Assert.NotNull(lower.ScoredChunks);
        Assert.Equal(2, upper.ScoredChunks!.Count);
        Assert.Equal(2, lower.ScoredChunks!.Count);

        var upperD1 = Assert.Single(upper.ScoredChunks, sc => sc.ChunkId == "D1_0");
        var upperD2 = Assert.Single(upper.ScoredChunks, sc => sc.ChunkId == "D2_0");
        var lowerD1 = Assert.Single(lower.ScoredChunks, sc => sc.ChunkId == "D1_0");
        var lowerD2 = Assert.Single(lower.ScoredChunks, sc => sc.ChunkId == "D2_0");

        Assert.True(lowerD1.GraphScore > 0f);
        Assert.True(lowerD2.GraphScore > 0f);
        Assert.Equal(upperD1.GraphScore, lowerD1.GraphScore, precision: 5);
        Assert.Equal(upperD2.GraphScore, lowerD2.GraphScore, precision: 5);
    }

    [Fact]
    public void GraphRagEngine_CaseInsensitiveQueryEntity_PostRrfQueryFallback_IncludesDirectEdgeWhenBudgetRemains()
    {
        var vectors = new Dictionary<string, float[]>(StringComparer.OrdinalIgnoreCase)
        {
            ["\"queryanchor\" DecoyHub"] = [1.0f, 0.0f, 0.0f, 0.0f],
            ["DecoyHub"] = [0.99f, 0.141f, 0.0f, 0.0f],
            ["QueryAnchor"] = [0.10f, 0.995f, 0.0f, 0.0f]
        };

        using var rag = new GraphRagEngine(new DeterministicLookupEmbeddingModel(4, vectors));
        rag.IndexDocument("R1", "QueryAnchor connects to BridgeTarget.");
        rag.IndexDocument("R2", "DecoyHub uses DecoyB.");

        var res = rag.Retrieve("\"queryanchor\" DecoyHub", new RagOptions { TopK = 1, MaxGraphHops = 2, MaxTriplets = 3 });

        Assert.NotNull(res.ScoredChunks);
        Assert.Single(res.ScoredChunks!);
        Assert.Equal("R2_0", res.ScoredChunks[0].ChunkId);
        Assert.Contains(res.GraphRelations, r => r.Source == "DecoyHub" && r.Target == "DecoyB");
        Assert.Contains(res.GraphRelations, r => r.Source == "QueryAnchor" && r.Target == "BridgeTarget");
    }

    [Fact]
    public void GraphRagEngine_PostRrfQueryFallback_BoundedToOneHop_DoesNotLeakDroppedTwoHopChain()
    {
        var vectors = new Dictionary<string, float[]>(StringComparer.OrdinalIgnoreCase)
        {
            ["QueryAnchor DecoyHub"] = [1.0f, 0.0f, 0.0f, 0.0f],
            ["DecoyHub"] = [0.99f, 0.141f, 0.0f, 0.0f],
            ["IntermediateNode"] = [0.85f, 0.527f, 0.0f, 0.0f],
            ["QueryAnchor"] = [0.10f, 0.995f, 0.0f, 0.0f]
        };

        using var rag = new GraphRagEngine(new DeterministicLookupEmbeddingModel(4, vectors));
        rag.IndexDocument("R1", "QueryAnchor AlphaAnchor uses AlphaTarget.");
        rag.IndexDocument("R2", "DecoyHub uses DecoyB.");
        rag.IndexDocument("R3", "IntermediateNode logs events.");

        var res = rag.Retrieve("QueryAnchor DecoyHub", new RagOptions { TopK = 1, MaxGraphHops = 2, MaxTriplets = 5 });

        Assert.NotNull(res.ScoredChunks);
        Assert.Single(res.ScoredChunks!);
        Assert.Equal("R2_0", res.ScoredChunks[0].ChunkId);
        Assert.Contains(res.GraphRelations, r => r.Source == "DecoyHub" && r.Target == "DecoyB");
        Assert.DoesNotContain(res.GraphRelations, r =>
            r.Source.Equals("AlphaAnchor", StringComparison.OrdinalIgnoreCase) ||
            r.Target.Equals("AlphaTarget", StringComparison.OrdinalIgnoreCase));
        Assert.DoesNotContain("AlphaTarget", res.SynthesizedContext, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("AlphaTarget", res.DiscoveredEntities, StringComparer.OrdinalIgnoreCase);
    }

    [Fact]
    public void HybridGraphScorer_DirectCallWithCaseVariantQueryEntity_ResolvesGraphScoresAcrossAllOverloads()
    {
        using var store = new GraphStore();
        store.AddEdge("OrderService", "PaymentGateway", "CALLS", 2.0f);
        store.AddEdge("PaymentGateway", "FraudDetector", "VALIDATES", 1.5f);
        var search = new Glacier.Graph.Traversal.GraphSearch(store);
        var inMem = InMemoryCsrGraph.CompileFromForwardStar(store);

        var vectorMatches = new SearchResult[]
        {
            new(0, 0.80f, "chunk-fraud"),
            new(1, 0.40f, "chunk-payment")
        };

        var upperQuery = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "OrderService" };
        var lowerQuery = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "orderservice" };
        var chunkEntities = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase)
        {
            ["chunk-fraud"] = ["FraudDetector"],
            ["chunk-payment"] = ["PaymentGateway"]
        };
        var contentMap = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["chunk-fraud"] = "FraudDetector",
            ["chunk-payment"] = "PaymentGateway"
        };

        var csrUpper = HybridGraphScorer.ScoreAndRerankHippoRag(vectorMatches, upperQuery, chunkEntities, contentMap, inMem);
        var csrLower = HybridGraphScorer.ScoreAndRerankHippoRag(vectorMatches, lowerQuery, chunkEntities, contentMap, inMem);
        var spanLower = HybridGraphScorer.ScoreAndRerankHippoRag(
            vectorMatches, lowerQuery, chunkEntities, contentMap,
            inMem.RowOffsets, inMem.ColumnIndices, inMem.EdgeWeights, inMem.NodeCount, inMem.ExternalToInternalId);
        var bfsUpper = HybridGraphScorer.ScoreAndRerank(vectorMatches, upperQuery, chunkEntities, contentMap, store, search);
        var bfsLower = HybridGraphScorer.ScoreAndRerank(vectorMatches, lowerQuery, chunkEntities, contentMap, store, search);

        Assert.Equal(csrUpper.Count, csrLower.Count);
        Assert.Equal(csrUpper.Count, spanLower.Count);
        for (int i = 0; i < csrUpper.Count; i++)
        {
            Assert.True(csrLower[i].GraphScore > 0f);
            Assert.Equal(csrUpper[i].ChunkId, csrLower[i].ChunkId);
            Assert.Equal(csrUpper[i].GraphScore, csrLower[i].GraphScore, precision: 5);
            Assert.Equal(csrUpper[i].ChunkId, spanLower[i].ChunkId);
            Assert.Equal(csrUpper[i].GraphScore, spanLower[i].GraphScore, precision: 5);
        }

        Assert.Equal(bfsUpper.Count, bfsLower.Count);
        for (int i = 0; i < bfsUpper.Count; i++)
        {
            Assert.True(bfsLower[i].GraphScore > 0f);
            Assert.Equal(bfsUpper[i].ChunkId, bfsLower[i].ChunkId);
            Assert.Equal(bfsUpper[i].GraphScore, bfsLower[i].GraphScore, precision: 5);
        }
    }

    [Fact]
    public void BuildAugmentedPrompt_EscapesAllControlTokensInContextAndQuery_AndIsIdempotent()
    {
        string[] adversarialTokens =
        [
            "<|im_start|>", "<|im_end|>", "<|im_sep|>", "<|endoftext|>",
            "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|begin_of_text|>", "<|end_of_text|>",
            "<start_of_turn>", "<end_of_turn>", "<|turn>", "<turn|>",
            "<|channel>", "<channel|>", "<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>",
            "<bos>", "<eos>", "<think>", "</think>", "<|think|>", "<|thought|>", "<|/thought|>",
            "<｜System｜>", "<｜User｜>", "<｜Assistant｜>", "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>",
            "<｜ref｜>", "<｜/grounding｜>", "｜DSML｜",
            "<|message_user|>", "<|message_model|>", "<|message_system|>", "<|message_tool|>",
            "<|content_text|>", "<|end_message|>", "<|content_model_end_sampling|>", "<|unused_200054|>",
            "[INST]", "[/INST]", "<s>", "</s>", "<unk>", "[TOOL_CALLS]", "[TOOL_RESULTS]", "[/TOOL_RESULTS]"
        ];

        string injectedPayload = string.Join(" ", adversarialTokens);

        using var rag = new GraphRagEngine(new FastHashEmbeddingModel(128));
        rag.IndexDocument("INJ-DOC", $"PoisonService uses TargetStore. Payload: {injectedPayload}");

        var retrieval = rag.Retrieve($"PoisonService {injectedPayload}", new RagOptions { TopK = 1 });
        string prompt = rag.BuildAugmentedPrompt($"Explain PoisonService {injectedPayload}", retrieval);

        // Only the 3 real ChatML wrapper delimiters may remain in the outer prompt.
        Assert.Equal(3, CountOccurrences(prompt, "<|im_start|>"));
        Assert.Equal(2, CountOccurrences(prompt, "<|im_end|>"));

        foreach (string token in adversarialTokens)
        {
            if (token is "<|im_start|>" or "<|im_end|>")
                continue;

            Assert.DoesNotContain(token, prompt, StringComparison.Ordinal);
        }

        // Idempotency + zero-alloc fast path check on clean text
        string once = PromptSanitizer.Sanitize(injectedPayload);
        string twice = PromptSanitizer.Sanitize(once);
        Assert.Equal(once, twice);

        string clean = "Plain text without any delimiter tokens.";
        Assert.Same(clean, PromptSanitizer.Sanitize(clean));
    }

    [Fact]
    public async Task OpenAiEmbeddingModel_BatchGeneration_AndIndexDocumentAsync_BatchCorrectly()
    {
        using var handler = new BatchStubHandler((body, _, _) =>
        {
            using var doc = JsonDocument.Parse(body);
            var inputEl = doc.RootElement.GetProperty("input");
            if (inputEl.ValueKind == JsonValueKind.Array)
            {
                int count = inputEl.GetArrayLength();
                // Return embeddings in reverse index order to verify Index mapping!
                var items = new List<string>(count);
                for (int i = count - 1; i >= 0; i--)
                {
                    float x = i + 1;
                    items.Add($"{{\"index\":{i},\"embedding\":[{x},0.0,0.0,0.0]}}");
                }
                string json = $"{{\"object\":\"list\",\"data\":[{string.Join(',', items)}]}}";
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent(json, Encoding.UTF8, "application/json")
                });
            }

            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent("{\"object\":\"list\",\"data\":[{\"index\":0,\"embedding\":[0.0,3.0,4.0,0.0]}]}", Encoding.UTF8, "application/json")
            });
        });

        using var http = new HttpClient(handler);
        using var model = new OpenAiEmbeddingModel("http://127.0.0.1:52625", "embeddinggemma", dimensions: 4, httpClient: http);

        // 1. Multi-batch partition: 3 inputs with maxBatchSize = 2 -> 1 array request (2 items) + 1 single request (1 item)
        float[][] batch = await model.GenerateEmbeddingsAsync(["alpha", "beta", "gamma"], maxBatchSize: 2);
        Assert.Equal(3, batch.Length);
        Assert.Equal(2, handler.RequestBodies.Count);
        Assert.Equal(1.0f, batch[0][0], precision: 4);
        Assert.Equal(1.0f, batch[1][0], precision: 4);
        Assert.Equal(0.6f, batch[2][1], precision: 4);
        Assert.Equal(0.8f, batch[2][2], precision: 4);

        handler.RequestBodies.Clear();

        // 2. GraphRagEngine.IndexDocumentAsync batches multi-chunk doc into 1 HTTP call
        using var rag = new GraphRagEngine(model);
        string multiChunk = string.Join("\n\n", Enumerable.Range(0, 12).Select(i => $"Component_{i} depends on Storage_{i} in microservice architecture {new string('x', 80)}."));
        int chunkCount = await rag.IndexDocumentAsync("BATCH-DOC", multiChunk);
        Assert.True(chunkCount > 1);
        Assert.Single(handler.RequestBodies);

        // 3. Error guards on malformed batch responses (duplicate index / count mismatch / non-finite)
        using var dupHandler = new BatchStubHandler((_, _, _) =>
            Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent("""{"data":[{"index":0,"embedding":[1,0,0,0]},{"index":0,"embedding":[0,1,0,0]}]}""", Encoding.UTF8, "application/json")
            }));
        using var dupHttp = new HttpClient(dupHandler);
        using var dupModel = new OpenAiEmbeddingModel("http://127.0.0.1:52625", "m", dimensions: 4, httpClient: dupHttp);
        await Assert.ThrowsAsync<InvalidOperationException>(() => dupModel.GenerateEmbeddingsAsync(["a", "b"]));

        using var badCountHandler = new BatchStubHandler((_, _, _) =>
            Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent("""{"data":[{"index":0,"embedding":[1,0,0,0]}]}""", Encoding.UTF8, "application/json")
            }));
        using var badCountHttp = new HttpClient(badCountHandler);
        using var badCountModel = new OpenAiEmbeddingModel("http://127.0.0.1:52625", "m", dimensions: 4, httpClient: badCountHttp);
        await Assert.ThrowsAsync<InvalidOperationException>(() => badCountModel.GenerateEmbeddingsAsync(["a", "b"]));
    }

    [Fact]
    public void LinearProjectionEmbeddingModel_HighDimSimdAndZeroSteadyStateAllocations()
    {
        using var upstream = new FastHashEmbeddingModel(1536);
        Assert.Throws<ArgumentOutOfRangeException>(() => new LinearProjectionEmbeddingModel(upstream, outputDim: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LinearProjectionEmbeddingModel(upstream, outputDim: -4));

        using var proj = new LinearProjectionEmbeddingModel(upstream, outputDim: 387, seed: 99);
        float[] output = new float[387];

        // Warmup JIT & ArrayPool
        proj.GenerateEmbedding("High dimensional enterprise graph embedding warmup", output);

        long allocBefore = GC.GetAllocatedBytesForCurrentThread();
        proj.GenerateEmbedding("High dimensional enterprise graph embedding steady state", output);
        long allocAfter = GC.GetAllocatedBytesForCurrentThread();

        Assert.Equal(0L, allocAfter - allocBefore);

        float sumSq = 0f;
        for (int i = 0; i < output.Length; i++)
        {
            Assert.True(float.IsFinite(output[i]));
            sumSq += output[i] * output[i];
        }
        Assert.InRange(sumSq, 0.999f, 1.001f);

        // Zero-input stability check
        proj.GenerateEmbedding(ReadOnlySpan<char>.Empty, output);
        Assert.All(output, v => Assert.True(float.IsFinite(v)));
    }

    [Fact]
    public void DocumentChunker_TrimmedOffsetsMatchSourceSliceExactly()
    {
        string text = "   \r\n  AlphaService connects to BetaDb.\r\n\r\n   GammaWorker validates DeltaQueue.   \n\n  EpsilonNode stores ZetaBlob.  ";
        var chunks = DocumentChunker.ChunkText("offsets", text, maxChunkLength: 55, overlap: 15);

        Assert.NotEmpty(chunks);
        foreach (var chunk in chunks)
        {
            Assert.Equal(chunk.Content, text[chunk.StartChar..chunk.EndChar]);
            Assert.False(char.IsWhiteSpace(chunk.Content[0]));
            Assert.False(char.IsWhiteSpace(chunk.Content[^1]));
        }
    }

    [Fact]
    public void EntityExtractor_SinglePassSentenceProximity_PreservesMultiTriggerAndTieBreaking()
    {
        string text =
            "\"Order Gateway\" depends on SQL; AuthService validates JWT and routes to SecurityStore.\n" +
            "TelemetryWorker calls MetricsSink. BackgroundWorker uses \"Cache Tier\".";

        var (entities, triplets) = EntityExtractor.Extract(text);

        Assert.Contains("Order Gateway", entities);
        Assert.Contains("SQL", entities);
        Assert.Contains("AuthService", entities);
        Assert.Contains("JWT", entities);
        Assert.Contains("SecurityStore", entities);
        Assert.Contains("TelemetryWorker", entities);
        Assert.Contains("MetricsSink", entities);
        Assert.Contains("BackgroundWorker", entities);
        Assert.Contains("Cache Tier", entities);

        Assert.Contains(triplets, t => t.Subject == "Order Gateway" && t.Predicate == "DEPENDS_ON" && t.Object == "SQL");
        Assert.Contains(triplets, t => t.Subject == "AuthService" && t.Predicate == "VALIDATES" && t.Object == "JWT");
        Assert.Contains(triplets, t => t.Subject == "JWT" && t.Predicate == "ROUTES_TO" && t.Object == "SecurityStore");
        Assert.Contains(triplets, t => t.Subject == "TelemetryWorker" && t.Predicate == "CALLS" && t.Object == "MetricsSink");
        Assert.Contains(triplets, t => t.Subject == "BackgroundWorker" && t.Predicate == "USES" && t.Object == "Cache Tier");
    }

    private static int CountOccurrences(string haystack, string needle)
    {
        int count = 0;
        int idx = 0;
        while ((idx = haystack.IndexOf(needle, idx, StringComparison.Ordinal)) >= 0)
        {
            count++;
            idx += needle.Length;
        }
        return count;
    }
}
