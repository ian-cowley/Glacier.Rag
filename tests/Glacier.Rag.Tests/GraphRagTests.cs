namespace Glacier.Rag.Tests;

using System;
using System.IO;
using Glacier.Rag.Chunking;
using Glacier.Rag.Embeddings;
using Glacier.Rag.Engine;
using Glacier.Rag.Extraction;
using Xunit;

public class GraphRagTests
{
    [Fact]
    public void DocumentChunker_ChunksWithOverlap_Correctly()
    {
        string text = "Paragraph 1: Introduction to Fintech architecture.\n\nParagraph 2: Customer transactions stream via ADO.NET.\n\nParagraph 3: LedgerAccount balances are validated.";
        var chunks = DocumentChunker.ChunkText("doc1", text, maxChunkLength: 60, overlap: 10);

        Assert.True(chunks.Count >= 2);
        Assert.Equal("doc1", chunks[0].DocumentId);
        Assert.False(string.IsNullOrWhiteSpace(chunks[0].Content));
    }

    [Fact]
    public void FastHashEmbedding_ProducesUnitVectors()
    {
        using var model = new FastHashEmbeddingModel(128);
        Span<float> emb = stackalloc float[128];
        model.GenerateEmbedding("LedgerAccountDto validation service", emb);

        float sumSq = 0f;
        for (int i = 0; i < 128; i++) sumSq += emb[i] * emb[i];

        Assert.True(MathF.Abs(sumSq - 1.0f) < 1e-4f, "L2 norm must be 1.0");
    }

    [Fact]
    public void EntityExtractor_FindsEntitiesAndTriplets()
    {
        string text = "CustomerRecord streams from SqlServer. LedgerAccountDto validates through FluentValidation.";
        var (entities, triplets) = EntityExtractor.Extract(text);

        Assert.Contains("CustomerRecord", entities);
        Assert.Contains("SqlServer", entities);
        Assert.Contains("LedgerAccountDto", entities);
        Assert.Contains("FluentValidation", entities);
        Assert.True(triplets.Count > 0);
    }

    [Fact]
    public void GraphRagEngine_IndexesAndRetrieves_Sub15Ms()
    {
        using var rag = new GraphRagEngine(new FastHashEmbeddingModel(128));

        string doc = @"
Fintech Core Architecture Specification.
The LedgerService manages CustomerAccount transactions.
CustomerAccount streams from SqlServerDatabase using high performance ADO.NET pipelines.
All balance updates route to TransactionJournal before committing to the database.
LedgerAccountDto validates through FluentValidation rules.";

        int chunkCount = rag.IndexDocument("SPEC-001", doc);
        Assert.True(chunkCount > 0);
        Assert.True(rag.GraphNodeCount >= 4);
        Assert.True(rag.GraphEdgeCount >= 2);

        // Warmup JIT compilation and static hardware probes
        _ = rag.Retrieve("warmup", new RagOptions { TopK = 1, MaxGraphHops = 1 });

        var result = rag.Retrieve("How does CustomerAccount stream data?", new RagOptions { TopK = 2, MaxGraphHops = 2 });

        Assert.NotNull(result);
        Assert.True(result.VectorMatches.Count > 0);
        Assert.True(result.TotalRetrievalLatencyMs < 15.0, $"Retrieval must be < 15ms, took {result.TotalRetrievalLatencyMs:F2}ms");
        Assert.Contains("CustomerAccount", result.SynthesizedContext);
    }

    [Fact]
    public void GraphRagEngine_ScalesWithO1Retrieval_AcrossManyChunks()
    {
        using var rag = new GraphRagEngine(new FastHashEmbeddingModel(128));

        // Index 2,000 synthetic chunks
        for (int i = 0; i < 2000; i++)
        {
            rag.IndexDocument($"DOC_{i}", $"AccountEntity_{i} connects to DatabaseService_{i % 50} with TransactionLog_{i}.");
        }

        // Warmup JIT
        _ = rag.Retrieve("warmup", new RagOptions { TopK = 1, MaxGraphHops = 1 });

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var result = rag.Retrieve("AccountEntity_1000", new RagOptions { TopK = 5, MaxGraphHops = 1 });
        sw.Stop();

        Assert.NotNull(result);
        Assert.True(result.TotalRetrievalLatencyMs < 10.0, $"O(1) retrieval should take < 10ms for 2,000 chunks, took {result.TotalRetrievalLatencyMs:F3}ms (vec={result.VectorSearchLatencyMs:F3}ms, graph={result.GraphTraversalLatencyMs:F3}ms, totalElapsed={sw.Elapsed.TotalMilliseconds:F3}ms)");
    }

    [Fact]
    public void EntityExtractor_FindsAcronymsAndQuotedEntities_AndExpandedRelations()
    {
        string text = @"""Payment Gateway"" connects to SQL. The HTTP API persists to DatabaseStore.";
        var (entities, triplets) = EntityExtractor.Extract(text);

        Assert.Contains("Payment Gateway", entities);
        Assert.Contains("SQL", entities);
        Assert.Contains("HTTP", entities);
        Assert.Contains("API", entities);
        Assert.Contains("DatabaseStore", entities);

        Assert.Contains(triplets, t => t.Predicate == "CONNECTS_TO");
        Assert.Contains(triplets, t => t.Predicate == "PERSISTS_TO");
    }

    [Fact]
    public void GraphRagEngine_PreservesTruePredicates_InSynthesizedContext()
    {
        using var rag = new GraphRagEngine(new FastHashEmbeddingModel(128));

        string doc = "CustomerAccount streams from SqlServerDatabase. LedgerAccountDto validates through FluentValidation.";
        rag.IndexDocument("SPEC-REL", doc);

        var result = rag.Retrieve("CustomerAccount");

        Assert.NotNull(result);
        Assert.NotEmpty(result.GraphRelations);
        Assert.Contains(result.GraphRelations, r => r.Relation == "STREAMS_FROM" || r.Relation == "VALIDATES");
        Assert.Contains("STREAMS_FROM", result.SynthesizedContext);
    }

    [Fact]
    public void GraphRagEngine_PrunesHubNodes_PreventsContextExplosion()
    {
        using var rag = new GraphRagEngine(new FastHashEmbeddingModel(128));

        // Create a central hub node connected to 40 distinct services
        for (int i = 0; i < 40; i++)
        {
            rag.IndexDocument($"HUB_{i}", $"CentralHubService manages MicroserviceNode_{i}.");
        }

        var result = rag.Retrieve("CentralHubService", new RagOptions { MaxDegreePerNode = 5, MaxTriplets = 10 });

        Assert.NotNull(result);
        Assert.True(result.GraphRelations.Count <= 10, $"Should not exceed MaxTriplets=10, got {result.GraphRelations.Count}");
    }
}
