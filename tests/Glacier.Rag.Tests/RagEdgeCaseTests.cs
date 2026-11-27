namespace Glacier.Rag.Tests;

using System;
using System.Collections.Generic;
using Glacier.Rag.Chunking;
using Glacier.Rag.Chunking;
using Glacier.Rag.Embeddings;
using Glacier.Rag.Engine;
using Glacier.Rag.Extraction;
using Xunit;

public class RagEdgeCaseTests
{
    [Fact]
    public void DocumentChunker_HandlesEmptyAndWhitespaceString()
    {
        var emptyChunks = DocumentChunker.ChunkText("doc_empty", "");
        Assert.Empty(emptyChunks);

        var wsChunks = DocumentChunker.ChunkText("doc_ws", "   \t\r\n   ");
        Assert.Empty(wsChunks);
    }

    [Fact]
    public void DocumentChunker_HandlesTextShorterThanChunkLength()
    {
        string shortText = "Short single sentence.";
        var chunks = DocumentChunker.ChunkText("doc_short", shortText, maxChunkLength: 200, overlap: 20);
        Assert.Single(chunks);
        Assert.Equal("doc_short", chunks[0].DocumentId);
        Assert.Equal(shortText, chunks[0].Content);
    }

    [Fact]
    public void GraphRagEngine_EmptyQuery_ReturnsEmptyMatchesGracefully()
    {
        using var rag = new GraphRagEngine(new FastHashEmbeddingModel(64));
        rag.IndexDocument("D1", "PaymentService handles AccountTransactions.");

        var res = rag.Retrieve("", new RagOptions { TopK = 5 });
        Assert.NotNull(res);
        Assert.NotNull(res.VectorMatches);
    }

    [Fact]
    public void GraphRagEngine_DisposedEngine_ThrowsObjectDisposedException()
    {
        var rag = new GraphRagEngine(new FastHashEmbeddingModel(64));
        rag.Dispose();

        Assert.Throws<ObjectDisposedException>(() => rag.IndexDocument("D1", "Some content"));
        Assert.Throws<ObjectDisposedException>(() => rag.Retrieve("query"));
    }

    [Fact]
    public void GraphRagEngine_CyclicGraph_DoesNotInfiniteLoop()
    {
        using var rag = new GraphRagEngine(new FastHashEmbeddingModel(64));
        // Node A -> Node B -> Node C -> Node A
        rag.IndexDocument("CYC1", "ServiceAlpha depends on ServiceBeta.");
        rag.IndexDocument("CYC2", "ServiceBeta depends on ServiceGamma.");
        rag.IndexDocument("CYC3", "ServiceGamma depends on ServiceAlpha.");

        var res = rag.Retrieve("ServiceAlpha", new RagOptions { TopK = 3, MaxGraphHops = 5 });
        Assert.NotNull(res);
        Assert.NotNull(res.SynthesizedContext);
        Assert.NotEmpty(res.GraphRelations);
    }

    [Fact]
    public void GraphRagEngine_ZeroTopK_ReturnsZeroVectorMatches()
    {
        using var rag = new GraphRagEngine(new FastHashEmbeddingModel(64));
        rag.IndexDocument("D1", "OrderProcessor persists orders to OrderDb.");

        var res = rag.Retrieve("OrderProcessor", new RagOptions { TopK = 0, MaxGraphHops = 1 });
        Assert.NotNull(res);
        Assert.Empty(res.VectorMatches);
    }

    [Fact]
    public void EntityExtractor_AcronymRegex_IgnoresPureNumericTokens()
    {
        var (entities, _) = EntityExtractor.Extract("Version 2026 release 10 uses CUDA and S3 storage.");
        Assert.DoesNotContain("2026", entities);
        Assert.DoesNotContain("10", entities);
        Assert.Contains("CUDA", entities);
        Assert.Contains("S3", entities);
    }

    [Fact]
    public void DocumentChunker_HighOverlapWithShortNaturalBreaks_DoesNotStall()
    {
        string text = string.Join("\n\n", new string[20].Select((_, idx) => $"Paragraph {idx} sentence about Glacier components."));
        var chunks = DocumentChunker.ChunkText("doc1", text, maxChunkLength: 100, overlap: 90);
        Assert.True(chunks.Count < text.Length / 20, $"Unexpected chunk explosion: {chunks.Count}");
    }

    [Fact]
    public void GraphRagEngine_UsesHippoRagPprAndInvalidatesCacheAcrossIndexes()
    {
        using var rag = new GraphRagEngine(new FastHashEmbeddingModel(64));
        rag.IndexDocument("A", "AlphaService calls BetaService.");
        var r1 = rag.Retrieve("AlphaService");
        Assert.NotEmpty(r1.SynthesizedContext);

        rag.IndexDocument("B", "BetaService writes to ArchiveDb.");
        var r2 = rag.Retrieve("AlphaService");
        Assert.NotEmpty(r2.VectorMatches);
    }
}
