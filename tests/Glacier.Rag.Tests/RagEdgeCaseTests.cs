namespace Glacier.Rag.Tests;

using System;
using System.Collections.Generic;
using Glacier.Rag.Chunking;
using Glacier.Rag.Embeddings;
using Glacier.Rag.Engine;
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
        Assert.True(res.TotalRetrievalLatencyMs < 20.0);
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
}
