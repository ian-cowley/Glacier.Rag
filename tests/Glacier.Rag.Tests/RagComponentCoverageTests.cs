namespace Glacier.Rag.Tests;

using System;
using System.Numerics;
using Glacier.Rag.Chunking;
using Glacier.Rag.Embeddings;
using Glacier.Rag.Extraction;
using Xunit;

public class RagComponentCoverageTests
{
    [Theory]
    [InlineData(0, 0)]
    [InlineData(-10, 0)]
    [InlineData(64, -1)]
    [InlineData(64, 64)]
    [InlineData(64, 100)]
    public void DocumentChunker_InvalidArguments_ThrowArgumentOutOfRangeException(int maxLen, int overlap)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => DocumentChunker.ChunkText("doc1", "some text", maxLen, overlap));
    }

    [Fact]
    public void DocumentChunker_EmptyAndMultiParagraph_ChunksCleanly()
    {
        Assert.Empty(DocumentChunker.ChunkText("doc", string.Empty));

        string text = "First paragraph sentence one. First paragraph sentence two.\n\nSecond paragraph content here is also fairly long.";
        var chunks = DocumentChunker.ChunkText("docA", text, maxChunkLength: 70, overlap: 10);

        Assert.True(chunks.Count >= 2);
        Assert.Equal("docA_0", chunks[0].ChunkId);
        Assert.StartsWith("[docA#0]", chunks[0].ToString());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-8)]
    [InlineData(12)]
    public void FastHashEmbeddingModel_InvalidDimensions_ThrowsArgumentException(int dims)
    {
        Assert.Throws<ArgumentException>(() => new FastHashEmbeddingModel(dims));
    }

    [Fact]
    public void FastHashEmbeddingModel_EmptyAndNormalizedOutput_BehavesDeterministically()
    {
        var model = new FastHashEmbeddingModel(64);
        Assert.Throws<ArgumentException>(() => model.GenerateEmbedding("hello", new float[32]));

        var emptyVec = new float[64];
        model.GenerateEmbedding(ReadOnlySpan<char>.Empty, emptyVec);
        Assert.All(emptyVec, v => Assert.Equal(0f, v));

        var v1 = new float[64];
        var v2 = new float[64];
        model.GenerateEmbedding("Glacier SIMD vector hashing", v1);
        model.GenerateEmbedding("Glacier SIMD vector hashing", v2);

        Assert.Equal(v1, v2);
        float normSq = 0f;
        for (int i = 0; i < v1.Length; i++) normSq += v1[i] * v1[i];
        Assert.InRange(MathF.Sqrt(normSq), 0.999f, 1.001f);
    }

    [Fact]
    public void LinearProjectionEmbeddingModel_GuardsAndProjectionNormalization_Work()
    {
        var upstream = new FastHashEmbeddingModel(64);

        Assert.Throws<ArgumentNullException>(() => new LinearProjectionEmbeddingModel(null!, 16));
        Assert.Throws<ArgumentNullException>(() => new LinearProjectionEmbeddingModel(upstream, 16, null!));
        Assert.Throws<ArgumentException>(() => new LinearProjectionEmbeddingModel(upstream, 16, new float[10]));
        Assert.Throws<ArgumentException>(() => new LinearProjectionEmbeddingModel(upstream, 16, new float[16 * 64], new float[8]));

        var projected = new LinearProjectionEmbeddingModel(upstream, 16, seed: 42);
        Assert.Equal(16, projected.Dimensions);
        Assert.Equal(64, projected.InputDimensions);
        Assert.Same(upstream, projected.UpstreamModel);

        Assert.Throws<ArgumentException>(() => projected.GenerateEmbedding("test", new float[8]));

        var output = new float[16];
        projected.GenerateEmbedding("GraphRAG projection test", output);
        float normSq = 0f;
        for (int i = 0; i < output.Length; i++) normSq += output[i] * output[i];
        Assert.InRange(MathF.Sqrt(normSq), 0.999f, 1.001f);
    }

    [Fact]
    public void EntityExtractor_ExtractsQuotedPascalCaseAcronymsAndTriplets()
    {
        var (emptyEntities, emptyTriplets) = EntityExtractor.Extract("   ");
        Assert.Empty(emptyEntities);
        Assert.Empty(emptyTriplets);

        string text = "OrderService depends on \"SQLite Store\" and publishes to KAFKA.";
        var (entities, triplets) = EntityExtractor.Extract(text);

        Assert.Contains("OrderService", entities);
        Assert.Contains("SQLite Store", entities);
        Assert.Contains("KAFKA", entities);
        Assert.NotEmpty(triplets);
    }
}
