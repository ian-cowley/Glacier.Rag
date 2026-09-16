namespace Glacier.Rag.Embeddings;

using System;

/// <summary>
/// High-performance embedding generator interface.
/// Generates normalized floating-point vectors for text segments with zero intermediate heap allocations.
/// </summary>
public interface IEmbeddingModel : IDisposable
{
    /// <summary>
    /// Dimensionality of the produced embeddings (e.g. 128, 384, 768, 1536, 3584).
    /// </summary>
    int Dimensions { get; }

    /// <summary>
    /// Computes normalized embedding vector for the specified text.
    /// </summary>
    void GenerateEmbedding(ReadOnlySpan<char> text, Span<float> destination);
}
