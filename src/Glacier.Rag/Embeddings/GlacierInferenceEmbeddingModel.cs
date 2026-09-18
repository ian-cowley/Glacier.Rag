namespace Glacier.Rag.Embeddings;

using System;
using Glacier.Inference.Engine;

/// <summary>
/// High-throughput, semantic embedding model powered natively by Glacier.Inference.
/// Runs decoder or encoder GGUF models directly in-process via hardware SIMD or GPU with zero IPC overhead.
/// </summary>
public sealed class GlacierInferenceEmbeddingModel : IEmbeddingModel
{
    private readonly InferenceSession _session;
    private readonly PoolingStrategy _poolingStrategy;
    private readonly bool _ownsSession;
    private bool _disposed;

    public int Dimensions => _session.Weights.EmbeddingLength;
    public InferenceSession Session => _session;

    public GlacierInferenceEmbeddingModel(
        string ggufModelPath,
        string? device = "auto",
        PoolingStrategy poolingStrategy = PoolingStrategy.LastToken)
    {
        _session = new InferenceSession(ggufModelPath, maxSeqLen: 512, device: device);
        _poolingStrategy = poolingStrategy;
        _ownsSession = true;
    }

    public GlacierInferenceEmbeddingModel(
        InferenceSession session,
        PoolingStrategy poolingStrategy = PoolingStrategy.LastToken)
    {
        _session = session ?? throw new ArgumentNullException(nameof(session));
        _poolingStrategy = poolingStrategy;
        _ownsSession = false;
    }

    public void GenerateEmbedding(ReadOnlySpan<char> text, Span<float> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _session.ExtractEmbedding(text, destination, _poolingStrategy);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            if (_ownsSession)
            {
                _session.Dispose();
            }
        }
    }
}
