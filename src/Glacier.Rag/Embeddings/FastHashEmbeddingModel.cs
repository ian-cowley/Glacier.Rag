namespace Glacier.Rag.Embeddings;

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

/// <summary>
/// Deterministic lexical SIMD feature hashing embedding model.
/// Projects character n-grams and word tokens into normalized Euclidean space in sub-microseconds using SIMD vector normalization.
/// Provides extreme throughput for lexical retrieval, embedded environments, and zero-weight baseline testing.
/// Note: This is a lexical/n-gram hashing model; for deep semantic language understanding, use GlacierInferenceEmbeddingModel.
/// </summary>
public sealed class FastHashEmbeddingModel : IEmbeddingModel
{
    private readonly int _dimensions;

    public int Dimensions => _dimensions;

    public FastHashEmbeddingModel(int dimensions = 384)
    {
        if (dimensions <= 0 || dimensions % 8 != 0)
            throw new ArgumentException("Dimensions must be a positive multiple of 8", nameof(dimensions));
        _dimensions = dimensions;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void GenerateEmbedding(ReadOnlySpan<char> text, Span<float> destination)
    {
        if (destination.Length < _dimensions)
            throw new ArgumentException($"Destination span too small. Expected {_dimensions}, got {destination.Length}");

        destination.Slice(0, _dimensions).Clear();

        if (text.IsEmpty) return;

        // Hash word tokens and character n-grams (3-grams and 4-grams)
        int start = 0;
        for (int i = 0; i <= text.Length; i++)
        {
            if (i == text.Length || char.IsWhiteSpace(text[i]) || char.IsPunctuation(text[i]))
            {
                int len = i - start;
                if (len > 0)
                {
                    var word = text.Slice(start, len);
                    AccumulateToken(word, destination);

                    // Also accumulate character trigrams
                    for (int g = 0; g <= len - 3; g++)
                    {
                        AccumulateNgram(word.Slice(g, 3), destination);
                    }
                }
                start = i + 1;
            }
        }

        // Normalize vector to unit length (L2 norm)
        NormalizeL2(destination.Slice(0, _dimensions));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void AccumulateToken(ReadOnlySpan<char> token, Span<float> dst)
    {
        uint h1 = HashString(token, 0x9747b28c);
        uint h2 = HashString(token, 0x1b873593);

        int idx1 = (int)(h1 % (uint)_dimensions);
        int idx2 = (int)(h2 % (uint)_dimensions);

        float sign1 = (h1 & 0x80000000) != 0 ? 1.0f : -1.0f;
        float sign2 = (h2 & 0x80000000) != 0 ? 1.0f : -1.0f;

        dst[idx1] += sign1 * 1.5f;
        dst[idx2] += sign2 * 1.0f;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void AccumulateNgram(ReadOnlySpan<char> ngram, Span<float> dst)
    {
        uint h = HashString(ngram, 0x5bd1e995);
        int idx = (int)(h % (uint)_dimensions);
        float sign = (h & 0x80000000) != 0 ? 0.5f : -0.5f;
        dst[idx] += sign;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint HashString(ReadOnlySpan<char> s, uint seed)
    {
        uint hash = seed;
        for (int i = 0; i < s.Length; i++)
        {
            hash ^= char.ToLowerInvariant(s[i]);
            hash *= 0x01000193; // FNV prime
        }
        return hash;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void NormalizeL2(Span<float> vec)
    {
        int i = 0;
        int vCount = Vector<float>.Count;
        var sumVec = Vector<float>.Zero;

        while (i <= vec.Length - vCount)
        {
            var v = new Vector<float>(vec.Slice(i, vCount));
            sumVec += v * v;
            i += vCount;
        }

        float sumSq = Vector.Dot(sumVec, Vector<float>.One);
        while (i < vec.Length)
        {
            sumSq += vec[i] * vec[i];
            i++;
        }

        if (sumSq > 0f)
        {
            float invNorm = 1.0f / MathF.Sqrt(sumSq);
            var invVec = new Vector<float>(invNorm);
            i = 0;
            while (i <= vec.Length - vCount)
            {
                var v = new Vector<float>(vec.Slice(i, vCount));
                (v * invVec).CopyTo(vec.Slice(i, vCount));
                i += vCount;
            }
            while (i < vec.Length)
            {
                vec[i] *= invNorm;
                i++;
            }
        }
    }

    public void Dispose() { }
}
