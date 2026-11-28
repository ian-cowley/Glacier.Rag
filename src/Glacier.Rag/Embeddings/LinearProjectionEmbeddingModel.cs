namespace Glacier.Rag.Embeddings;

using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// Vectorized linear projection embedding model implementing AVX-512 / AVX2 / Vector128 GEMV projection and SIMD L2 normalization.
/// Transforms upstream embeddings (e.g. 4096-dim token representations or intermediate embeddings)
/// into a target vector space (e.g. 768 or 384 dimensions) with zero per-call heap allocation on the hot path.
/// </summary>
public sealed class LinearProjectionEmbeddingModel : IEmbeddingModel
{
    private readonly IEmbeddingModel _upstreamModel;
    private readonly int _inputDim;
    private readonly int _outputDim;
    private readonly float[] _weights; // outputDim x inputDim (Row-Major)
    private readonly float[] _bias;    // outputDim

    public int Dimensions => _outputDim;
    public int InputDimensions => _inputDim;
    public IEmbeddingModel UpstreamModel => _upstreamModel;

    public LinearProjectionEmbeddingModel(
        IEmbeddingModel upstreamModel,
        int outputDim,
        float[] weights,
        float[]? bias = null)
    {
        _upstreamModel = upstreamModel ?? throw new ArgumentNullException(nameof(upstreamModel));
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(outputDim, 0);
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(upstreamModel.Dimensions, 0);

        _inputDim = upstreamModel.Dimensions;
        _outputDim = outputDim;
        int expectedWeightCount = checked(_outputDim * _inputDim);

        _weights = weights ?? throw new ArgumentNullException(nameof(weights));
        _bias = bias ?? new float[outputDim];

        if (weights.Length != expectedWeightCount)
            throw new ArgumentException($"Weights length must be {outputDim} x {_inputDim} = {expectedWeightCount}, got {weights.Length}");
        if (_bias.Length != _outputDim)
            throw new ArgumentException($"Bias length must be {outputDim}, got {_bias.Length}");
    }

    /// <summary>
    /// Initializes a linear projection model with deterministic pseudo-random normalized projection weights.
    /// </summary>
    public LinearProjectionEmbeddingModel(
        IEmbeddingModel upstreamModel,
        int outputDim,
        int seed = 1337)
    {
        _upstreamModel = upstreamModel ?? throw new ArgumentNullException(nameof(upstreamModel));
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(outputDim, 0);
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(upstreamModel.Dimensions, 0);

        _inputDim = upstreamModel.Dimensions;
        _outputDim = outputDim;
        int totalWeights = checked(_outputDim * _inputDim);
        _weights = new float[totalWeights];
        _bias = new float[_outputDim];

        // Initialize with scaled Gaussian-like values (Xavier/He initialization style: std = sqrt(2 / inputDim))
        var rng = new Random(seed);
        float scale = MathF.Sqrt(2.0f / _inputDim);
        for (int i = 0; i < _weights.Length; i++)
        {
            // Box-Muller transform for standard normal distribution
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            _weights[i] = (float)randStdNormal * scale;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void GenerateEmbedding(ReadOnlySpan<char> text, Span<float> destination)
    {
        if (destination.Length < _outputDim)
            throw new ArgumentException($"Destination span too small. Expected {_outputDim}, got {destination.Length}");

        // Compute upstream embedding without per-call heap allocations (stackalloc <= 1024, ArrayPool > 1024)
        float[]? rentedInput = null;
        Span<float> inputEmb = _inputDim <= 1024
            ? stackalloc float[_inputDim]
            : (rentedInput = ArrayPool<float>.Shared.Rent(_inputDim)).AsSpan(0, _inputDim);

        try
        {
            _upstreamModel.GenerateEmbedding(text, inputEmb);

            // Compute GEMV: destination = Weights * inputEmb + Bias
            Span<float> outSlice = destination.Slice(0, _outputDim);
            GemvProject(inputEmb, outSlice);

            // Normalize to unit length (SIMD L2 norm)
            SimdEmbeddingMath.NormalizeL2InPlace(outSlice);
        }
        finally
        {
            if (rentedInput != null)
            {
                ArrayPool<float>.Shared.Return(rentedInput);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void GemvProject(ReadOnlySpan<float> src, Span<float> dst)
    {
        fixed (float* pSrc = src)
        fixed (float* pW = _weights)
        fixed (float* pB = _bias)
        fixed (float* pDst = dst)
        {
            for (int r = 0; r < _outputDim; r++)
            {
                float* rowW = pW + (nuint)r * (nuint)_inputDim;
                float sum = pB[r];

                int c = 0;

                // 1. AVX-512 FMA Vector Path (16 floats per iteration)
                if (Avx512F.IsSupported && _inputDim >= 16)
                {
                    var acc = Vector512<float>.Zero;
                    for (; c <= _inputDim - 16; c += 16)
                    {
                        acc = Avx512F.FusedMultiplyAdd(Vector512.Load(pSrc + c), Vector512.Load(rowW + c), acc);
                    }
                    sum += Vector512.Sum(acc);
                }
                // 2. AVX2 / FMA Vector Path (8 floats per iteration)
                else if (Avx2.IsSupported && _inputDim >= 8)
                {
                    var acc = Vector256<float>.Zero;
                    for (; c <= _inputDim - 8; c += 8)
                    {
                        acc = Fma.IsSupported
                            ? Fma.MultiplyAdd(Vector256.Load(pSrc + c), Vector256.Load(rowW + c), acc)
                            : Vector256.Add(acc, Vector256.Multiply(Vector256.Load(pSrc + c), Vector256.Load(rowW + c)));
                    }
                    sum += Vector256.Sum(acc);
                }
                // 3. Vector128 (ARM64 NEON / SSE) Vector Path (4 floats per iteration)
                else if (Vector128.IsHardwareAccelerated && _inputDim >= 4)
                {
                    var acc = Vector128<float>.Zero;
                    for (; c <= _inputDim - 4; c += 4)
                    {
                        var vSrc = Vector128.Load(pSrc + c);
                        var vW = Vector128.Load(rowW + c);
                        acc = Fma.IsSupported
                            ? Fma.MultiplyAdd(vSrc, vW, acc)
                            : AdvSimd.IsSupported
                                ? AdvSimd.FusedMultiplyAdd(acc, vSrc, vW)
                                : Vector128.Add(acc, Vector128.Multiply(vSrc, vW));
                    }
                    sum += Vector128.Sum(acc);
                }

                // 4. Scalar cleanup
                for (; c < _inputDim; c++)
                {
                    sum += pSrc[c] * rowW[c];
                }

                pDst[r] = sum;
            }
        }
    }

    public void Dispose()
    {
        _upstreamModel?.Dispose();
    }
}
