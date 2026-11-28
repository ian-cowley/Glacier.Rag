namespace Glacier.Rag.Embeddings;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

internal static class SimdEmbeddingMath
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void NormalizeL2InPlace(Span<float> vec, float minSumSqThreshold = 0f)
    {
        int len = vec.Length;
        if (len == 0)
            return;

        float sumSq = 0f;
        int i = 0;
        ref float pVec = ref MemoryMarshal.GetReference(vec);

        if (Vector512.IsHardwareAccelerated && len >= 16)
        {
            var acc = Vector512<float>.Zero;
            for (; i <= len - 16; i += 16)
            {
                var v = Vector512.LoadUnsafe(ref Unsafe.Add(ref pVec, i));
                acc = Avx512F.IsSupported
                    ? Avx512F.FusedMultiplyAdd(v, v, acc)
                    : acc + (v * v);
            }
            sumSq += Vector512.Sum(acc);
        }
        else if (Vector256.IsHardwareAccelerated && len >= 8)
        {
            var acc = Vector256<float>.Zero;
            for (; i <= len - 8; i += 8)
            {
                var v = Vector256.LoadUnsafe(ref Unsafe.Add(ref pVec, i));
                acc = Fma.IsSupported
                    ? Fma.MultiplyAdd(v, v, acc)
                    : acc + (v * v);
            }
            sumSq += Vector256.Sum(acc);
        }
        else if (Vector128.IsHardwareAccelerated && len >= 4)
        {
            var acc = Vector128<float>.Zero;
            for (; i <= len - 4; i += 4)
            {
                var v = Vector128.LoadUnsafe(ref Unsafe.Add(ref pVec, i));
                acc = Fma.IsSupported
                    ? Fma.MultiplyAdd(v, v, acc)
                    : AdvSimd.IsSupported
                        ? AdvSimd.FusedMultiplyAdd(acc, v, v)
                        : acc + (v * v);
            }
            sumSq += Vector128.Sum(acc);
        }

        for (; i < len; i++)
        {
            float val = Unsafe.Add(ref pVec, i);
            sumSq += val * val;
        }

        if (sumSq > minSumSqThreshold)
        {
            float invNorm = 1.0f / MathF.Sqrt(sumSq);
            i = 0;

            if (Vector512.IsHardwareAccelerated && len >= 16)
            {
                var invVec = Vector512.Create(invNorm);
                for (; i <= len - 16; i += 16)
                {
                    var v = Vector512.LoadUnsafe(ref Unsafe.Add(ref pVec, i));
                    (v * invVec).StoreUnsafe(ref Unsafe.Add(ref pVec, i));
                }
            }
            else if (Vector256.IsHardwareAccelerated && len >= 8)
            {
                var invVec = Vector256.Create(invNorm);
                for (; i <= len - 8; i += 8)
                {
                    var v = Vector256.LoadUnsafe(ref Unsafe.Add(ref pVec, i));
                    (v * invVec).StoreUnsafe(ref Unsafe.Add(ref pVec, i));
                }
            }
            else if (Vector128.IsHardwareAccelerated && len >= 4)
            {
                var invVec = Vector128.Create(invNorm);
                for (; i <= len - 4; i += 4)
                {
                    var v = Vector128.LoadUnsafe(ref Unsafe.Add(ref pVec, i));
                    (v * invVec).StoreUnsafe(ref Unsafe.Add(ref pVec, i));
                }
            }

            for (; i < len; i++)
            {
                Unsafe.Add(ref pVec, i) *= invNorm;
            }
        }
    }
}
