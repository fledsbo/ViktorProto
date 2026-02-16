namespace ViktorProto.Viktor
{
    using System;
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Numerics;
    using System.Numerics;
    using System.Runtime.CompilerServices;
    using System.Runtime.CompilerServices;
    using System.Security.Cryptography.X509Certificates;
    using System.Text;
    using System.Threading.Tasks;
    using static ViktorProto.Viktor.BinaryQuant;

    internal class BinaryQuant
    {
        // Represents one match in the result set
        public readonly struct Neighbor
        {
            public readonly int Index;     // index in the database
            public readonly int Distance;  // Hamming distance
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public Neighbor(int index, int distance)
            {
                Index = index;
                Distance = distance;
            }
        }

        // Quantize a float embedding (length d) into 1-bit-per-dim blocks of ulong.
        // Output length = (d + 63) / 64.
        public static ulong[] QuantizeSignBits(ReadOnlySpan<float> embedding)
        {
            int dim = embedding.Length;
            int blocks = (dim + 63) >> 6; // divide by 64 round up
            var result = new ulong[blocks];

            int i = 0;
            int blockIndex = 0;

            // We'll fill 64 bits at a time into a ulong.
            while (i < dim)
            {
                ulong bits = 0UL;
                int bitCount = Math.Min(64, dim - i);

                // Unroll 8 at a time for fewer branches.
                int j = 0;
                for (; j <= bitCount - 8; j += 8)
                {
                    // Manually build mask without branching
                    bits |= (embedding[i + j + 0] >= 0f ? 1UL : 0UL) << (j + 0);
                    bits |= (embedding[i + j + 1] >= 0f ? 1UL : 0UL) << (j + 1);
                    bits |= (embedding[i + j + 2] >= 0f ? 1UL : 0UL) << (j + 2);
                    bits |= (embedding[i + j + 3] >= 0f ? 1UL : 0UL) << (j + 3);
                    bits |= (embedding[i + j + 4] >= 0f ? 1UL : 0UL) << (j + 4);
                    bits |= (embedding[i + j + 5] >= 0f ? 1UL : 0UL) << (j + 5);
                    bits |= (embedding[i + j + 6] >= 0f ? 1UL : 0UL) << (j + 6);
                    bits |= (embedding[i + j + 7] >= 0f ? 1UL : 0UL) << (j + 7);
                }

                // Remainder <8
                for (; j < bitCount; j++)
                {
                    bits |= (embedding[i + j] >= 0f ? 1UL : 0UL) << j;
                }

                result[blockIndex++] = bits;
                i += bitCount;
            }

            return result;
        }

        // Variant using Vector<float> to potentially speed up sign extraction
        // on large dims (>= Vector<float>.Count). This can be faster for big models
        // (e.g. 768, 1024, 3072+). Falls back to scalar build per-ulong.
        // Note: More complex, but still branchless and safe.
        public static ulong[] QuantizeSignBitsVectorized(ReadOnlySpan<float> embedding)
        {
            int dim = embedding.Length;
            int blocks = (dim + 63) >> 6;
            var result = new ulong[blocks];

            // We'll still output as 64-bit chunks. We'll just speed up reading
            // the sign mask for those 64 floats.
            int i = 0;
            int blockIndex = 0;

            int vecWidth = Vector<float>.Count;
            Span<float> tmp = stackalloc float[64]; // local buffer for one block

            while (i < dim)
            {
                int bitCount = Math.Min(64, dim - i);

                // Copy up to 64 floats into tmp (contiguous) so we can vectorize on it
                embedding.Slice(i, bitCount).CopyTo(tmp);

                ulong bits = 0UL;

                int j = 0;
                // Vectorized loop in chunks of vecWidth
                for (; j <= bitCount - vecWidth; j += vecWidth)
                {
                    var v = new Vector<float>(tmp.Slice(j));
                    // Compare >=0f -> returns all-bits 1 for true lanes, 0 for false
                    var mask = Vector.GreaterThanOrEqual(v, Vector<float>.Zero);

                    // Extract lane-wise bits from mask.
                    // We still have to walk the lanes because .NET doesn't expose
                    // direct bitpack from a Vector<T> to integer.
                    for (int lane = 0; lane < vecWidth; lane++)
                    {
                        // true => all bits 1 => GetElement() will be all-ones uint
                        // false => all bits 0.
                        // We just need 1 bit.
                        if (mask[lane] != 0)
                        {
                            int bitPos = j + lane;
                            bits |= 1UL << bitPos;
                        }
                    }
                }

                // Remainder scalar
                for (; j < bitCount; j++)
                {
                    if (tmp[j] >= 0f)
                    {
                        bits |= 1UL << j;
                    }
                }

                result[blockIndex++] = bits;
                i += bitCount;
            }

            return result;
        }

        // Approximate cosine similarity from Hamming distance.
        // Range ~[-1, 1].
        // dim = original float embedding dimension.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float CosineSimilarityApprox(int hammingDistance, int dim)
        {
            // matches = dim - hamming
            // sim ≈ 1 - 2 * ham / dim
            return 1f - (2f * hammingDistance / dim);
        }

        // Convenience end-to-end
        public static float ApproximateCosineFromFloatEmbeddings(
            ReadOnlySpan<float> a,
            ReadOnlySpan<float> b,
            bool useVectorized = false)
        {
            if (a.Length != b.Length)
                throw new ArgumentException("Embeddings must have same dimensionality");

            ulong[] qa = useVectorized
                ? QuantizeSignBitsVectorized(a)
                : QuantizeSignBits(a);

            ulong[] qb = useVectorized
                ? QuantizeSignBitsVectorized(b)
                : QuantizeSignBits(b);

            int ham = HammingDistance(qa, qb);
            return CosineSimilarityApprox(ham, a.Length);
        }

    

    // Public API:
    //   queryBits: packed 1-bit embedding of the query (len = L)
    //   dbBits:   array of packed embeddings, each dbBits[i].Length == L
    //   k:        how many nearest you want
    //
    // Returns an array of length <= k, sorted by ascending Hamming distance.
    public static Neighbor[] GetTopKNearest(
        ReadOnlySpan<ulong> queryBits,
        ReadOnlySpan<ulong[]> dbBits,
        int k)
        {
            if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
            if (dbBits.Length == 0) return Array.Empty<Neighbor>();

            // We'll keep a fixed-size max-heap of capacity k.
            // The "worst" (largest distance) lives at heap[0].
            // If we find something better than heap[0], we replace and heapify-down.
            var heap = new Neighbor[Math.Min(k, dbBits.Length)];
            int heapSize = 0;

            // Precompute length once for bounds check elision
            int codeLen = queryBits.Length;

            for (int i = 0; i < dbBits.Length; i++)
            {
                ulong[] candidate = dbBits[i];
                // Safety check (debug-style). In production, assume uniform.
                if (candidate.Length != codeLen)
                    throw new ArgumentException("All embeddings must have equal length");

                int dist = HammingDistance(queryBits, candidate);

                if (heapSize < heap.Length)
                {
                    // Heap not full yet -> push
                    heap[heapSize++] = new Neighbor(i, dist);
                    HeapifyUpMax(heap, heapSize - 1);
                }
                else if (dist < heap[0].Distance)
                {
                    // Better than current worst -> replace root and fix
                    heap[0] = new Neighbor(i, dist);
                    HeapifyDownMax(heap, 0, heapSize);
                }
            }

            // Now heap holds up to k best, but in max-heap order.
            // We want ascending by distance.
            Array.Sort(heap, 0, heapSize, NeighborDistanceComparer.Instance);

            if (heapSize == heap.Length) return heap;
            // db smaller than k
            var trimmed = new Neighbor[heapSize];
            Array.Copy(heap, trimmed, heapSize);
            return trimmed;
        }

        // ----------------------------
        // Core inner distance routine
        // ----------------------------

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int HammingDistance(ReadOnlySpan<ulong> a, ReadOnlySpan<ulong> b)
        {
            int sum = 0;
            // unroll by 4 to reduce loop overhead
            int i = 0;
            int len = a.Length;

            for (; i <= len - 4; i += 4)
            {
                sum += BitOperations.PopCount(a[i + 0] ^ b[i + 0]);
                sum += BitOperations.PopCount(a[i + 1] ^ b[i + 1]);
                sum += BitOperations.PopCount(a[i + 2] ^ b[i + 2]);
                sum += BitOperations.PopCount(a[i + 3] ^ b[i + 3]);
            }
            for (; i < len; i++)
            {
                sum += BitOperations.PopCount(a[i] ^ b[i]);
            }

            return sum;
        }

        // ----------------------------
        // Max-heap helpers
        // ----------------------------

        // We maintain a max-heap by Neighbor.Distance.
        // Parent index p, children 2p+1 and 2p+2.

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void HeapifyUpMax(Neighbor[] heap, int idx)
        {
            // Bubble up while current > parent
            while (idx > 0)
            {
                int parent = (idx - 1) >> 1;
                if (heap[idx].Distance <= heap[parent].Distance)
                    break;

                (heap[idx], heap[parent]) = (heap[parent], heap[idx]);
                idx = parent;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void HeapifyDownMax(Neighbor[] heap, int idx, int heapSize)
        {
            // Push down while current < one of the children
            while (true)
            {
                int left = (idx << 1) + 1;
                int right = left + 1;
                if (left >= heapSize) break;

                int swapIdx = left;
                if (right < heapSize && heap[right].Distance > heap[left].Distance)
                {
                    swapIdx = right;
                }

                if (heap[idx].Distance >= heap[swapIdx].Distance)
                    break;

                (heap[idx], heap[swapIdx]) = (heap[swapIdx], heap[idx]);
                idx = swapIdx;
            }
        }

    // ----------------------------
    // Sort comparer (ascending distance)
    // ----------------------------

    private sealed class NeighborDistanceComparer : System.Collections.Generic.IComparer<Neighbor>
    {
        public static readonly NeighborDistanceComparer Instance = new NeighborDistanceComparer();
        public int Compare(Neighbor x, Neighbor y)
        {
            int d = x.Distance - y.Distance;
            if (d != 0) return d;
            // tie-break by index just for determinism
            return x.Index - y.Index;
        }
    }
}
}

