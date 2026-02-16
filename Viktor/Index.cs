namespace Viktor
{
    using HdrHistogram;
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Numerics;
    using System.Runtime.CompilerServices;
    using System.Runtime.InteropServices;
    using System.Runtime.InteropServices.Marshalling;
    using System.Runtime.Intrinsics;
    using System.Runtime.Intrinsics.X86;
    using System.Text;
    using System.Threading.Tasks;
    using ViktorProto.Viktor;

    internal class Index
    {
        public int Dimensions { get; private set; }

        public List<int> Items { get; set; } = new List<int>();

        public List<float[]> Embeddings { get; set; } = new List<float[]>();

        private SpanList<ulong[]> BinEmbeddings = new SpanList<ulong[]>(1024);

        public LongHistogram densityHistogram = new LongHistogram(1, 10000, 3);

        public Index(int dimensions)
        {
            this.Dimensions = dimensions;
        }

        public List<int> FindClosest(float[] queryEmbedding, int topK)
        {
            L2Normalize(queryEmbedding);
            return FindClosestInList(Embeddings, queryEmbedding, topK).Select(x => Items[x]).ToList();
        }

        public static List<int> FindClosestInList(List<float[]> embeddings, float[] queryEmbedding, int topK)
        {
            int n = embeddings.Count;
            if (n == 0 || topK <= 0) return new List<int>(0);
            if (topK > n) topK = n;

            int dim = queryEmbedding.Length;

            // Query must be normalized before this call
            // (and all embeddings too).

            // Max-heap (worst = largest distance)
            var heapIdx = new int[topK];
            var heapDist = new float[topK];
            int heapSize = 0;

            for (int i = 0; i < n; i++)
            {
                var v = embeddings[i];
                float dot = 0f;

                int j = 0;
                for (; j <= dim - 4; j += 4)
                {
                    dot += queryEmbedding[j + 0] * v[j + 0]
                         + queryEmbedding[j + 1] * v[j + 1]
                         + queryEmbedding[j + 2] * v[j + 2]
                         + queryEmbedding[j + 3] * v[j + 3];
                }
                for (; j < dim; j++)
                    dot += queryEmbedding[j] * v[j];

                float dist = 1f - dot; // cosine distance for normalized vectors

                if (heapSize < topK)
                {
                    heapIdx[heapSize] = i;
                    heapDist[heapSize] = dist;
                    HeapifyUpMax(heapDist, heapIdx, heapSize);
                    heapSize++;
                }
                else if (dist < heapDist[0]) // better (closer) than current worst
                {
                    heapDist[0] = dist;
                    heapIdx[0] = i;
                    HeapifyDownMax(heapDist, heapIdx, 0, heapSize);
                }
            }

            // Sort kept results ascending by distance
            var resultIdx = new int[heapSize];
            var resultDist = new float[heapSize];
            Array.Copy(heapIdx, resultIdx, heapSize);
            Array.Copy(heapDist, resultDist, heapSize);
            Array.Sort(resultDist, resultIdx, 0, heapSize);

            var list = new List<int>(heapSize);
            for (int k = 0; k < heapSize; k++) list.Add(resultIdx[k]);
            return list;
        }

        // --- max-heap helpers ---
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void HeapifyUpMax(float[] key, int[] idx, int child)
        {
            while (child > 0)
            {
                int parent = (child - 1) >> 1;
                if (key[child] <= key[parent]) break;
                (key[child], key[parent]) = (key[parent], key[child]);
                (idx[child], idx[parent]) = (idx[parent], idx[child]);
                child = parent;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void HeapifyDownMax(float[] key, int[] idx, int root, int size)
        {
            while (true)
            {
                int left = (root << 1) + 1;
                if (left >= size) break;
                int right = left + 1;
                int largest = (right < size && key[right] > key[left]) ? right : left;
                if (key[root] >= key[largest]) break;
                (key[root], key[largest]) = (key[largest], key[root]);
                (idx[root], idx[largest]) = (idx[largest], idx[root]);
                root = largest;
            }
        }

        public List<int> FindClosestBinary(float[] queryEmbedding, int topK)
        {
            List<int> result;
            lock (this)
            {
                var neighbors = BinaryQuant.GetTopKNearest(queryBits: BinaryQuant.QuantizeSignBits(queryEmbedding), dbBits: BinEmbeddings.AsReadOnlySpan(), k: topK);
                result = neighbors.Select(n => Items[n.Index]).ToList();
            }
            return result;
        }

        public void AddItem(Item item)
        {
            lock (this)
            {
                if (item.Embedding == null || item.Embedding.Length == 0)
                {
                    throw new ArgumentException("Item embedding cannot be null or empty.");
                }

                Items.Add(item.Id);
                L2Normalize(item.Embedding);
                Embeddings.Add(item.Embedding);
                var binEmbedding = BinaryQuant.QuantizeSignBits(item.Embedding);
                int bitCount = 0;
                foreach (var val in binEmbedding)
                {
                    bitCount += BitOperations.PopCount(val);
                }
                densityHistogram.RecordValue(bitCount + 1);
                BinEmbeddings.Add(binEmbedding);
            }
        }

        public void Reindex(List<Item> items)
        {
            Items.Clear();
            Embeddings.Clear();
            foreach (var item in items)
            {
                AddItem(item);
            }
        }

        public static void L2Normalize(float[] v)
        {
            float sumSq = 0f;
            for (int i = 0; i < v.Length; i++)
                sumSq += v[i] * v[i];

            float norm = MathF.Sqrt(sumSq);
            if (norm > 0f)
            {
                float inv = 1f / norm;
                for (int i = 0; i < v.Length; i++)
                    v[i] *= inv;
            }
        }
    }

    public sealed class SpanList<T>
    {
        private T[] _items;
        private int _count;

        public SpanList(int capacity = 4)
        {
            _items = new T[capacity];
            _count = 0;
        }

        public int Count => _count;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Add(T item)
        {
            if (_count == _items.Length)
                Grow();
            _items[_count++] = item;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private void Grow()
        {
            int newSize = _items.Length * 2;
            if (newSize < 4) newSize = 4;
            Array.Resize(ref _items, newSize);
        }

        public ref T this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref _items[index];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<T> AsReadOnlySpan() => new ReadOnlySpan<T>(_items, 0, _count);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Span<T> AsSpan() => new Span<T>(_items, 0, _count);
    }
}
