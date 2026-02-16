namespace Viktor
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using System.IO; // Needed for MemoryStream
    using HdrHistogram;
    using System.Diagnostics;

    internal class Kernel
    {
        public IStore Store { get; private set; }

        public IEmbedder Embedder { get; private set; }

        public int MaxId { get; private set; }

        private Index index;

        private Stopwatch sw = new Stopwatch();

        public LongHistogram EmbedHist { get; } = new LongHistogram(1, TimeStamp.Minutes(1), 3);
        public LongHistogram SearchHist { get; } = new LongHistogram(1, TimeStamp.Minutes(1), 3);
        public LongHistogram SearchBinHist { get; } = new LongHistogram(1, TimeStamp.Minutes(1), 3);
        public LongHistogram ReadHist { get; } = new LongHistogram(1, TimeStamp.Minutes(1), 3);
        public LongHistogram ReorderHist { get; } = new LongHistogram(1, TimeStamp.Minutes(1), 3);

        public Kernel(IStore store, IEmbedder embedder)
        {
            this.Store = store;
            this.Embedder = embedder;
            this.index = new Index(embedder.Dimensions);
            this.MaxId = 0;
        }

        public void LoadIndex()
        {
            int size = 0;
            foreach (var id in Store.ReadAllItems())
            {
                if (Store.ReadItem(id, out var data))
                {
                    var item = ProtoBuf.Serializer.Deserialize<Item>(new ReadOnlyMemory<byte>(data));
                    index.AddItem(item);
                    if (item.Id > MaxId)
                    {
                        MaxId = item.Id;
                    }
                    size++;
                }
            }
            Console.WriteLine($"Loaded {size} items into index.");
            if (size > 0)
            {
                Console.WriteLine("Bit index density:");
                Console.WriteLine($"avg: {index.densityHistogram.GetMean()}");
                Console.WriteLine($"p10: {index.densityHistogram.GetValueAtPercentile(10)}");
                Console.WriteLine($"p90: {index.densityHistogram.GetValueAtPercentile(90)}");
            }

        }

        public void SaveItem(Item item)
        {
            if (item.Embedding == null || item.Embedding.Length == 0)
            {
                item.Embedding = Embedder.Embed(item.SemanticKey);
            }

            // Fix for CS0815 and CS1501:
            // Serialize to a MemoryStream, then get the byte array.
            using (var ms = new MemoryStream())
            {
                ProtoBuf.Serializer.Serialize(ms, item);
                var data = ms.ToArray();
                Store.UpsertItem(item.Id, data);
            }
            index.AddItem(item);
        }

        public void SaveItems(IEnumerable<Item> items)
        {
            // Calculate embeddings for items that don't have them
            var itemsToEmbed = items.Where(i => i.Embedding == null || i.Embedding.Length == 0).ToList();
            if (itemsToEmbed.Count > 0)
            {
                var semanticKeys = itemsToEmbed.Select(i => i.SemanticKey).ToList();
                var embeddings = Embedder.EmbedBatch(semanticKeys);
                for (int i = 0; i < itemsToEmbed.Count; i++)
                {
                    itemsToEmbed[i].Embedding = embeddings[i];
                }
            }

            foreach (Item item in items)
            {
                SaveItem(item);
            }
        }


        public IEnumerable<string> Search(string query, int topK = 5)
        {
            sw.Restart();
            var queryEmbedding = Embedder.Embed(query);
            sw.Stop();
            EmbedHist.RecordValue(sw.ElapsedTicks * 1_000_000 / Stopwatch.Frequency);
            return this.Search(queryEmbedding, topK);
        }

        public IEnumerable<string> Search(float[] queryEmbedding, int topK = 5)
        { 
            sw.Restart();
            var itemIds = index.FindClosest(queryEmbedding, topK);
            sw.Stop();
            SearchHist.RecordValue(sw.ElapsedTicks * 1_000_000 / Stopwatch.Frequency);
            var results = new List<string>();
            sw.Restart();
            foreach (var id in itemIds)
            {
                if (Store.ReadItem(id, out var data))
                {
                    // Deserialize to Item, then add the Payload (string) to results
                    var item = ProtoBuf.Serializer.Deserialize<Item>(new ReadOnlyMemory<byte>(data));
                    results.Add(item.Payload);
                }
            }
            sw.Stop();
            ReadHist.RecordValue(sw.ElapsedTicks * 1_000_000 / Stopwatch.Frequency);
            return results;
        }

        public IEnumerable<int> SearchIds(float[] queryEmbedding, int topK = 5)
        { 
            sw.Restart();
            var itemIds = index.FindClosest(queryEmbedding, topK);
            sw.Stop();
            SearchHist.RecordValue(sw.ElapsedTicks * 1_000_000 / Stopwatch.Frequency);
            return itemIds;
        }

        public IEnumerable<string> SearchBinary(string query, int topK = 5, bool reorder = false, int overshoot = 0)
        {
            sw.Restart();
            var queryEmbedding = Embedder.Embed(query);
            sw.Stop();
            EmbedHist.RecordValue(sw.ElapsedTicks * 1_000_000 / Stopwatch.Frequency);
            return this.SearchBinary(queryEmbedding, topK, reorder, overshoot);
        }

        public IEnumerable<string> SearchBinary(float[] queryEmbedding, int topK = 5, bool reorder = false, int overshoot = 0)
        { 
            if (!reorder) overshoot = 0;

            sw.Restart();
            var itemIds = index.FindClosestBinary(queryEmbedding, topK + overshoot);
            sw.Stop();
            SearchBinHist.RecordValue(sw.ElapsedTicks * 1_000_000 / Stopwatch.Frequency);

            var results = new List<string>();
            var fullVectors = new List<float[]>();
            sw.Restart();
            foreach (var id in itemIds)
            {
                if (Store.ReadItem(id, out var data))
                {
                    // Deserialize to Item, then add the Payload (string) to results
                    var item = ProtoBuf.Serializer.Deserialize<Item>(new ReadOnlyMemory<byte>(data));
                    results.Add(item.Payload);
                    fullVectors.Add(item.Embedding);
                }
            }
            sw.Stop();
            ReadHist.RecordValue(sw.ElapsedTicks * 1_000_000 / Stopwatch.Frequency);

            
            if (reorder)
            {
                sw.Restart();
                var result = Index.FindClosestInList(fullVectors, queryEmbedding, topK);
                var ordered = new List<string>();
                foreach (var idx in result)
                {
                    ordered.Add(results[idx]);
                    if (ordered.Count >= topK) break;
                }
                sw.Stop();
                ReorderHist.RecordValue(sw.ElapsedTicks * 1_000_000 / Stopwatch.Frequency);

                return ordered;
            }
            return results;
        }

        public IEnumerable<int> SearchBinaryIds(float[] queryEmbedding, int topK = 5, bool reorder = false, int overshoot = 0)
        { 
            if (!reorder) overshoot = 0;

            sw.Restart();
            var itemIds = index.FindClosestBinary(queryEmbedding, topK + overshoot);
            sw.Stop();
            SearchBinHist.RecordValue(sw.ElapsedTicks * 1_000_000 / Stopwatch.Frequency);

            if (!reorder)
            {
                return itemIds;
            }

            var results = new List<int>();
            var fullVectors = new List<float[]>();
            sw.Restart();
            foreach (var id in itemIds)
            {
                if (Store.ReadItem(id, out var data))
                {
                    var item = ProtoBuf.Serializer.Deserialize<Item>(new ReadOnlyMemory<byte>(data));
                    results.Add(id);
                    fullVectors.Add(item.Embedding);
                }
            }
            sw.Stop();
            ReadHist.RecordValue(sw.ElapsedTicks * 1_000_000 / Stopwatch.Frequency);

            sw.Restart();
            var reorderedIndices = Index.FindClosestInList(fullVectors, queryEmbedding, topK);
            var ordered = new List<int>();
            foreach (var idx in reorderedIndices)
            {
                ordered.Add(results[idx]);
                if (ordered.Count >= topK) break;
            }
            sw.Stop();
            ReorderHist.RecordValue(sw.ElapsedTicks * 1_000_000 / Stopwatch.Frequency);

            return ordered;
        }

        public Item? GetItem(int id)
        {
            if (Store.ReadItem(id, out var data))
            {
                return ProtoBuf.Serializer.Deserialize<Item>(new ReadOnlyMemory<byte>(data));
            }
            return null;
        }

        public Index Index => index;
    }
}
