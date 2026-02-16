namespace ViktorProto
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using Azure.AI.OpenAI;
    using Azure;
    using OpenAI.Embeddings;
    using global::Viktor;

    internal class OpenAIEmbedder : IEmbedder
    {
        private readonly EmbeddingClient _embeddingClient;

        public int Dimensions => 1536; // text-embedding-ada-002 has 1536 dimensions

        /// <summary>
        /// Initializes a new instance of OpenAIEmbedder for Azure OpenAI
        /// </summary>
        /// <param name="endpoint">Azure OpenAI endpoint URL</param>
        /// <param name="apiKey">Azure OpenAI API key</param>
        /// <param name="deploymentName">Name of the embedding model deployment (default: text-embedding-ada-002)</param>
        public OpenAIEmbedder(string endpoint, string apiKey, string deploymentName = "text-embedding-ada-002")
        {
            if (string.IsNullOrWhiteSpace(endpoint))
                throw new ArgumentException("Endpoint cannot be null or empty", nameof(endpoint));
            if (string.IsNullOrWhiteSpace(apiKey))
                throw new ArgumentException("API key cannot be null or empty", nameof(apiKey));
            if (string.IsNullOrWhiteSpace(deploymentName))
                throw new ArgumentException("Deployment name cannot be null or empty", nameof(deploymentName));

            var client = new AzureOpenAIClient(new Uri(endpoint), new AzureKeyCredential(apiKey));
            _embeddingClient = client.GetEmbeddingClient(deploymentName);
        }

        public float[] Embed(string text)
        {
            return this.EmbedAsync(text).GetAwaiter().GetResult();
        }

        /// <summary>
        /// Async version of Embed method for better performance in concurrent scenarios
        /// </summary>
        public async Task<float[]> EmbedAsync(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("Text cannot be null or empty", nameof(text));

            try
            {
                var result = await _embeddingClient.GenerateEmbeddingAsync(text);
                return result.Value.ToFloats().ToArray();
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to generate embedding for text: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Embed multiple strings in a single batch request for improved throughput
        /// </summary>
        /// <param name="texts">Collection of texts to embed</param>
        /// <returns>Array of embeddings in the same order as input texts</returns>
        public float[][] EmbedBatch(IEnumerable<string> texts)
        {
            return EmbedBatchAsync(texts).GetAwaiter().GetResult();
        }

        /// <summary>
        /// Async version of batch embedding for improved throughput when embedding multiple strings
        /// </summary>
        /// <param name="texts">Collection of texts to embed</param>
        /// <returns>Array of embeddings in the same order as input texts</returns>
        public async Task<float[][]> EmbedBatchAsync(IEnumerable<string> texts)
        {
            if (texts == null)
                throw new ArgumentNullException(nameof(texts));

            var textList = texts.ToList();
            if (textList.Count == 0)
                return Array.Empty<float[]>();

            // Validate all texts are non-empty
            for (int i = 0; i < textList.Count; i++)
            {
                if (string.IsNullOrWhiteSpace(textList[i]))
                    throw new ArgumentException($"Text at index {i} cannot be null or empty", nameof(texts));
            }

            try
            {
                var result = await _embeddingClient.GenerateEmbeddingsAsync(textList);
                return result.Value.Select(embedding => embedding.ToFloats().ToArray()).ToArray();
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to generate batch embeddings: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Embed multiple strings in batches with automatic chunking to handle large collections
        /// </summary>
        /// <param name="texts">Collection of texts to embed</param>
        /// <param name="batchSize">Maximum number of texts per batch (default: 100)</param>
        /// <returns>Array of embeddings in the same order as input texts</returns>
        public async Task<float[][]> EmbedBatchChunkedAsync(IEnumerable<string> texts, int batchSize = 100)
        {
            if (texts == null)
                throw new ArgumentNullException(nameof(texts));
            if (batchSize <= 0)
                throw new ArgumentException("Batch size must be greater than 0", nameof(batchSize));

            var textList = texts.ToList();
            if (textList.Count == 0)
                return Array.Empty<float[]>();

            var results = new List<float[]>();

            // Process in chunks
            for (int i = 0; i < textList.Count; i += batchSize)
            {
                var chunk = textList.Skip(i).Take(batchSize);
                var chunkResults = await EmbedBatchAsync(chunk);
                results.AddRange(chunkResults);
            }

            return results.ToArray();
        }
    }
}
