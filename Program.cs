using global::Viktor;
using HdrHistogram;
using Microsoft.Extensions.Configuration;
using System.Diagnostics;
using System.Linq;
using ProtoBuf;
using OpenAI.RealtimeConversation;

namespace ViktorProto
{
    public static class EnumerableExtensions
    {
        public static IEnumerable<List<T>> Batch<T>(this IEnumerable<T> source, int size)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            if (size <= 0) throw new ArgumentOutOfRangeException(nameof(size));

            List<T>? batch = null;

            foreach (var item in source)
            {
                batch ??= new List<T>(size);
                batch.Add(item);

                if (batch.Count == size)
                {
                    yield return batch;
                    batch = null;
                }
            }

            // Return the last batch if it has any items
            if (batch != null && batch.Count > 0)
                yield return batch;
        }
    }   

    internal class Program
    {
        [ProtoContract]
        private class Query
        {
            [ProtoMember(1)]
            public string QueryString { get; set; } = string.Empty;

            [ProtoMember(2)]
            public float[] Embedding { get; set; } = Array.Empty<float>();
        }

        static async Task Main(string[] args)
        {
            // Build configuration from appsettings.json
            var configuration = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("appsettings.json", optional: false, reloadOnChange: true)
                .Build();

            // Read Azure OpenAI configuration
            var azureOpenAIConfig = configuration.GetSection("AzureOpenAI");
            var endpoint = azureOpenAIConfig["Endpoint"]
                ?? throw new InvalidOperationException("Azure OpenAI Endpoint not found in configuration");
            var apiKey = azureOpenAIConfig["ApiKey"]
                ?? throw new InvalidOperationException("Azure OpenAI ApiKey not found in configuration");
            var deploymentName = azureOpenAIConfig["DeploymentName"] ?? "text-embedding-ada-002";

            // Initialize components
            var embedder = new OpenAIEmbedder(endpoint, apiKey, deploymentName);
            var store = new SimpleStore();
            var kernel = new Kernel(store, embedder);
            kernel.LoadIndex();

            Console.WriteLine("Viktor Proto initialized successfully!");
            Console.WriteLine($"Using Azure OpenAI endpoint: {endpoint}");
            Console.WriteLine($"Using deployment: {deploymentName}");

            // Parse arguments
            if (args.Length == 0)
            {
                Console.WriteLine("Please provide a command: serve [port] | inputfile <file> | search <query> [topK] | preparequeries <infile> <outfile> | readqueries <file> | testqueries <file>");
                return;
            }
            var command = args[0].ToLower();

            switch (command)
            {
                case "serve":
                    {
                        int port = 5000;
                        if (args.Length >= 2 && int.TryParse(args[1], out var parsedPort))
                        {
                            port = parsedPort;
                        }

                        await StartWebService(kernel, embedder, port);
                        break;
                    }

                case "inputfile":
                    {
                        if (args.Length < 2) { Console.WriteLine("Please provide a file path to read."); return; }
                        var filePath = args[1];
                        if (!File.Exists(filePath))
                        {
                            Console.WriteLine($"File not found: {filePath}");
                            return;
                        }
                        var lines = File.ReadAllLines(filePath);
                        int id = kernel.MaxId;
                        foreach (var batch in lines.Batch(20))
                        {
                            var items = batch.Select(x => new Item { SemanticKey = x, Payload = x, Id = ++id });
                            kernel.SaveItems(items);
                        }
                        break;
                    }

                case "search":
                    {
                        if (args.Length < 2)
                        {
                            Console.WriteLine("Please provide a search query.");
                            return;
                        }

                        // Parse topK from the last argument if it's a number
                        int topK = 5;
                        var queryArgs = args.Skip(1).ToList();

                        if (queryArgs.Count > 1 && int.TryParse(queryArgs.Last(), out var parsedTopK))
                        {
                            topK = parsedTopK;
                            queryArgs.RemoveAt(queryArgs.Count - 1); // Remove the topK from query args
                        }

                        // Concatenate all remaining arguments to form the complete query
                        var query = string.Join(" ", queryArgs);

                        var results = kernel.Search(query, topK);
                        Console.WriteLine($"Top {topK} results for query '{query}':");
                        foreach (var result in results)
                        {
                            Console.WriteLine(result);
                        }

                        results = kernel.SearchBinary(query, topK);
                        Console.WriteLine($"Top {topK} results for query '{query}' (binary):");
                        foreach (var result in results)
                        {
                            Console.WriteLine(result);
                        }

                        break;
                    }

                case "preparequeries":
                    {
                        if (args.Length < 3) { Console.WriteLine("Usage: preparequeries <infile> <outfile>"); return; }
                        var filePath = args[1];
                        var outFilePath = args[2];
                        if (!File.Exists(filePath))
                        {
                            Console.WriteLine($"File not found: {filePath}");
                            return;
                        }

                        var lines = File.ReadAllLines(filePath);
                        using var outFile = File.OpenWrite(outFilePath);

                        Console.WriteLine($"Processing {lines.Length} queries in batches of 20...");
                        int batchNumber = 0;
                        int totalProcessed = 0;

                        foreach (var batch in lines.Batch(20))
                        {
                            batchNumber++;
                            Console.WriteLine($"Processing batch {batchNumber} with {batch.Count} queries...");

                            try
                            {
                                // Use batch embedding for efficiency
                                var embeddings = await embedder.EmbedBatchAsync(batch);

                                // Create Query objects and serialize them to protobuf
                                for (int i = 0; i < batch.Count; i++)
                                {
                                    var query = new Query
                                    {
                                        QueryString = batch[i],
                                        Embedding = embeddings[i]
                                    };

                                    // Serialize to protobuf and write to file
                                    Serializer.SerializeWithLengthPrefix(outFile, query, PrefixStyle.Base128);
                                    totalProcessed++;
                                }

                                Console.WriteLine($"Completed batch {batchNumber}");
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"Error processing batch {batchNumber}: {ex.Message}");
                            }
                        }

                        Console.WriteLine($"Finished processing {totalProcessed} queries. Protobuf data written to {outFilePath}");
                        break;
                    }

                case "testqueries":
                    {
                        if (args.Length < 2) { Console.WriteLine("Usage: readqueries <file>"); return; }
                        var filePath = args[1];
                        if (!File.Exists(filePath))
                        {
                            Console.WriteLine($"File not found: {filePath}");
                            return;
                        }

                        using var inFile = File.OpenRead(filePath);
                        var queries = new List<Query>();

                        try
                        {
                            while (inFile.Position < inFile.Length)
                            {
                                var query = Serializer.DeserializeWithLengthPrefix<Query>(inFile, PrefixStyle.Base128);
                                if (query != null)
                                {
                                    queries.Add(query);
                                }
                            }
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Error reading protobuf file: {ex.Message}");
                            return;
                        }

                        var binHist = new LongHistogram(1, TimeStamp.Minutes(1), 3);
                        var regHist = new LongHistogram(1, TimeStamp.Minutes(1), 3);
                        var matchHist = new LongHistogram(1, 11, 3);
                        var sw = new Stopwatch();

                        for (int i = 0; i < 1; i++)
                        {
                            foreach (var query in queries)
                            {
                                sw.Restart();
                                var regResults = kernel.Search(query.Embedding, 10);
                                sw.Stop();
                                regHist.RecordValue(sw.ElapsedTicks * 1_000_000 / Stopwatch.Frequency);

                                sw.Restart();
                                var binResults = kernel.SearchBinary(query.Embedding, 10, true, 30);
                                sw.Stop();
                                binHist.RecordValue(sw.ElapsedTicks * 1_000_000 / Stopwatch.Frequency);
                                int match = 0;
                                foreach (var res in regResults)
                                {
                                    if (binResults.Any(x => x == res))
                                    {
                                        match++;
                                    }
                                }
                                matchHist.RecordValue(match);
                            }
                        }

                        // Print percentiles:
                        Console.WriteLine($"Regular: P90: {kernel.SearchHist.GetValueAtPercentile(90)}  P95: {kernel.SearchHist.GetValueAtPercentile(95)}  P99: {kernel.SearchHist.GetValueAtPercentile(99)} ");
                        Console.WriteLine($"Binary:  P90: {kernel.SearchBinHist.GetValueAtPercentile(90)}  P95: {kernel.SearchBinHist.GetValueAtPercentile(95)}  P99: {kernel.SearchBinHist.GetValueAtPercentile(99)} ");
                        Console.WriteLine($"BinTot:  P90: {binHist.GetValueAtPercentile(90)}  P95: {binHist.GetValueAtPercentile(95)}  P99: {binHist.GetValueAtPercentile(99)} ");

                        Console.WriteLine($"Read:    P90: {kernel.ReadHist.GetValueAtPercentile(90)}  P95: {kernel.ReadHist.GetValueAtPercentile(95)}  P99: {kernel.ReadHist.GetValueAtPercentile(99)} ");
                        Console.WriteLine($"Reorder: P90: {kernel.ReorderHist.GetValueAtPercentile(90)}  P95: {kernel.ReorderHist.GetValueAtPercentile(95)}  P99: {kernel.ReorderHist.GetValueAtPercentile(99)} ");

                        Console.WriteLine($"Matching: Avg: {matchHist.GetMean()}  P10: {matchHist.GetValueAtPercentile(10)}");

                        break;

                    }


                default:
                    Console.WriteLine("Unknown command.");
                    return;
            }
        }

        private static async Task StartWebService(Kernel kernel, OpenAIEmbedder embedder, int port)
        {
            var builder = WebApplication.CreateBuilder();
            
            // Add services
            builder.Services.AddSingleton(kernel);
            builder.Services.AddSingleton(embedder);
            builder.Services.AddEndpointsApiExplorer();
            builder.Services.AddSwaggerGen(c =>
            {
                c.SwaggerDoc("v1", new() { Title = "Viktor Search API", Version = "v1" });
            });
            builder.Services.AddCors(options =>
            {
                options.AddDefaultPolicy(policy =>
                {
                    policy.AllowAnyOrigin()
                          .AllowAnyMethod()
                          .AllowAnyHeader();
                });
            });

            // Configure Kestrel to listen on specified port
            builder.WebHost.UseUrls($"http://localhost:{port}");

            var app = builder.Build();

            // Configure middleware
            app.UseSwagger();
            app.UseSwaggerUI(c =>
            {
                c.SwaggerEndpoint("/swagger/v1/swagger.json", "Viktor Search API v1");
                c.RoutePrefix = string.Empty; // Swagger at root
            });
            app.UseCors();

            // API Endpoints

            // Search endpoint - finds N closest records based on query text
            app.MapPost("/api/search", async (SearchRequest request, Kernel kern, OpenAIEmbedder emb) =>
            {
                try
                {
                    if (string.IsNullOrWhiteSpace(request.Query))
                    {
                        return Results.BadRequest(new { error = "Query cannot be empty" });
                    }

                    int topK = request.TopK ?? 10;
                    if (topK <= 0 || topK > 100)
                    {
                        return Results.BadRequest(new { error = "TopK must be between 1 and 100" });
                    }

                    // Embed the query
                    var queryEmbedding = await emb.EmbedAsync(request.Query);

                    // Search using ANN
                    var resultIds = request.UseBinary == true
                        ? kern.SearchBinaryIds(queryEmbedding, topK, true, request.BinaryTopK ?? 100)
                        : kern.SearchIds(queryEmbedding, topK);

                    // Retrieve the items
                    var items = resultIds
                        .Select(id => kern.GetItem(id))
                        .Where(item => item != null)
                        .Select(item => new SearchResultItem
                        {
                            Id = item!.Id,
                            SemanticKey = item.SemanticKey,
                            Payload = item.Payload
                        })
                        .ToList();

                    return Results.Ok(new SearchResponse
                    {
                        Query = request.Query,
                        Results = items,
                        Count = items.Count
                    });
                }
                catch (Exception ex)
                {
                    return Results.Problem(
                        detail: ex.Message,
                        statusCode: 500,
                        title: "Search failed");
                }
            })
            .WithName("Search")
            .WithDescription("Search for items using text query (ANN on embeddings)");

            // Search by embedding - finds N closest records based on pre-computed embedding
            app.MapPost("/api/search/embedding", (SearchByEmbeddingRequest request, Kernel kern) =>
            {
                try
                {
                    if (request.Embedding == null || request.Embedding.Length == 0)
                    {
                        return Results.BadRequest(new { error = "Embedding cannot be empty" });
                    }

                    int topK = request.TopK ?? 10;
                    if (topK <= 0 || topK > 100)
                    {
                        return Results.BadRequest(new { error = "TopK must be between 1 and 100" });
                    }

                    // Search using ANN
                    var resultIds = request.UseBinary == true
                        ? kern.SearchBinaryIds(request.Embedding, topK, true, request.BinaryTopK ?? 100)
                        : kern.SearchIds(request.Embedding, topK);

                    // Retrieve the items
                    var items = resultIds
                        .Select(id => kern.GetItem(id))
                        .Where(item => item != null)
                        .Select(item => new SearchResultItem
                        {
                            Id = item!.Id,
                            SemanticKey = item.SemanticKey,
                            Payload = item.Payload
                        })
                        .ToList();

                    return Results.Ok(new SearchResponse
                    {
                        Query = "[Pre-computed embedding]",
                        Results = items,
                        Count = items.Count
                    });
                }
                catch (Exception ex)
                {
                    return Results.Problem(
                        detail: ex.Message,
                        statusCode: 500,
                        title: "Search failed");
                }
            })
            .WithName("SearchByEmbedding")
            .WithDescription("Search for items using pre-computed embedding (ANN on embeddings)");

            // Get by ID - retrieves one record by document ID
            app.MapGet("/api/items/{id:int}", (int id, Kernel kern) =>
            {
                try
                {
                    var item = kern.GetItem(id);
                    if (item == null)
                    {
                        return Results.NotFound(new { error = $"Item with ID {id} not found" });
                    }

                    return Results.Ok(new ItemResponse
                    {
                        Id = item.Id,
                        SemanticKey = item.SemanticKey,
                        Payload = item.Payload,
                        HasEmbedding = item.Embedding != null && item.Embedding.Length > 0
                    });
                }
                catch (Exception ex)
                {
                    return Results.Problem(
                        detail: ex.Message,
                        statusCode: 500,
                        title: "Retrieval failed");
                }
            })
            .WithName("GetItemById")
            .WithDescription("Retrieve a single item by document ID");

            // Get multiple items by IDs
            app.MapPost("/api/items/batch", (BatchGetRequest request, Kernel kern) =>
            {
                try
                {
                    if (request.Ids == null || request.Ids.Length == 0)
                    {
                        return Results.BadRequest(new { error = "IDs cannot be empty" });
                    }

                    if (request.Ids.Length > 100)
                    {
                        return Results.BadRequest(new { error = "Cannot retrieve more than 100 items at once" });
                    }

                    var items = request.Ids
                        .Select(id => kern.GetItem(id))
                        .Where(item => item != null)
                        .Select(item => new ItemResponse
                        {
                            Id = item!.Id,
                            SemanticKey = item.SemanticKey,
                            Payload = item.Payload,
                            HasEmbedding = item.Embedding != null && item.Embedding.Length > 0
                        })
                        .ToList();

                    return Results.Ok(new BatchGetResponse
                    {
                        Items = items,
                        RequestedCount = request.Ids.Length,
                        FoundCount = items.Count
                    });
                }
                catch (Exception ex)
                {
                    return Results.Problem(
                        detail: ex.Message,
                        statusCode: 500,
                        title: "Batch retrieval failed");
                }
            })
            .WithName("GetItemsBatch")
            .WithDescription("Retrieve multiple items by document IDs");

            // Health check endpoint
            app.MapGet("/api/health", () => Results.Ok(new
            {
                status = "healthy",
                service = "Viktor Search API",
                timestamp = DateTime.UtcNow
            }))
            .WithName("HealthCheck")
            .WithDescription("Health check endpoint");

            Console.WriteLine($"Starting Viktor Web Service on http://localhost:{port}");
            Console.WriteLine($"Swagger UI available at http://localhost:{port}");
            Console.WriteLine($"Index contains {kernel.Index.Items.Count} items");
            Console.WriteLine("Press Ctrl+C to stop the server");

            await app.RunAsync();
        }

        // Request/Response DTOs
        private record SearchRequest
        {
            public string Query { get; init; } = string.Empty;
            public int? TopK { get; init; }
            public bool? UseBinary { get; init; }
            public int? BinaryTopK { get; init; }
        }

        private record SearchByEmbeddingRequest
        {
            public float[] Embedding { get; init; } = Array.Empty<float>();
            public int? TopK { get; init; }
            public bool? UseBinary { get; init; }
            public int? BinaryTopK { get; init; }
        }

        private record SearchResponse
        {
            public string Query { get; init; } = string.Empty;
            public List<SearchResultItem> Results { get; init; } = new();
            public int Count { get; init; }
        }

        private record SearchResultItem
        {
            public int Id { get; init; }
            public string SemanticKey { get; init; } = string.Empty;
            public string Payload { get; init; } = string.Empty;
        }

        private record ItemResponse
        {
            public int Id { get; init; }
            public string SemanticKey { get; init; } = string.Empty;
            public string Payload { get; init; } = string.Empty;
            public bool HasEmbedding { get; init; }
        }

        private record BatchGetRequest
        {
            public int[] Ids { get; init; } = Array.Empty<int>();
        }

        private record BatchGetResponse
        {
            public List<ItemResponse> Items { get; init; } = new();
            public int RequestedCount { get; init; }
            public int FoundCount { get; init; }
        }
    }
}