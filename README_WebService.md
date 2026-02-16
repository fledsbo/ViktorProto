# Viktor Web Service API

A high-performance web service for semantic search using approximate nearest neighbor (ANN) search on embeddings.

## Starting the Web Service

```bash
# Start on default port 5000
dotnet run -- serve

# Start on custom port
dotnet run -- serve 8080
```

The service will start with:
- **API**: `http://localhost:5000`
- **Swagger UI**: `http://localhost:5000` (interactive API documentation)

## API Endpoints

### 1. Search by Text Query (`POST /api/search`)

Performs semantic search using Azure OpenAI embeddings.

**Request:**
```json
{
  "query": "cute cat videos",
  "topK": 10,
  "useBinary": false,
  "binaryTopK": 100
}
```

**Parameters:**
- `query` (required): Text query to search for
- `topK` (optional): Number of results to return (default: 10, max: 100)
- `useBinary` (optional): Use binary quantization for faster search (default: false)
- `binaryTopK` (optional): Number of candidates for binary search reordering (default: 100)

**Response:**
```json
{
  "query": "cute cat videos",
  "results": [
    {
      "id": 1,
      "semanticKey": "cat playing with yarn",
      "payload": "video:123:cat playing with yarn"
    }
  ],
  "count": 1
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "cute cat videos", "topK": 5}'
```

### 2. Search by Embedding (`POST /api/search/embedding`)

Performs search using a pre-computed embedding vector.

**Request:**
```json
{
  "embedding": [0.1, 0.2, 0.3, ...],
  "topK": 10,
  "useBinary": false,
  "binaryTopK": 100
}
```

**Parameters:**
- `embedding` (required): Float array of embedding values (1536 dimensions for text-embedding-ada-002)
- `topK` (optional): Number of results to return (default: 10, max: 100)
- `useBinary` (optional): Use binary quantization for faster search
- `binaryTopK` (optional): Number of candidates for binary search reordering

**Response:** Same format as search by text

**Example:**
```bash
curl -X POST http://localhost:5000/api/search/embedding \
  -H "Content-Type: application/json" \
  -d '{"embedding": [0.1, 0.2, ...], "topK": 5}'
```

### 3. Get Item by ID (`GET /api/items/{id}`)

Retrieves a single record by document ID.

**Response:**
```json
{
  "id": 1,
  "semanticKey": "cat playing with yarn",
  "payload": "video:123:cat playing with yarn",
  "hasEmbedding": true
}
```

**Example:**
```bash
curl http://localhost:5000/api/items/1
```

### 4. Get Multiple Items (`POST /api/items/batch`)

Retrieves multiple records by document IDs.

**Request:**
```json
{
  "ids": [1, 2, 3, 10, 15]
}
```

**Parameters:**
- `ids` (required): Array of document IDs (max 100 IDs per request)

**Response:**
```json
{
  "items": [
    {
      "id": 1,
      "semanticKey": "cat playing with yarn",
      "payload": "video:123:cat playing with yarn",
      "hasEmbedding": true
    }
  ],
  "requestedCount": 5,
  "foundCount": 1
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/items/batch \
  -H "Content-Type: application/json" \
  -d '{"ids": [1, 2, 3, 10, 15]}'
```

### 5. Health Check (`GET /api/health`)

Checks if the service is running.

**Response:**
```json
{
  "status": "healthy",
  "service": "Viktor Search API",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Example:**
```bash
curl http://localhost:5000/api/health
```

## Performance Tips

### Binary Quantization

For faster search on large datasets, use binary quantization with reordering:

```json
{
  "query": "your search query",
  "topK": 10,
  "useBinary": true,
  "binaryTopK": 50
}
```

**How it works:**
1. Uses fast binary quantization to get top `binaryTopK` candidates (e.g., 50)
2. Reorders those candidates using full precision to get final `topK` results (e.g., 10)
3. Much faster than full precision search on all items
4. Typical accuracy: 90%+ recall on top-10

**When to use:**
- Large datasets (1000+ items)
- Latency-sensitive applications
- Can tolerate slight accuracy tradeoff

### SIMD Optimization

The search uses SIMD vectorization automatically when available:
- 2-8x faster cosine similarity calculations
- Enabled by default on modern CPUs
- No configuration needed

## Complete Workflow Example

```bash
# 1. Load data into the index
dotnet run -- inputfile data.txt

# 2. Start the web service
dotnet run -- serve 5000

# 3. Search for items (in another terminal)
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning tutorials", "topK": 5}'

# 4. Get a specific item
curl http://localhost:5000/api/items/42

# 5. Get multiple items
curl -X POST http://localhost:5000/api/items/batch \
  -H "Content-Type: application/json" \
  -d '{"ids": [1, 5, 10, 15, 20]}'
```

## Configuration

Edit `appsettings.json` to configure Azure OpenAI:

```json
{
  "AzureOpenAI": {
    "Endpoint": "https://your-resource.openai.azure.com/",
    "ApiKey": "your-api-key",
    "DeploymentName": "text-embedding-ada-002"
  }
}
```

## Features

? **Fast ANN Search**: SIMD-optimized cosine similarity  
? **Binary Quantization**: 10x+ speedup with minimal accuracy loss  
? **Batch Retrieval**: Get multiple items in one request  
? **Swagger UI**: Interactive API documentation  
? **Health Checks**: Monitor service status  
? **Azure OpenAI**: State-of-the-art embeddings  

## Architecture

- **ASP.NET Core Minimal APIs**: Modern, high-performance web framework
- **In-Memory Index**: Lightning-fast search (no disk I/O during queries)
- **Protobuf Serialization**: Compact binary storage format
- **SIMD Vectorization**: Hardware-accelerated computations
- **Binary Quantization**: Optional fast approximate search

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200 OK`: Successful request
- `400 Bad Request`: Invalid parameters
- `404 Not Found`: Item not found
- `500 Internal Server Error`: Server error (check logs)

Error response format:
```json
{
  "error": "Description of what went wrong"
}
```
