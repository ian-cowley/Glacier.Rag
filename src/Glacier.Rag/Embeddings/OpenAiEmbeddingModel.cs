namespace Glacier.Rag.Embeddings;

using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

internal sealed class OpenAiEmbeddingRequest
{
    [JsonPropertyName("model")]
    public string Model { get; set; } = string.Empty;

    [JsonPropertyName("input")]
    public string Input { get; set; } = string.Empty;
}

internal sealed class OpenAiEmbeddingResponse
{
    [JsonPropertyName("data")]
    public List<OpenAiEmbeddingData>? Data { get; set; }
}

internal sealed class OpenAiEmbeddingData
{
    [JsonPropertyName("embedding")]
    public float[]? Embedding { get; set; }
}

[JsonSourceGenerationOptions(PropertyNameCaseInsensitive = true)]
[JsonSerializable(typeof(OpenAiEmbeddingRequest))]
[JsonSerializable(typeof(OpenAiEmbeddingResponse))]
internal partial class OpenAiEmbeddingJsonContext : JsonSerializerContext
{
}

/// <summary>
/// NativeAOT-compatible <see cref="IEmbeddingModel"/> that delegates vector embedding generation
/// to any OpenAI-compatible <c>POST /v1/embeddings</c> endpoint — including local AMD XDNA 2 NPU
/// sidecar servers (FastFlowLM <c>flm serve</c> with EmbeddingGemma or Lemonade Server) so RAG
/// indexing can run at low wattage with zero GPU VRAM overhead.
/// </summary>
public sealed class OpenAiEmbeddingModel : IEmbeddingModel
{
    private const string DimensionProbeText = "dimension probe";

    private readonly HttpClient _http;
    private readonly bool _ownsHttpClient;
    private readonly string _embeddingsUrl;
    private readonly string _modelId;
    private readonly string? _apiKey;
    private bool _disposed;

    public int Dimensions { get; }

    public string BaseUrl { get; }

    public string ModelId => _modelId;

    /// <summary>
    /// Initializes an OpenAI-compatible embedding client.
    /// </summary>
    /// <param name="baseUrl">Base server URL (e.g., <c>http://127.0.0.1:52625</c> for FastFlowLM or <c>http://127.0.0.1:8000</c> for Lemonade).</param>
    /// <param name="modelId">Model identifier served by the endpoint (e.g., <c>embeddinggemma</c>).</param>
    /// <param name="dimensions">Expected embedding vector dimensionality (e.g., 768).</param>
    /// <param name="apiKey">Optional bearer token for authenticated endpoints (leave null for local NPU sidecars).</param>
    /// <param name="httpClient">Optional caller-managed <see cref="HttpClient"/>.</param>
    /// <param name="timeout">Optional request timeout when creating an internal <see cref="HttpClient"/> (defaults to 30 seconds).</param>
    public OpenAiEmbeddingModel(
        string baseUrl,
        string modelId,
        int dimensions,
        string? apiKey = null,
        HttpClient? httpClient = null,
        TimeSpan? timeout = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        if (dimensions <= 0)
            throw new ArgumentOutOfRangeException(nameof(dimensions), "Embedding dimensions must be positive.");
        if (timeout.HasValue && timeout.Value <= TimeSpan.Zero && timeout.Value != Timeout.InfiniteTimeSpan)
            throw new ArgumentOutOfRangeException(nameof(timeout), "Timeout must be positive.");

        string normalized = NormalizeBaseUrl(baseUrl);

        BaseUrl = normalized;
        _embeddingsUrl = BuildEmbeddingsUrl(normalized);
        _modelId = modelId.Trim();
        _apiKey = string.IsNullOrWhiteSpace(apiKey) ? null : apiKey.Trim();
        Dimensions = dimensions;

        if (httpClient != null)
        {
            _http = httpClient;
            _ownsHttpClient = false;
        }
        else
        {
            _http = new HttpClient { Timeout = timeout ?? TimeSpan.FromSeconds(30) };
            _ownsHttpClient = true;
        }
    }

    /// <summary>
    /// Probes an OpenAI-compatible <c>POST /v1/embeddings</c> endpoint using a short probe input
    /// to discover the embedding dimensionality produced by <paramref name="modelId"/>.
    /// </summary>
    public static async Task<int> ProbeDimensionsAsync(
        string baseUrl,
        string modelId,
        string? apiKey = null,
        HttpClient? httpClient = null,
        TimeSpan? timeout = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        string normalized = NormalizeBaseUrl(baseUrl);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        if (timeout.HasValue && timeout.Value <= TimeSpan.Zero && timeout.Value != Timeout.InfiniteTimeSpan)
            throw new ArgumentOutOfRangeException(nameof(timeout), "Timeout must be positive.");

        string embeddingsUrl = BuildEmbeddingsUrl(normalized);
        string trimmedModelId = modelId.Trim();
        string? normalizedKey = string.IsNullOrWhiteSpace(apiKey) ? null : apiKey.Trim();

        bool ownsClient = httpClient == null;
        HttpClient client = httpClient ?? new HttpClient { Timeout = timeout ?? TimeSpan.FromSeconds(30) };

        CancellationTokenSource? timeoutCts = null;
        CancellationToken effectiveToken = cancellationToken;
        if (timeout.HasValue && timeout.Value != Timeout.InfiniteTimeSpan)
        {
            timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeoutCts.CancelAfter(timeout.Value);
            effectiveToken = timeoutCts.Token;
        }

        try
        {
            float[] raw = await FetchRawEmbeddingAsync(
                client,
                embeddingsUrl,
                trimmedModelId,
                normalizedKey,
                DimensionProbeText,
                effectiveToken).ConfigureAwait(false);

            return raw.Length;
        }
        catch (OperationCanceledException) when (timeoutCts is { IsCancellationRequested: true } && !cancellationToken.IsCancellationRequested)
        {
            throw new TaskCanceledException(
                $"Embedding dimension probe to '{embeddingsUrl}' timed out after {timeout!.Value.TotalMilliseconds:F0} ms.",
                new TimeoutException());
        }
        finally
        {
            timeoutCts?.Dispose();
            if (ownsClient)
                client.Dispose();
        }
    }

    /// <summary>
    /// Probes the target endpoint to discover embedding dimensionality (unless <paramref name="knownDimensions"/> is supplied)
    /// and constructs a configured <see cref="OpenAiEmbeddingModel"/>.
    /// </summary>
    /// <remarks>
    /// <paramref name="timeout"/> governs the <see cref="ProbeDimensionsAsync"/> request and the internal
    /// <see cref="HttpClient"/> created when <paramref name="httpClient"/> is <see langword="null"/>; caller-supplied
    /// <see cref="HttpClient"/> instances retain their own configured timeout for subsequent embedding calls.
    /// </remarks>
    public static async Task<OpenAiEmbeddingModel> CreateAutoAsync(
        string baseUrl,
        string modelId,
        string? apiKey = null,
        HttpClient? httpClient = null,
        TimeSpan? timeout = null,
        int? knownDimensions = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        string normalized = NormalizeBaseUrl(baseUrl);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        if (timeout.HasValue && timeout.Value <= TimeSpan.Zero && timeout.Value != Timeout.InfiniteTimeSpan)
            throw new ArgumentOutOfRangeException(nameof(timeout), "Timeout must be positive.");
        if (knownDimensions is <= 0)
            throw new ArgumentOutOfRangeException(nameof(knownDimensions), "Embedding dimensions must be positive.");

        int dimensions = knownDimensions
            ?? await ProbeDimensionsAsync(normalized, modelId, apiKey, httpClient, timeout, cancellationToken).ConfigureAwait(false);

        return new OpenAiEmbeddingModel(normalized, modelId, dimensions, apiKey, httpClient, timeout);
    }

    /// <inheritdoc />
    public void GenerateEmbedding(ReadOnlySpan<char> text, Span<float> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (destination.Length < Dimensions)
            throw new ArgumentException($"Destination span length ({destination.Length}) is smaller than Dimensions ({Dimensions}).", nameof(destination));

        string textStr = text.ToString();
        float[] vector = SynchronizationContext.Current is not null
            ? Task.Run(() => GenerateEmbeddingAsync(textStr, CancellationToken.None)).GetAwaiter().GetResult()
            : GenerateEmbeddingAsync(textStr, CancellationToken.None).GetAwaiter().GetResult();
        vector.AsSpan(0, Dimensions).CopyTo(destination);
    }

    /// <summary>
    /// Asynchronously fetches and L2-normalizes an embedding vector from the configured OpenAI-compatible endpoint.
    /// </summary>
    public async Task<float[]> GenerateEmbeddingAsync(string text, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(text);

        float[] raw = await FetchRawEmbeddingAsync(
            _http,
            _embeddingsUrl,
            _modelId,
            _apiKey,
            text,
            cancellationToken).ConfigureAwait(false);

        if (raw.Length != Dimensions)
        {
            throw new InvalidOperationException(
                $"Embedding endpoint '{_embeddingsUrl}' returned {raw.Length} dimensions, expected {Dimensions}.");
        }

        NormalizeL2InPlace(raw);
        return raw;
    }

    private static string NormalizeBaseUrl(string baseUrl)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(baseUrl);
        string trimmed = baseUrl.Trim();
        int suffixIdx = trimmed.IndexOfAny(['?', '#']);
        string pathPart = (suffixIdx >= 0 ? trimmed[..suffixIdx] : trimmed).TrimEnd('/');
        string suffix = suffixIdx >= 0 ? trimmed[suffixIdx..] : string.Empty;

        if (pathPart.EndsWith("/v1", StringComparison.OrdinalIgnoreCase))
            pathPart = pathPart[..^3].TrimEnd('/');

        if (string.IsNullOrWhiteSpace(pathPart))
            throw new ArgumentException("Base URL cannot be empty after normalizing trailing '/v1'.", nameof(baseUrl));

        return string.Concat(pathPart, suffix);
    }

    private static string BuildEmbeddingsUrl(string normalizedBaseUrl)
    {
        int suffixIdx = normalizedBaseUrl.IndexOfAny(['?', '#']);
        string pathPart = (suffixIdx >= 0 ? normalizedBaseUrl[..suffixIdx] : normalizedBaseUrl).TrimEnd('/');
        string suffix = suffixIdx >= 0 ? normalizedBaseUrl[suffixIdx..] : string.Empty;
        return $"{pathPart}/v1/embeddings{suffix}";
    }

    private static async Task<float[]> FetchRawEmbeddingAsync(
        HttpClient httpClient,
        string embeddingsUrl,
        string modelId,
        string? apiKey,
        string input,
        CancellationToken cancellationToken)
    {
        var payload = new OpenAiEmbeddingRequest
        {
            Model = modelId,
            Input = input
        };

        string requestJson = JsonSerializer.Serialize(payload, OpenAiEmbeddingJsonContext.Default.OpenAiEmbeddingRequest);
        using var request = new HttpRequestMessage(HttpMethod.Post, embeddingsUrl)
        {
            Content = new StringContent(requestJson, Encoding.UTF8, "application/json")
        };

        if (apiKey != null)
        {
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);
        }

        using var response = await httpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);
        string body = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
        {
            throw new HttpRequestException(
                $"Embedding endpoint '{embeddingsUrl}' returned HTTP {(int)response.StatusCode}: {body}",
                null,
                response.StatusCode);
        }

        var parsed = JsonSerializer.Deserialize(body, OpenAiEmbeddingJsonContext.Default.OpenAiEmbeddingResponse);
        float[]? raw = parsed?.Data is { Count: > 0 } ? parsed.Data[0]?.Embedding : null;
        if (raw == null || raw.Length == 0)
        {
            throw new InvalidOperationException(
                $"Embedding endpoint '{embeddingsUrl}' returned an empty embedding payload.");
        }

        for (int i = 0; i < raw.Length; i++)
        {
            if (!float.IsFinite(raw[i]))
            {
                throw new InvalidOperationException(
                    $"Embedding endpoint '{embeddingsUrl}' returned non-finite value at index {i}.");
            }
        }

        return raw;
    }

    private static void NormalizeL2InPlace(float[] vec)
    {
        double sumSq = 0.0;
        for (int i = 0; i < vec.Length; i++)
        {
            double v = vec[i];
            sumSq += v * v;
        }

        if (sumSq > 1e-12)
        {
            float invNorm = (float)(1.0 / Math.Sqrt(sumSq));
            for (int i = 0; i < vec.Length; i++)
            {
                vec[i] *= invNorm;
            }
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_ownsHttpClient)
            _http.Dispose();
    }
}
