namespace Glacier.Rag.Tests;

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Rag.Embeddings;
using Glacier.Rag.Engine;
using Xunit;

public sealed class OpenAiEmbeddingModelAutoProbeTests
{
    private sealed class StubEmbeddingHandler : HttpMessageHandler
    {
        private readonly Func<HttpRequestMessage, string, CancellationToken, Task<HttpResponseMessage>> _responder;
        public List<(Uri Uri, HttpMethod Method, string? AuthScheme, string? AuthParam, string Body)> Requests { get; } = [];
        public bool WasDisposed { get; private set; }

        public StubEmbeddingHandler(Func<HttpRequestMessage, string, CancellationToken, Task<HttpResponseMessage>> responder)
        {
            _responder = responder;
        }

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            string body = request.Content != null
                ? await request.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false)
                : string.Empty;

            Requests.Add((
                request.RequestUri!,
                request.Method,
                request.Headers.Authorization?.Scheme,
                request.Headers.Authorization?.Parameter,
                body));

            return await _responder(request, body, cancellationToken).ConfigureAwait(false);
        }

        protected override void Dispose(bool disposing)
        {
            WasDisposed = true;
            base.Dispose(disposing);
        }
    }

    private static HttpResponseMessage BuildEmbeddingJsonResponse(int dimensions, float startValue = 1.0f)
    {
        var sb = new StringBuilder("{\"object\":\"list\",\"data\":[{\"index\":0,\"embedding\":[");
        for (int i = 0; i < dimensions; i++)
        {
            if (i > 0) sb.Append(',');
            float val = startValue + (i % 7) * 0.25f;
            sb.Append(val.ToString("G", CultureInfo.InvariantCulture));
        }
        sb.Append("]}]}");
        return new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent(sb.ToString(), Encoding.UTF8, "application/json")
        };
    }

    [Theory]
    [InlineData(384)]
    [InlineData(768)]
    [InlineData(1536)]
    public async Task ProbeDimensionsAsync_ReturnsVectorLength_AndSendsDimensionProbeInput(int expectedDims)
    {
        using var handler = new StubEmbeddingHandler((_, _, _) =>
            Task.FromResult(BuildEmbeddingJsonResponse(expectedDims)));
        using var client = new HttpClient(handler);

        int dims = await OpenAiEmbeddingModel.ProbeDimensionsAsync(
            "http://127.0.0.1:52625",
            "embeddinggemma",
            httpClient: client);

        Assert.Equal(expectedDims, dims);
        Assert.Single(handler.Requests);
        var req = handler.Requests[0];
        Assert.Equal(HttpMethod.Post, req.Method);
        Assert.Equal("http://127.0.0.1:52625/v1/embeddings", req.Uri.ToString());

        using var doc = JsonDocument.Parse(req.Body);
        Assert.Equal("embeddinggemma", doc.RootElement.GetProperty("model").GetString());
        Assert.Equal("dimension probe", doc.RootElement.GetProperty("input").GetString());
    }

    [Theory]
    [InlineData("http://127.0.0.1:52625")]
    [InlineData("http://127.0.0.1:52625/")]
    [InlineData("http://127.0.0.1:52625/v1")]
    [InlineData("http://127.0.0.1:52625/V1/")]
    public async Task NormalizeBaseUrl_StripsTrailingSlashAndV1_AcrossProbeAndCreateAuto(string rawBaseUrl)
    {
        using var handler = new StubEmbeddingHandler((_, _, _) =>
            Task.FromResult(BuildEmbeddingJsonResponse(768)));
        using var client = new HttpClient(handler);

        using var model = await OpenAiEmbeddingModel.CreateAutoAsync(
            rawBaseUrl,
            "embeddinggemma",
            httpClient: client);

        Assert.Equal("http://127.0.0.1:52625", model.BaseUrl);
        Assert.Equal(768, model.Dimensions);
        Assert.Single(handler.Requests);
        Assert.Equal("http://127.0.0.1:52625/v1/embeddings", handler.Requests[0].Uri.ToString());
    }

    [Fact]
    public async Task ProbeDimensionsAsync_AuthHeader_SentOnlyWhenNonWhitespaceApiKeyProvided()
    {
        using var handler = new StubEmbeddingHandler((_, _, _) =>
            Task.FromResult(BuildEmbeddingJsonResponse(384)));
        using var client = new HttpClient(handler);

        await OpenAiEmbeddingModel.ProbeDimensionsAsync("http://127.0.0.1:52625", "m1", apiKey: "   ", httpClient: client);
        await OpenAiEmbeddingModel.ProbeDimensionsAsync("http://127.0.0.1:52625", "m1", apiKey: " sk-test-token ", httpClient: client);

        Assert.Equal(2, handler.Requests.Count);
        Assert.Null(handler.Requests[0].AuthScheme);
        Assert.Equal("Bearer", handler.Requests[1].AuthScheme);
        Assert.Equal("sk-test-token", handler.Requests[1].AuthParam);
    }

    [Fact]
    public async Task ProbeDimensionsAsync_AndCreateAutoAsync_ValidateArguments()
    {
        await Assert.ThrowsAnyAsync<ArgumentException>(() =>
            OpenAiEmbeddingModel.ProbeDimensionsAsync("   ", "embeddinggemma"));
        await Assert.ThrowsAnyAsync<ArgumentException>(() =>
            OpenAiEmbeddingModel.ProbeDimensionsAsync("/v1", "embeddinggemma"));
        await Assert.ThrowsAnyAsync<ArgumentException>(() =>
            OpenAiEmbeddingModel.ProbeDimensionsAsync("http://127.0.0.1:52625", " "));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() =>
            OpenAiEmbeddingModel.ProbeDimensionsAsync("http://127.0.0.1:52625", "m", timeout: TimeSpan.Zero));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() =>
            OpenAiEmbeddingModel.CreateAutoAsync("http://127.0.0.1:52625", "m", knownDimensions: 0));
    }

    [Theory]
    [InlineData(HttpStatusCode.Unauthorized)]
    [InlineData(HttpStatusCode.NotFound)]
    [InlineData(HttpStatusCode.InternalServerError)]
    public async Task ProbeDimensionsAsync_NonSuccessStatus_ThrowsHttpRequestExceptionWithStatusCode(HttpStatusCode statusCode)
    {
        using var handler = new StubEmbeddingHandler((_, _, _) =>
            Task.FromResult(new HttpResponseMessage(statusCode)
            {
                Content = new StringContent("server error detail", Encoding.UTF8, "text/plain")
            }));
        using var client = new HttpClient(handler);

        var ex = await Assert.ThrowsAsync<HttpRequestException>(() =>
            OpenAiEmbeddingModel.ProbeDimensionsAsync("http://127.0.0.1:52625", "embeddinggemma", httpClient: client));

        Assert.Equal(statusCode, ex.StatusCode);
    }

    [Theory]
    [InlineData("{}")]
    [InlineData("{\"data\":[]}")]
    [InlineData("{\"data\":[{}]}")]
    [InlineData("{\"data\":[{\"embedding\":[]}]}")]
    public async Task ProbeDimensionsAsync_EmptyEmbeddingPayload_ThrowsInvalidOperationException(string json)
    {
        using var handler = new StubEmbeddingHandler((_, _, _) =>
            Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(json, Encoding.UTF8, "application/json")
            }));
        using var client = new HttpClient(handler);

        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            OpenAiEmbeddingModel.ProbeDimensionsAsync("http://127.0.0.1:52625", "embeddinggemma", httpClient: client));
    }

    [Fact]
    public async Task ProbeDimensionsAsync_MalformedJson_ThrowsJsonException()
    {
        using var handler = new StubEmbeddingHandler((_, _, _) =>
            Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent("<html>oops</html>", Encoding.UTF8, "text/html")
            }));
        using var client = new HttpClient(handler);

        await Assert.ThrowsAnyAsync<JsonException>(() =>
            OpenAiEmbeddingModel.ProbeDimensionsAsync("http://127.0.0.1:52625", "embeddinggemma", httpClient: client));
    }

    [Fact]
    public async Task ProbeDimensionsAsync_Timeout_ThrowsTaskCanceledWithTimeoutInner_WithoutMutatingCallerClient()
    {
        using var handler = new StubEmbeddingHandler(async (_, _, ct) =>
        {
            await Task.Delay(500, ct).ConfigureAwait(false);
            return BuildEmbeddingJsonResponse(384);
        });
        TimeSpan originalTimeout = TimeSpan.FromSeconds(42);
        using var client = new HttpClient(handler) { Timeout = originalTimeout };

        var ex = await Assert.ThrowsAsync<TaskCanceledException>(() =>
            OpenAiEmbeddingModel.ProbeDimensionsAsync(
                "http://127.0.0.1:52625",
                "embeddinggemma",
                httpClient: client,
                timeout: TimeSpan.FromMilliseconds(40)));

        Assert.IsType<TimeoutException>(ex.InnerException);
        Assert.Equal(originalTimeout, client.Timeout);
        Assert.False(handler.WasDisposed);
    }

    [Fact]
    public async Task ProbeDimensionsAsync_CallerCancellation_ThrowsOperationCanceledException()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        using var handler = new StubEmbeddingHandler((_, _, _) =>
            Task.FromResult(BuildEmbeddingJsonResponse(384)));
        using var client = new HttpClient(handler);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            OpenAiEmbeddingModel.ProbeDimensionsAsync(
                "http://127.0.0.1:52625",
                "embeddinggemma",
                httpClient: client,
                cancellationToken: cts.Token));
    }

    [Fact]
    public async Task CreateAutoAsync_ProbesOnce_ProducesL2NormalizedVectors_AndDoesNotDisposeCallerClient()
    {
        using var handler = new StubEmbeddingHandler((_, _, _) =>
            Task.FromResult(BuildEmbeddingJsonResponse(768, startValue: 2.0f)));
        using var client = new HttpClient(handler);

        OpenAiEmbeddingModel model = await OpenAiEmbeddingModel.CreateAutoAsync(
            "http://127.0.0.1:52625",
            "embeddinggemma",
            httpClient: client);

        Assert.Equal(768, model.Dimensions);
        Assert.Single(handler.Requests);

        float[] vec = await model.GenerateEmbeddingAsync("semantic NPU search");
        Assert.Equal(768, vec.Length);

        double norm = Math.Sqrt(vec.Sum(static v => (double)v * v));
        Assert.InRange(norm, 0.9999, 1.0001);

        var spanBuf = new float[768];
        model.GenerateEmbedding("sync bridge", spanBuf);
        double spanNorm = Math.Sqrt(spanBuf.Sum(static v => (double)v * v));
        Assert.InRange(spanNorm, 0.9999, 1.0001);

        model.Dispose();
        Assert.False(handler.WasDisposed);
    }

    [Fact]
    public async Task CreateAutoAsync_KnownDimensions_SkipsProbeRequest()
    {
        using var handler = new StubEmbeddingHandler((_, _, _) =>
            Task.FromResult(BuildEmbeddingJsonResponse(384)));
        using var client = new HttpClient(handler);

        using var model = await OpenAiEmbeddingModel.CreateAutoAsync(
            "http://127.0.0.1:52625",
            "all-minilm",
            httpClient: client,
            knownDimensions: 384);

        Assert.Equal(384, model.Dimensions);
        Assert.Empty(handler.Requests);

        float[] vec = await model.GenerateEmbeddingAsync("hello");
        Assert.Equal(384, vec.Length);
        Assert.Single(handler.Requests);
    }

    [Fact]
    public async Task CreateAutoAsync_WithGraphRagEngine_IndexesAndRetrievesAt768Dimensions()
    {
        using var handler = new StubEmbeddingHandler((_, body, _) =>
        {
            float seed = body.Contains("XDNA2", StringComparison.OrdinalIgnoreCase) ? 3.0f : 1.0f;
            return Task.FromResult(BuildEmbeddingJsonResponse(768, startValue: seed));
        });
        using var client = new HttpClient(handler);

        using var model = await OpenAiEmbeddingModel.CreateAutoAsync(
            "http://127.0.0.1:52625",
            "embeddinggemma",
            httpClient: client);

        using var rag = new GraphRagEngine(model);
        rag.IndexDocument("doc-npu", "AMD Ryzen AI XDNA2 NPU runs EmbeddingGemma at low wattage.");
        rag.IndexDocument("doc-cpu", "CPU baseline notes.");

        var results = rag.Retrieve("XDNA2 EmbeddingGemma");
        Assert.NotEmpty(results.VectorMatches);
        Assert.Contains("EmbeddingGemma", results.SynthesizedContext, StringComparison.Ordinal);
    }
}
