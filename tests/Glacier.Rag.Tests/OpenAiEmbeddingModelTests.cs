namespace Glacier.Rag.Tests;

using System;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Rag.Embeddings;
using Glacier.Rag.Engine;
using Xunit;

public sealed class OpenAiEmbeddingModelTests
{
    [Fact]
    public async Task GenerateEmbedding_CallsV1EmbeddingsAndNormalizesL2()
    {
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;

        string? capturedRequest = null;
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        Task serverTask = Task.Run(async () =>
        {
            using var client = await listener.AcceptTcpClientAsync(cts.Token);
            using var stream = client.GetStream();
            byte[] buf = new byte[4096];
            int n = await stream.ReadAsync(buf, cts.Token);
            capturedRequest = Encoding.UTF8.GetString(buf, 0, n);

            // Raw vector [3, 4, 0, 0] -> L2 normalized should be [0.6, 0.8, 0.0, 0.0]
            const string json = """{"object":"list","data":[{"object":"embedding","index":0,"embedding":[3.0,4.0,0.0,0.0]}]}""";
            string resp = $"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {Encoding.UTF8.GetByteCount(json)}\r\nConnection: close\r\n\r\n{json}";
            await stream.WriteAsync(Encoding.UTF8.GetBytes(resp), cts.Token);
            await stream.FlushAsync(cts.Token);
        }, cts.Token);

        using var model = new OpenAiEmbeddingModel($"http://127.0.0.1:{port}/v1", "embeddinggemma", dimensions: 4);
        float[] dest = new float[4];
        model.GenerateEmbedding("AMD XDNA 2 NPU sidecar", dest);

        await serverTask;

        Assert.NotNull(capturedRequest);
        Assert.Contains("POST /v1/embeddings", capturedRequest, StringComparison.Ordinal);
        Assert.Contains("embeddinggemma", capturedRequest, StringComparison.Ordinal);
        Assert.Equal(0.6f, dest[0], precision: 4);
        Assert.Equal(0.8f, dest[1], precision: 4);
        Assert.Equal(0.0f, dest[2], precision: 4);
        Assert.Equal(0.0f, dest[3], precision: 4);
    }

    [Fact]
    public async Task GenerateEmbedding_DimensionMismatch_ThrowsInvalidOperationException()
    {
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;

        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        Task serverTask = Task.Run(async () =>
        {
            using var client = await listener.AcceptTcpClientAsync(cts.Token);
            using var stream = client.GetStream();
            byte[] buf = new byte[2048];
            _ = await stream.ReadAsync(buf, cts.Token);

            const string json = """{"data":[{"embedding":[1.0,2.0]}]}""";
            string resp = $"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {Encoding.UTF8.GetByteCount(json)}\r\nConnection: close\r\n\r\n{json}";
            await stream.WriteAsync(Encoding.UTF8.GetBytes(resp), cts.Token);
            await stream.FlushAsync(cts.Token);
        }, cts.Token);

        using var model = new OpenAiEmbeddingModel($"http://127.0.0.1:{port}", "embeddinggemma", dimensions: 4);
        await Assert.ThrowsAsync<InvalidOperationException>(() => model.GenerateEmbeddingAsync("hello", cts.Token));
        await serverTask;
    }

    [Fact]
    public async Task GraphRagEngine_CanIndexAndRetrieveUsingOpenAiEmbeddingModel()
    {
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;

        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        Task serverTask = Task.Run(async () =>
        {
            // Handle 3 requests: 2 ingested documents + 1 query
            for (int r = 0; r < 3; r++)
            {
                using var client = await listener.AcceptTcpClientAsync(cts.Token);
                using var stream = client.GetStream();
                byte[] buf = new byte[4096];
                int n = await stream.ReadAsync(buf, cts.Token);
                string req = Encoding.UTF8.GetString(buf, 0, n);

                string vec = req.Contains("XDNA", StringComparison.OrdinalIgnoreCase)
                    ? "[1.0, 0.0, 0.0, 0.0]"
                    : "[0.0, 1.0, 0.0, 0.0]";
                string json = $$"""{"data":[{"embedding":{{vec}}}]}""";
                string resp = $"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {Encoding.UTF8.GetByteCount(json)}\r\nConnection: close\r\n\r\n{json}";
                await stream.WriteAsync(Encoding.UTF8.GetBytes(resp), cts.Token);
                await stream.FlushAsync(cts.Token);
            }
        }, cts.Token);

        using var embedder = new OpenAiEmbeddingModel($"http://127.0.0.1:{port}", "embeddinggemma", dimensions: 4);
        using var rag = new GraphRagEngine(embedder);

        rag.IndexDocument("doc-npu", "AMD XDNA 2 NPU runs EmbeddingGemma via FastFlowLM.");
        rag.IndexDocument("doc-other", "PostgreSQL indexing with B-tree structures.");

        var result = rag.Retrieve("XDNA NPU sidecar", new RagOptions { TopK = 1, MaxGraphHops = 1 });
        await serverTask;

        Assert.NotNull(result);
        Assert.NotEmpty(result.VectorMatches);
        Assert.Contains("FastFlowLM", result.SynthesizedContext, StringComparison.Ordinal);
    }
}
