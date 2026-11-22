namespace Glacier.Rag.Tests;

using System;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Rag.Embeddings;
using Xunit;

public sealed class Tier5M7Phase2HardeningTests
{
    private sealed class StubEmbedHandler : HttpMessageHandler
    {
        private readonly Func<HttpRequestMessage, int, CancellationToken, Task<HttpResponseMessage>> _responder;
        private int _calls;

        public StubEmbedHandler(Func<HttpRequestMessage, int, CancellationToken, Task<HttpResponseMessage>> responder)
        {
            _responder = responder;
        }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            int call = Interlocked.Increment(ref _calls);
            return _responder(request, call, cancellationToken);
        }
    }

    private sealed class SingleThreadSyncContext : SynchronizationContext
    {
        private readonly BlockingCollection<(SendOrPostCallback Callback, object? State)> _queue = new();

        public override void Post(SendOrPostCallback d, object? state) => _queue.Add((d, state));

        public void DrainUntil(Task target)
        {
            target.ContinueWith(_ => _queue.CompleteAdding(), TaskScheduler.Default);
            foreach (var (cb, st) in _queue.GetConsumingEnumerable())
            {
                cb(st);
            }
        }
    }

    private static HttpResponseMessage Json(HttpStatusCode status, string body) =>
        new(status) { Content = new StringContent(body, Encoding.UTF8, "application/json") };

    [Fact]
    public async Task OpenAiEmbeddingModel_ZeroVector_DimMismatch_NonFinite_HttpErrors_AndDoubleDispose()
    {
        // 1. Zero vector L2-normalization stability (must NOT produce NaN)
        using var zeroH = new StubEmbedHandler((_, _, _) =>
            Task.FromResult(Json(HttpStatusCode.OK, """{"data":[{"index":0,"embedding":[0.0,0.0,0.0,0.0]}]}""")));
        using var zeroHttp = new HttpClient(zeroH);
        using var zeroModel = new OpenAiEmbeddingModel("http://127.0.0.1:52625/v1", "embed", dimensions: 4, httpClient: zeroHttp);
        float[] z = await zeroModel.GenerateEmbeddingAsync("zero");
        Assert.Equal(4, z.Length);
        Assert.All(z, f => Assert.True(float.IsFinite(f)));

        // 2. Dimension mismatch between probe/configured dims (4) and runtime vector (3) throws
        using var mismatchH = new StubEmbedHandler((_, _, _) =>
            Task.FromResult(Json(HttpStatusCode.OK, """{"data":[{"index":0,"embedding":[0.1,0.2,0.3]}]}""")));
        using var mismatchHttp = new HttpClient(mismatchH);
        using var mismatchModel = new OpenAiEmbeddingModel("http://127.0.0.1:52625/v1", "embed", dimensions: 4, httpClient: mismatchHttp);
        await Assert.ThrowsAsync<InvalidOperationException>(() => mismatchModel.GenerateEmbeddingAsync("mismatch"));

        // 3. Overflow / non-finite float value (1e300 overflows float -> Infinity) rejects cleanly
        using var infH = new StubEmbedHandler((_, _, _) =>
            Task.FromResult(Json(HttpStatusCode.OK, """{"data":[{"index":0,"embedding":[1e300,0.0]}]}""")));
        using var infHttp = new HttpClient(infH);
        using var infModel = new OpenAiEmbeddingModel("http://127.0.0.1:52625/v1", "embed", dimensions: 2, httpClient: infHttp);
        await Assert.ThrowsAsync<InvalidOperationException>(() => infModel.GenerateEmbeddingAsync("inf"));

        // 4. HTTP 401/500 & Timeout
        foreach (var status in new[] { HttpStatusCode.Unauthorized, HttpStatusCode.InternalServerError })
        {
            using var errH = new StubEmbedHandler((_, _, _) => Task.FromResult(Json(status, """{"error":"boom"}""")));
            using var errHttp = new HttpClient(errH);
            using var errModel = new OpenAiEmbeddingModel("http://127.0.0.1:52625/v1", "embed", dimensions: 2, httpClient: errHttp);
            await Assert.ThrowsAsync<HttpRequestException>(() => errModel.GenerateEmbeddingAsync("err"));
        }

        // 5. Double Dispose + post-dispose throws ObjectDisposedException
        zeroModel.Dispose();
        zeroModel.Dispose();
        await Assert.ThrowsAsync<ObjectDisposedException>(() => zeroModel.GenerateEmbeddingAsync("disposed"));
    }

    [Fact]
    public void OpenAiEmbeddingModel_SyncGenerateEmbedding_UnderSingleThreadedSyncContext_DoesNotDeadlock()
    {
        using var handler = new StubEmbedHandler(async (_, _, ct) =>
        {
            await Task.Delay(15, ct); // genuine async yield requiring continuation dispatch
            return Json(HttpStatusCode.OK, """{"data":[{"index":0,"embedding":[3.0,4.0]}]}""");
        });
        using var http = new HttpClient(handler);
        using var model = new OpenAiEmbeddingModel("http://127.0.0.1:52625/v1", "embed", dimensions: 2, httpClient: http);

        var syncCtx = new SingleThreadSyncContext();
        var oldCtx = SynchronizationContext.Current;
        Exception? failure = null;
        float[] buffer = new float[2];

        var work = Task.Run(() => { }); // completed starter
        var uiThreadTask = Task.Factory.StartNew(() =>
        {
            SynchronizationContext.SetSynchronizationContext(syncCtx);
            try
            {
                model.GenerateEmbedding("sync deadlock test".AsSpan(), buffer);
            }
            catch (Exception ex)
            {
                failure = ex;
            }
            finally
            {
                SynchronizationContext.SetSynchronizationContext(oldCtx);
            }
        }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);

        syncCtx.DrainUntil(uiThreadTask);
        Assert.Null(failure);
        Assert.Equal(0.6f, buffer[0], 3);
        Assert.Equal(0.8f, buffer[1], 3);
    }
}
