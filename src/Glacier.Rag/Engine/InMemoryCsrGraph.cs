namespace Glacier.Rag.Engine;

using System;
using System.Collections.Generic;
using Glacier.Graph.Storage;

/// <summary>
/// Immutable, managed Compressed Sparse Row (CSR) graph snapshot compiled from a
/// <see cref="GraphStore"/> using public read-locked APIs. Eliminates disk temp-file
/// I/O, software page-cache fallback, and concurrent disposal races on the query hot path.
/// </summary>
internal sealed class InMemoryCsrGraph
{
    public static InMemoryCsrGraph Empty { get; } = new(
        nodeCount: 0,
        edgeCount: 0,
        rowOffsets: [0, 0],
        columnIndices: [],
        edgeWeights: [],
        externalToInternalId: new Dictionary<string, int>(0, StringComparer.Ordinal));

    public int NodeCount { get; }
    public int EdgeCount { get; }
    public int[] RowOffsets { get; }
    public int[] ColumnIndices { get; }
    public float[] EdgeWeights { get; }
    public IReadOnlyDictionary<string, int> ExternalToInternalId => _externalToInternalId;
    internal IReadOnlyDictionary<string, List<int>> ExternalToInternalIdsIgnoreCase => _externalToInternalIdsIgnoreCase;

    private readonly Dictionary<string, int> _externalToInternalId;
    private readonly Dictionary<string, List<int>> _externalToInternalIdsIgnoreCase;

    public InMemoryCsrGraph(
        int nodeCount,
        int edgeCount,
        int[] rowOffsets,
        int[] columnIndices,
        float[] edgeWeights,
        Dictionary<string, int> externalToInternalId)
    {
        NodeCount = nodeCount;
        EdgeCount = edgeCount;
        RowOffsets = rowOffsets;
        ColumnIndices = columnIndices;
        EdgeWeights = edgeWeights;
        _externalToInternalId = externalToInternalId;
        _externalToInternalIdsIgnoreCase = new Dictionary<string, List<int>>(
            externalToInternalId.Count,
            StringComparer.OrdinalIgnoreCase);
        foreach (var kv in externalToInternalId)
        {
            if (!_externalToInternalIdsIgnoreCase.TryGetValue(kv.Key, out var ids))
            {
                ids = new List<int>(1);
                _externalToInternalIdsIgnoreCase[kv.Key] = ids;
            }
            ids.Add(kv.Value);
        }
    }

    public bool TryGetInternalId(string externalId, out int internalId)
    {
        if (_externalToInternalId.TryGetValue(externalId, out internalId))
        {
            return true;
        }

        if (_externalToInternalIdsIgnoreCase.TryGetValue(externalId, out var ids) && ids.Count > 0)
        {
            internalId = ids[0];
            return true;
        }

        internalId = 0;
        return false;
    }

    public static InMemoryCsrGraph CompileFromForwardStar(GraphStore source)
    {
        ArgumentNullException.ThrowIfNull(source);

        using (source.AcquireReadLock())
        {
            int nodeCount = source.NodeCount;
            if (nodeCount <= 0)
            {
                return Empty;
            }

            int edgeCount = source.EdgeCount;
            var externalToInternalId = new Dictionary<string, int>(nodeCount, StringComparer.Ordinal);
            for (int i = 1; i <= nodeCount; i++)
            {
                string ext = source.GetExternalId(i);
                if (!string.IsNullOrEmpty(ext))
                {
                    externalToInternalId.TryAdd(ext, i);
                }
            }

            var rowOffsets = new int[nodeCount + 2];
            var columnIndices = edgeCount > 0 ? new int[edgeCount] : [];
            var edgeWeights = edgeCount > 0 ? new float[edgeCount] : [];

            int runningOffset = 0;
            var perNodeEdges = new List<(int TargetNodeId, int RelationId, float Weight)>(8);

            for (int i = 1; i <= nodeCount; i++)
            {
                rowOffsets[i] = runningOffset;
                perNodeEdges.Clear();

                var edges = source.GetOutwardEdgesByInternalId(i);
                while (edges.MoveNext())
                {
                    perNodeEdges.Add((edges.CurrentTargetNodeId, edges.CurrentRelationId, edges.CurrentWeight));
                    edges.Advance();
                }

                if (perNodeEdges.Count > 1)
                {
                    perNodeEdges.Sort(static (a, b) =>
                    {
                        int cmp = a.TargetNodeId.CompareTo(b.TargetNodeId);
                        if (cmp != 0) return cmp;
                        cmp = a.RelationId.CompareTo(b.RelationId);
                        return cmp != 0 ? cmp : a.Weight.CompareTo(b.Weight);
                    });
                }

                for (int k = 0; k < perNodeEdges.Count; k++)
                {
                    if (runningOffset >= columnIndices.Length)
                    {
                        int newLen = Math.Max(runningOffset + 1, columnIndices.Length == 0 ? 16 : columnIndices.Length * 2);
                        Array.Resize(ref columnIndices, newLen);
                        Array.Resize(ref edgeWeights, newLen);
                    }

                    columnIndices[runningOffset] = perNodeEdges[k].TargetNodeId;
                    edgeWeights[runningOffset] = perNodeEdges[k].Weight;
                    runningOffset++;
                }
            }

            rowOffsets[nodeCount + 1] = runningOffset;

            if (runningOffset < columnIndices.Length)
            {
                Array.Resize(ref columnIndices, runningOffset);
                Array.Resize(ref edgeWeights, runningOffset);
            }

            return new InMemoryCsrGraph(
                nodeCount,
                runningOffset,
                rowOffsets,
                columnIndices,
                edgeWeights,
                externalToInternalId);
        }
    }
}
