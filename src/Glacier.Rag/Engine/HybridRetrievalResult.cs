namespace Glacier.Rag.Engine;

using System.Collections.Generic;
using Glacier.Vector.Index;

public sealed class GraphRelation
{
    public required string Source { get; init; }
    public required string Target { get; init; }
    public required string Relation { get; init; }

    public override string ToString() => $"{Source} --[{Relation}]--> {Target}";
}

/// <summary>
/// Result of an in-process hybrid GraphRAG retrieval pass.
/// Combines dense semantic vector chunks with multi-hop knowledge graph relationship paths.
/// </summary>
public sealed class HybridRetrievalResult
{
    public required string Query { get; init; }
    public required IReadOnlyList<SearchResult> VectorMatches { get; init; }
    public required IReadOnlyList<string> DiscoveredEntities { get; init; }
    public required IReadOnlyList<GraphRelation> GraphRelations { get; init; }
    public required string SynthesizedContext { get; init; }
    public IReadOnlyList<HybridGraphScorer.ScoredChunk>? ScoredChunks { get; init; }
    public double VectorSearchLatencyMs { get; init; }
    public double GraphTraversalLatencyMs { get; init; }
    public double TotalRetrievalLatencyMs => VectorSearchLatencyMs + GraphTraversalLatencyMs;
}
